"""
Site Decision Robustness Consolidation (PoC v2)

Consolidates site dispatch costs with RDM upgrade costs to evaluate
robustness of EB vs BB pathway decisions under uncertainty.

PoC v2 reporting overlay: introduces stylised uncertainty multipliers for site costs
to make site decision robustness non-trivial while keeping one-pass coupling intact.

Reads existing outputs only (no re-solving, no coupling, no dispatch re-runs).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List
from datetime import datetime
import shutil
import os
import time

# TOML loading
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Need tomllib (Python 3.11+) or tomli package")

from src.path_utils import repo_root, resolve_path, input_root


def safe_to_csv(
    df: pd.DataFrame,
    path: Path,
    retries: int = 5,
    backoff_s: List[float] = None
) -> None:
    """
    Atomically write DataFrame to CSV with retry logic for Windows file locking.
    
    Writes to a temporary file in the same directory, then atomically replaces
    the target file. Retries on PermissionError with exponential backoff.
    
    Args:
        df: DataFrame to write
        path: Target CSV path
        retries: Maximum number of retry attempts
        backoff_s: List of backoff delays in seconds (default: [0.2, 0.5, 1.0, 1.5, 2.0])
        
    Raises:
        PermissionError: If file appears locked after all retries
    """
    if backoff_s is None:
        backoff_s = [0.2, 0.5, 1.0, 1.5, 2.0]
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Temporary file in same directory
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    
    last_error = None
    for attempt in range(retries):
        try:
            # Clean up any existing temp file
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except (PermissionError, OSError):
                    pass  # Best effort cleanup
            
            # Write to temp file
            df.to_csv(tmp_path, index=False)
            
            # Atomically replace target
            os.replace(tmp_path, path)
            
            # Success
            return
            
        except PermissionError as e:
            last_error = e
            if attempt < retries - 1:
                delay = backoff_s[min(attempt, len(backoff_s) - 1)]
                time.sleep(delay)
                continue
            else:
                # Final attempt failed - clean up temp file if it exists
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except (PermissionError, OSError):
                        pass  # Best effort cleanup
                
                raise PermissionError(
                    f"File appears locked (Excel/preview). Close it and retry: {path}"
                ) from e
        except Exception as e:
            # Other errors - clean up temp file
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except (PermissionError, OSError):
                    pass  # Best effort cleanup
            raise


def find_config_file(config_path_override: Optional[str] = None) -> Path:
    """
    Find config file using search order:
    1. Override path (if provided)
    2. Input/configs/site_decision_robustness.toml
    3. configs/site_decision_robustness_min.toml
    4. configs/site_decision_robustness.toml
    """
    if config_path_override:
        return Path(resolve_path(config_path_override))
    
    # Search order
    search_paths = [
        Path(input_root()) / 'configs' / 'site_decision_robustness.toml',
        ROOT / 'configs' / 'site_decision_robustness_min.toml',
        ROOT / 'configs' / 'site_decision_robustness.toml',
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(
        f"Config file not found. Searched:\n" +
        "\n".join([f"  - {p}" for p in search_paths])
    )


def load_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load TOML configuration file with validation.
    
    Args:
        config_path: Optional override path (if None, uses find_config_file)
        
    Returns:
        Validated configuration dictionary
    """
    if config_path is None:
        config_path = find_config_file()
    elif not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    # Validate config structure
    validate_config(config)
    
    return config


def validate_config(config: Dict) -> None:
    """Validate configuration structure and values."""
    # Check multipliers section
    multipliers = config.get('multipliers', {})
    valid_distributions = {'uniform', 'triangular', 'normal'}
    
    for mult_name, mult_config in multipliers.items():
        dist_type = mult_config.get('distribution', 'triangular')
        if dist_type not in valid_distributions:
            raise ValueError(
                f"Invalid distribution '{dist_type}' for {mult_name}. "
                f"Must be one of: {valid_distributions}"
            )
        
        range_vals = mult_config.get('range', [])
        if not isinstance(range_vals, list) or len(range_vals) != 2:
            raise ValueError(
                f"Invalid range for {mult_name}: must be [min, max] list"
            )
        
        range_min, range_max = float(range_vals[0]), float(range_vals[1])
        if range_min >= range_max:
            raise ValueError(
                f"Invalid range for {mult_name}: min ({range_min}) must be < max ({range_max})"
            )
        
        if dist_type == 'triangular':
            mode = mult_config.get('mode')
            if mode is None:
                raise ValueError(
                    f"Triangular distribution for {mult_name} requires 'mode' parameter"
                )
            mode = float(mode)
            if not (range_min <= mode <= range_max):
                raise ValueError(
                    f"Mode ({mode}) for {mult_name} must be within range [{range_min}, {range_max}]"
                )
    
    # Check generation section
    generation = config.get('generation', {})
    seed = generation.get('seed', 42)
    if not isinstance(seed, int):
        raise ValueError(f"Generation seed must be integer, got: {type(seed)}")
    
    # Check satisficing section
    satisficing = config.get('satisficing', {})
    if 'upgrade_cost_nzd_max' in satisficing:
        try:
            float(satisficing['upgrade_cost_nzd_max'])
        except (ValueError, TypeError):
            raise ValueError("satisficing.upgrade_cost_nzd_max must be numeric")


def find_dispatch_summary(bundle_dir: Path, epoch_tag: str) -> Optional[Path]:
    """
    Find dispatch summary CSV for an epoch tag.
    
    Searches: Output/runs/<bundle>/epoch<epoch_tag>/dispatch/**/site_dispatch_<epoch_tag>_summary.csv
    """
    epoch_dir = bundle_dir / f'epoch{epoch_tag}'
    if not epoch_dir.exists():
        return None
    
    pattern = f'**/site_dispatch_{epoch_tag}_summary.csv'
    matches = list(epoch_dir.glob(pattern))
    
    if len(matches) == 0:
        return None
    
    return matches[0]


def load_dispatch_total(bundle_dir: Path, epoch_tag: str) -> Optional[pd.Series]:
    """Load TOTAL row from dispatch summary CSV."""
    summary_path = find_dispatch_summary(bundle_dir, epoch_tag)
    if summary_path is None:
        print(f"[ERROR] Dispatch summary not found for {epoch_tag}")
        return None
    
    try:
        df = pd.read_csv(summary_path)
        
        if 'unit_id' not in df.columns:
            print(f"[ERROR] Summary CSV {summary_path} missing 'unit_id' column")
            return None
        
        total_row = df[df['unit_id'] == 'TOTAL']
        if len(total_row) == 0:
            print(f"[ERROR] No TOTAL row found in {summary_path}")
            return None
        
        return total_row.iloc[0]
    except Exception as e:
        print(f"[ERROR] Failed to load dispatch summary from {summary_path}: {e}")
        return None


def extract_field_with_fallback(row: pd.Series, field_variants: list) -> float:
    """Extract field value with fallback to alternative field names."""
    for field in field_variants:
        if field in row.index:
            value = row[field]
            if pd.notna(value):
                return float(value)
    return 0.0


def decompose_site_costs(dispatch_row: pd.Series) -> Dict[str, float]:
    """
    Decompose site costs into components for multiplier application.
    
    Returns:
        Dictionary with: elec_cost, biomass_cost, carbon_cost, other_cost, total_cost
    """
    # Electricity cost (prefer _total, fallback chain)
    elec_cost = extract_field_with_fallback(
        dispatch_row,
        ['annual_electricity_cost_nzd_total', 'annual_electricity_cost_nzd_effective',
         'annual_electricity_cost_nzd_derived', 'annual_electricity_cost_nzd']
    )
    
    # Total fuel cost
    fuel_cost = extract_field_with_fallback(dispatch_row, ['annual_fuel_cost_nzd'])
    
    # Carbon cost
    carbon_cost = extract_field_with_fallback(dispatch_row, ['annual_carbon_cost_nzd'])
    
    # Biomass cost proxy (conservative: max(fuel_cost - elec_cost, 0))
    biomass_cost = max(fuel_cost - elec_cost, 0.0)
    
    # Total cost
    total_cost = extract_field_with_fallback(dispatch_row, ['annual_total_cost_nzd'])
    
    # Other cost (should be near 0, but use max(..., 0) to avoid negative)
    other_cost = max(total_cost - fuel_cost - carbon_cost, 0.0)
    
    return {
        'elec_cost': elec_cost,
        'biomass_cost': biomass_cost,
        'carbon_cost': carbon_cost,
        'other_cost': other_cost,
        'total_cost': total_cost
    }


def ensure_canonical_futures(
    bundle_dir: Path,
    futures_template_path: Optional[Path] = None
) -> Path:
    """
    Ensure canonical futures.csv exists per bundle.
    
    Creates Output/runs/<bundle>/rdm/futures.csv by copying from template
    if it doesn't exist.
    
    Args:
        bundle_dir: Bundle directory
        futures_template_path: Optional override for template (default: Input/rdm/futures_2035.csv)
        
    Returns:
        Path to canonical futures.csv
    """
    rdm_dir = bundle_dir / 'rdm'
    rdm_dir.mkdir(parents=True, exist_ok=True)
    
    canonical_futures_path = rdm_dir / 'futures.csv'
    
    # If canonical futures already exists, return it
    if canonical_futures_path.exists():
        return canonical_futures_path
    
    # Determine template path
    if futures_template_path is None:
        futures_template_path = Path(input_root()) / 'rdm' / 'futures_2035.csv'
    
    if not futures_template_path.exists():
        raise FileNotFoundError(
            f"Futures template not found: {futures_template_path}\n"
            f"Expected at: Input/rdm/futures_2035.csv"
        )
    
    # Copy template to canonical location
    print(f"[Init] Creating canonical futures.csv from template: {futures_template_path.name}")
    shutil.copy2(futures_template_path, canonical_futures_path)
    print(f"  [OK] Created: {canonical_futures_path}")
    
    return canonical_futures_path


def load_futures_csv(futures_path: Path) -> pd.DataFrame:
    """Load futures CSV (must have future_id column)."""
    if not futures_path.exists():
        raise FileNotFoundError(f"Futures CSV not found: {futures_path}")
    
    df = pd.read_csv(futures_path)
    
    if 'future_id' not in df.columns:
        raise ValueError(f"Futures CSV must have 'future_id' column: {futures_path}")
    
    return df


def generate_multiplier(dist_type: str, range_min: float, range_max: float, 
                       mode: Optional[float] = None) -> float:
    """
    Generate a single multiplier value based on distribution type.
    
    Args:
        dist_type: Distribution type ("uniform", "triangular", "normal")
        range_min: Minimum value
        range_max: Maximum value
        mode: Mode for triangular distribution (required for triangular)
        
    Returns:
        Multiplier value
    """
    if dist_type == "uniform":
        return np.random.uniform(range_min, range_max)
    elif dist_type == "triangular":
        if mode is None:
            raise ValueError("Mode is required for triangular distribution")
        return np.random.triangular(range_min, mode, range_max)
    elif dist_type == "normal":
        # Truncated normal: mean=1.0, stdev=(max-min)/6, clip to [min,max]
        mean = 1.0
        std = (range_max - range_min) / 6.0
        value = np.random.normal(mean, std)
        # Truncate to range
        return np.clip(value, range_min, range_max)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def init_futures_multipliers(
    futures_path: Path,
    config: Dict,
    force: bool = False
) -> None:
    """
    Initialize site multiplier columns in canonical futures CSV if missing.
    
    Adds P_elec_mult, P_biomass_mult, ETS_mult, D_heat_mult columns to futures.csv
    aligned by future_id. Preserves existing grid multiplier columns (U_*).
    
    Args:
        futures_path: Path to canonical futures CSV
        config: Configuration dictionary
        force: If True, overwrite existing site multiplier columns
    """
    df = load_futures_csv(futures_path)
    
    multipliers_config = config.get('multipliers', {})
    seed = config.get('generation', {}).get('seed', 42)
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Site multiplier columns (in order)
    site_multiplier_cols = ['P_elec_mult', 'P_biomass_mult', 'ETS_mult', 'D_heat_mult']
    needs_init = False
    
    for mult_name in site_multiplier_cols:
        if mult_name not in df.columns or force:
            if mult_name in multipliers_config:
                mult_config = multipliers_config[mult_name]
                dist_type = mult_config.get('distribution', 'triangular')
                range_vals = mult_config.get('range', [0.9, 1.1])
                mode = mult_config.get('mode', None)
                
                # Generate values for all futures
                df[mult_name] = [
                    generate_multiplier(dist_type, range_vals[0], range_vals[1], mode)
                    for _ in range(len(df))
                ]
                needs_init = True
                print(f"  [OK] Initialized {mult_name} ({dist_type}, range={range_vals}, mode={mode})")
            else:
                # Default to 1.0 if not in config (especially for D_heat_mult)
                if mult_name not in df.columns:
                    df[mult_name] = 1.0
                    needs_init = True
                    print(f"  [OK] Initialized {mult_name} (default: 1.0)")
    
    if needs_init:
        # Backup original
        backup_path = futures_path.with_suffix('.csv.bak')
        if futures_path.exists():
            shutil.copy2(futures_path, backup_path)
            print(f"  [OK] Backed up original to {backup_path.name}")
        
        # Write updated CSV (preserve column order: future_id, U_*, P_*, D_*)
        # Sort columns: future_id first, then U_* columns, then P_* and D_* columns
        col_order = ['future_id']
        col_order.extend([col for col in df.columns if col.startswith('U_') and col != 'future_id'])
        col_order.extend([col for col in df.columns if col.startswith('P_')])
        col_order.extend([col for col in df.columns if col.startswith('D_')])
        # Add any other columns
        col_order.extend([col for col in df.columns if col not in col_order])
        
        df = df[col_order]
        safe_to_csv(df, futures_path)
        print(f"  [OK] Updated futures CSV: {futures_path}")
        print(f"  [INFO] Generated {len(df)} futures with site multipliers (seed={seed})")
    else:
        print(f"  [INFO] All site multiplier columns already exist (use --force-init to overwrite)")


def load_rdm_summary(rdm_summary_path: Path) -> pd.DataFrame:
    """Load RDM summary CSV."""
    if not rdm_summary_path.exists():
        raise FileNotFoundError(f"RDM summary not found: {rdm_summary_path}")
    
    df = pd.read_csv(rdm_summary_path)
    
    required_cols = ['future_id', 'total_cost_nzd', 'annualised_upgrade_cost_nzd', 'annual_shed_cost_nzd']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"RDM summary missing required columns: {missing}")
    
    return df


def validate_rdm_summaries(rdm_eb: pd.DataFrame, rdm_bb: pd.DataFrame) -> None:
    """Validate RDM summaries for paired futures and cost closure."""
    futures_eb = set(rdm_eb['future_id'].unique())
    futures_bb = set(rdm_bb['future_id'].unique())
    
    if futures_eb != futures_bb:
        missing_eb = futures_bb - futures_eb
        missing_bb = futures_eb - futures_bb
        error_msg = "RDM summaries have mismatched future_id sets:\n"
        if missing_eb:
            error_msg += f"  Missing in EB: {sorted(missing_eb)}\n"
        if missing_bb:
            error_msg += f"  Missing in BB: {sorted(missing_bb)}\n"
        raise ValueError(error_msg)
    
    # Validate cost closure
    for idx, row in rdm_eb.iterrows():
        expected = row['annualised_upgrade_cost_nzd'] + row['annual_shed_cost_nzd']
        actual = row['total_cost_nzd']
        diff = abs(actual - expected)
        if diff > 1e-6:
            raise ValueError(
                f"EB RDM cost closure failed for future_id={row['future_id']}: "
                f"total_cost_nzd={actual:.6f} != upgrade_cost + shed_cost={expected:.6f} (diff={diff:.6f})"
            )
    
    for idx, row in rdm_bb.iterrows():
        expected = row['annualised_upgrade_cost_nzd'] + row['annual_shed_cost_nzd']
        actual = row['total_cost_nzd']
        diff = abs(actual - expected)
        if diff > 1e-6:
            raise ValueError(
                f"BB RDM cost closure failed for future_id={row['future_id']}: "
                f"total_cost_nzd={actual:.6f} != upgrade_cost + shed_cost={expected:.6f} (diff={diff:.6f})"
            )


def compute_robustness_table(
    dispatch_eb: pd.Series,
    dispatch_bb: pd.Series,
    rdm_eb: pd.DataFrame,
    rdm_bb: pd.DataFrame,
    futures_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Compute per-future robustness table with baseline and future-costed system costs.
    """
    # Decompose site costs
    eb_costs = decompose_site_costs(dispatch_eb)
    bb_costs = decompose_site_costs(dispatch_bb)
    
    # Warn if other_cost is not near 0
    if abs(eb_costs['other_cost']) > 1000:
        print(f"[WARN] EB other_cost is {eb_costs['other_cost']:.2f} NZD (expected near 0)")
    if abs(bb_costs['other_cost']) > 1000:
        print(f"[WARN] BB other_cost is {bb_costs['other_cost']:.2f} NZD (expected near 0)")
    
    # Extract baseline site costs
    eb_site_cost_baseline = eb_costs['total_cost']
    bb_site_cost_baseline = bb_costs['total_cost']
    
    # Extract other metrics
    eb_site_co2 = extract_field_with_fallback(dispatch_eb, ['annual_co2_tonnes'])
    bb_site_co2 = extract_field_with_fallback(dispatch_bb, ['annual_co2_tonnes'])
    eb_site_elec = extract_field_with_fallback(
        dispatch_eb,
        ['annual_electricity_MWh_total', 'annual_electricity_MWh_effective', 'annual_electricity_MWh']
    )
    bb_site_elec = extract_field_with_fallback(
        dispatch_bb,
        ['annual_electricity_MWh_total', 'annual_electricity_MWh_effective', 'annual_electricity_MWh']
    )
    
    # Merge RDM summaries
    merged = pd.merge(
        rdm_eb,
        rdm_bb,
        on='future_id',
        suffixes=('_EB', '_BB'),
        how='inner'
    )
    
    # Merge with futures to get multipliers
    merged = pd.merge(merged, futures_df[['future_id', 'P_elec_mult', 'P_biomass_mult', 'ETS_mult']],
                     on='future_id', how='left')
    
    # Check for missing multipliers
    missing_mult = merged[['P_elec_mult', 'P_biomass_mult', 'ETS_mult']].isna().any(axis=1)
    if missing_mult.any():
        print(f"[WARN] {missing_mult.sum()} futures missing multipliers, defaulting to 1.0")
        merged['P_elec_mult'] = merged['P_elec_mult'].fillna(1.0)
        merged['P_biomass_mult'] = merged['P_biomass_mult'].fillna(1.0)
        merged['ETS_mult'] = merged['ETS_mult'].fillna(1.0)
    
    # Build result table
    result = pd.DataFrame()
    result['future_id'] = merged['future_id']
    
    # Baseline site costs
    result['EB_site_cost_nzd'] = eb_site_cost_baseline
    result['BB_site_cost_nzd'] = bb_site_cost_baseline
    
    # Other baseline metrics
    result['EB_site_co2_tonnes'] = eb_site_co2
    result['BB_site_co2_tonnes'] = bb_site_co2
    result['EB_site_electricity_MWh'] = eb_site_elec
    result['BB_site_electricity_MWh'] = bb_site_elec
    
    # RDM upgrade info
    result['EB_upgrade_name'] = merged['selected_upgrade_name_EB'] if 'selected_upgrade_name_EB' in merged.columns else ''
    result['BB_upgrade_name'] = merged['selected_upgrade_name_BB'] if 'selected_upgrade_name_BB' in merged.columns else ''
    result['EB_upgrade_cost_nzd'] = merged['annualised_upgrade_cost_nzd_EB'] if 'annualised_upgrade_cost_nzd_EB' in merged.columns else 0.0
    result['BB_upgrade_cost_nzd'] = merged['annualised_upgrade_cost_nzd_BB'] if 'annualised_upgrade_cost_nzd_BB' in merged.columns else 0.0
    
    # Shedding info
    result['EB_shed_fraction'] = merged['shed_fraction_EB'] if 'shed_fraction_EB' in merged.columns else 0.0
    result['BB_shed_fraction'] = merged['shed_fraction_BB'] if 'shed_fraction_BB' in merged.columns else 0.0
    result['EB_shed_MWh'] = merged['annual_shed_MWh_EB'] if 'annual_shed_MWh_EB' in merged.columns else 0.0
    result['BB_shed_MWh'] = merged['annual_shed_MWh_BB'] if 'annual_shed_MWh_BB' in merged.columns else 0.0
    
    # RDM costs
    result['EB_rdm_cost_nzd'] = merged['total_cost_nzd_EB'] if 'total_cost_nzd_EB' in merged.columns else 0.0
    result['BB_rdm_cost_nzd'] = merged['total_cost_nzd_BB'] if 'total_cost_nzd_BB' in merged.columns else 0.0
    
    # Baseline system costs
    result['EB_system_cost_nzd'] = eb_site_cost_baseline + result['EB_rdm_cost_nzd']
    result['BB_system_cost_nzd'] = bb_site_cost_baseline + result['BB_rdm_cost_nzd']
    
    # Multipliers
    result['P_elec_mult'] = merged['P_elec_mult']
    result['P_biomass_mult'] = merged['P_biomass_mult']
    result['ETS_mult'] = merged['ETS_mult']
    
    # Future-costed site costs (apply multipliers)
    result['EB_site_cost_future_nzd'] = (
        eb_costs['other_cost'] +
        eb_costs['elec_cost'] * result['P_elec_mult'] +
        eb_costs['biomass_cost'] * result['P_biomass_mult'] +
        eb_costs['carbon_cost'] * result['ETS_mult']
    )
    result['BB_site_cost_future_nzd'] = (
        bb_costs['other_cost'] +
        bb_costs['elec_cost'] * result['P_elec_mult'] +
        bb_costs['biomass_cost'] * result['P_biomass_mult'] +
        bb_costs['carbon_cost'] * result['ETS_mult']
    )
    
    # Future-costed system costs
    result['EB_system_cost_future_nzd'] = result['EB_site_cost_future_nzd'] + result['EB_rdm_cost_nzd']
    result['BB_system_cost_future_nzd'] = result['BB_site_cost_future_nzd'] + result['BB_rdm_cost_nzd']
    
    # Winner (future-costed) - pathway name
    result['winner_system_cost_future'] = result.apply(
        lambda row: 'EB' if row['EB_system_cost_future_nzd'] < row['BB_system_cost_future_nzd'] else 'BB',
        axis=1
    )
    
    # Winner cost value (for reference)
    result['winner_system_cost_future_nzd'] = result[['EB_system_cost_future_nzd', 'BB_system_cost_future_nzd']].min(axis=1)
    
    # Regret (future-costed) - difference from winner
    result['regret_EB_future_nzd'] = result['EB_system_cost_future_nzd'] - result['winner_system_cost_future_nzd']
    result['regret_BB_future_nzd'] = result['BB_system_cost_future_nzd'] - result['winner_system_cost_future_nzd']
    
    # Interpretability deltas
    result['delta_site_cost_future_nzd'] = result['EB_site_cost_future_nzd'] - result['BB_site_cost_future_nzd']
    result['delta_upgrade_cost_nzd'] = result['EB_upgrade_cost_nzd'] - result['BB_upgrade_cost_nzd']
    result['delta_system_cost_future_nzd'] = result['EB_system_cost_future_nzd'] - result['BB_system_cost_future_nzd']
    
    return result


def compute_summary_metrics(robustness_df: pd.DataFrame, rdm_eb: pd.DataFrame, rdm_bb: pd.DataFrame, config: Dict) -> Dict:
    """Compute summary metrics using future-costed system costs."""
    n_futures = len(robustness_df)
    
    # Win rate (future-costed)
    eb_wins = (robustness_df['EB_system_cost_future_nzd'] < robustness_df['BB_system_cost_future_nzd']).sum()
    win_rate_eb = (eb_wins / n_futures * 100) if n_futures > 0 else 0.0
    
    # Regret statistics (future-costed)
    regret_eb = robustness_df['regret_EB_future_nzd'].values
    regret_bb = robustness_df['regret_BB_future_nzd'].values
    
    regret_eb = regret_eb[np.isfinite(regret_eb)]
    regret_bb = regret_bb[np.isfinite(regret_bb)]
    
    mean_regret_eb = np.mean(regret_eb) if len(regret_eb) > 0 else 0.0
    p95_regret_eb = np.percentile(regret_eb, 95) if len(regret_eb) > 0 else 0.0
    max_regret_eb = np.max(regret_eb) if len(regret_eb) > 0 else 0.0
    
    mean_regret_bb = np.mean(regret_bb) if len(regret_bb) > 0 else 0.0
    p95_regret_bb = np.percentile(regret_bb, 95) if len(regret_bb) > 0 else 0.0
    max_regret_bb = np.max(regret_bb) if len(regret_bb) > 0 else 0.0
    
    # Upgrade exposure
    upgrade_exposure_eb = 0.0
    if 'selected_capacity_MW' in rdm_eb.columns:
        exposure_count = (rdm_eb['selected_capacity_MW'] >= 150).sum()
        upgrade_exposure_eb = (exposure_count / n_futures * 100) if n_futures > 0 else 0.0
    
    upgrade_exposure_bb = 0.0
    if 'selected_capacity_MW' in rdm_bb.columns:
        exposure_count = (rdm_bb['selected_capacity_MW'] >= 150).sum()
        upgrade_exposure_bb = (exposure_count / n_futures * 100) if n_futures > 0 else 0.0
    
    # Satisficing rates
    satisficing_config = config.get('satisficing', {})
    shed_threshold = satisficing_config.get('shed_fraction_max', 0.0)
    upgrade_threshold = satisficing_config.get('upgrade_cost_nzd_max', 7e6)
    
    satisficing_eb = 0.0
    if 'shed_fraction' in rdm_eb.columns and 'annualised_upgrade_cost_nzd' in rdm_eb.columns:
        satisficing_mask = (
            (rdm_eb['shed_fraction'] <= shed_threshold) &
            (rdm_eb['annualised_upgrade_cost_nzd'] <= upgrade_threshold)
        )
        satisficing_eb = (satisficing_mask.sum() / n_futures * 100) if n_futures > 0 else 0.0
    
    satisficing_bb = 0.0
    if 'shed_fraction' in rdm_bb.columns and 'annualised_upgrade_cost_nzd' in rdm_bb.columns:
        satisficing_mask = (
            (rdm_bb['shed_fraction'] <= shed_threshold) &
            (rdm_bb['annualised_upgrade_cost_nzd'] <= upgrade_threshold)
        )
        satisficing_bb = (satisficing_mask.sum() / n_futures * 100) if n_futures > 0 else 0.0
    
    return {
        'n_futures': n_futures,
        'win_rate_EB_system_cost_future': win_rate_eb,
        'mean_regret_EB_nzd': mean_regret_eb,
        'p95_regret_EB_nzd': p95_regret_eb,
        'max_regret_EB_nzd': max_regret_eb,
        'mean_regret_BB_nzd': mean_regret_bb,
        'p95_regret_BB_nzd': p95_regret_bb,
        'max_regret_BB_nzd': max_regret_bb,
        'upgrade_exposure_EB_P150': upgrade_exposure_eb,
        'upgrade_exposure_BB_P150': upgrade_exposure_bb,
        'satisficing_rate_EB': satisficing_eb,
        'satisficing_rate_BB': satisficing_bb
    }


def plot_regret_cdf(robustness_df: pd.DataFrame, output_path: Path) -> None:
    """Plot regret CDF comparison using future-costed regrets."""
    regret_eb = robustness_df['regret_EB_future_nzd'].values
    regret_bb = robustness_df['regret_BB_future_nzd'].values
    
    regret_eb = regret_eb[np.isfinite(regret_eb)]
    regret_bb = regret_bb[np.isfinite(regret_bb)]
    
    if len(regret_eb) == 0 or len(regret_bb) == 0:
        print("[WARN] No valid regret data for CDF plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_eb = np.sort(regret_eb)
    sorted_bb = np.sort(regret_bb)
    p_eb = np.arange(1, len(sorted_eb) + 1) / len(sorted_eb)
    p_bb = np.arange(1, len(sorted_bb) + 1) / len(sorted_bb)
    
    ax.plot(sorted_eb, p_eb * 100, linewidth=2, color='#3498db', label='EB', marker='o', markersize=3)
    ax.plot(sorted_bb, p_bb * 100, linewidth=2, color='#e74c3c', label='BB', marker='s', markersize=3)
    
    ax.set_xlabel('Regret (NZD)', fontsize=11)
    ax.set_ylabel('Cumulative Probability (%)', fontsize=11)
    ax.set_title('Site Decision Robustness: Regret CDF Comparison (2035)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_system_cost_boxplot(robustness_df: pd.DataFrame, output_path: Path) -> None:
    """Plot boxplot of future-costed system costs (EB vs BB)."""
    cost_eb = robustness_df['EB_system_cost_future_nzd'].values
    cost_bb = robustness_df['BB_system_cost_future_nzd'].values
    
    cost_eb = cost_eb[np.isfinite(cost_eb)]
    cost_bb = cost_bb[np.isfinite(cost_bb)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    try:
        bp = ax.boxplot([cost_eb, cost_bb], tick_labels=['EB', 'BB'], patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
    except TypeError:
        bp = ax.boxplot([cost_eb, cost_bb], labels=['EB', 'BB'], patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
    
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('System Cost (NZD)', fontsize=11)
    ax.set_title('Site Decision Robustness: System Cost Distribution (2035)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def archive_file(file_path: Path, archive_dir: Path) -> None:
    """
    Archive a file before replacing it (tolerant of Windows file locking).
    
    If the file is locked (e.g., open in Excel), prints a warning and continues
    without crashing.
    """
    if not file_path.exists():
        return
    
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / file_path.name
        
        shutil.copy2(file_path, archive_path)
        print(f"  [ARCHIVE] Archived {file_path.name} to {archive_dir.name}/")
    except PermissionError:
        print(f"  [WARN] Could not archive {file_path.name} (likely locked). Skipping archive for this file.")
    except Exception as e:
        print(f"  [WARN] Could not archive {file_path.name}: {e}. Skipping archive for this file.")


def update_pointers(bundle_dir: Path, tables: list, figures: list) -> None:
    """
    Update pointers.md, replacing existing entries instead of appending duplicates.
    """
    pointers_path = bundle_dir / 'thesis_pack' / 'pointers.md'
    
    if not pointers_path.exists():
        print(f"[WARN] pointers.md not found at {pointers_path}, skipping update")
        return
    
    # Read existing content
    with open(pointers_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Build filename sets for deduplication
    table_filenames = {filename for filename, _ in tables}
    figure_filenames = {filename for filename, _ in figures}
    
    # Filter out existing entries for our files
    new_lines = []
    in_tables_section = False
    in_figures_section = False
    
    for line in lines:
        stripped = line.strip()
        
        if stripped == "## Tables":
            in_tables_section = True
            in_figures_section = False
            new_lines.append(line)
        elif stripped == "## Figures":
            in_tables_section = False
            in_figures_section = True
            new_lines.append(line)
        elif stripped.startswith("##"):
            in_tables_section = False
            in_figures_section = False
            new_lines.append(line)
        elif stripped.startswith("- ") and " -> " in stripped:
            # Check if this is one of our files
            filename = stripped.split(" -> ")[0][2:]  # Remove "- " prefix
            if (in_tables_section and filename in table_filenames) or \
               (in_figures_section and filename in figure_filenames):
                continue  # Skip existing entry
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Append new entries
    if tables:
        # Find Tables section end
        tables_end = len(new_lines)
        for i, line in enumerate(new_lines):
            if line.strip() == "## Figures":
                tables_end = i
                break
        
        for filename, source_path in tables:
            new_lines.insert(tables_end, f"- {filename} -> {source_path}\n")
    
    if figures:
        # Find Figures section end
        figures_end = len(new_lines)
        for i in range(len(new_lines) - 1, -1, -1):
            if new_lines[i].strip().startswith("##"):
                figures_end = i + 1
                break
        
        for filename, source_path in figures:
            new_lines.insert(figures_end, f"- {filename} -> {source_path}\n")
    
    # Write back
    with open(pointers_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"[OK] Updated {pointers_path.name}")


def validate_only(
    bundle_dir: Path,
    epoch_eb: str,
    epoch_bb: str,
    futures_path: Path,
    config: Dict,
    allow_extra_futures: bool = False
) -> None:
    """
    Validate-only mode: check files, schema, paired futures, and multiplier status.
    
    Args:
        bundle_dir: Bundle directory
        epoch_eb: EB epoch tag
        epoch_bb: BB epoch tag
        futures_path: Path to canonical futures CSV
        config: Configuration dictionary
        allow_extra_futures: If True, allow futures.csv to have extra futures beyond RDM
    """
    print("[VALIDATE] Running validation checks...")
    
    # Check dispatch summaries
    dispatch_eb_path = find_dispatch_summary(bundle_dir, epoch_eb)
    dispatch_bb_path = find_dispatch_summary(bundle_dir, epoch_bb)
    
    if dispatch_eb_path is None:
        print(f"  [FAIL] EB dispatch summary not found")
        return
    else:
        print(f"  [OK] EB dispatch summary: {dispatch_eb_path.name}")
    
    if dispatch_bb_path is None:
        print(f"  [FAIL] BB dispatch summary not found")
        return
    else:
        print(f"  [OK] BB dispatch summary: {dispatch_bb_path.name}")
    
    # Check RDM summaries
    rdm_dir = bundle_dir / 'rdm'
    rdm_eb_path = rdm_dir / f'rdm_summary_{epoch_eb}.csv'
    rdm_bb_path = rdm_dir / f'rdm_summary_{epoch_bb}.csv'
    
    if not rdm_eb_path.exists():
        print(f"  [FAIL] EB RDM summary not found: {rdm_eb_path}")
        return
    else:
        print(f"  [OK] EB RDM summary: {rdm_eb_path.name}")
    
    if not rdm_bb_path.exists():
        print(f"  [FAIL] BB RDM summary not found: {rdm_bb_path}")
        return
    else:
        print(f"  [OK] BB RDM summary: {rdm_bb_path.name}")
    
    # Check canonical futures CSV
    if not futures_path.exists():
        print(f"  [FAIL] Canonical futures CSV not found: {futures_path}")
        print(f"  [INFO] Run with --init-futures-multipliers to create it")
        return
    else:
        print(f"  [OK] Canonical futures CSV: {futures_path.name}")
    
    # Load and validate
    try:
        rdm_eb = load_rdm_summary(rdm_eb_path)
        rdm_bb = load_rdm_summary(rdm_bb_path)
        futures_df = load_futures_csv(futures_path)
        
        # Check paired futures (EB and BB must match exactly)
        futures_eb = set(rdm_eb['future_id'].unique())
        futures_bb = set(rdm_bb['future_id'].unique())
        futures_csv_ids = set(futures_df['future_id'].unique())
        
        if futures_eb != futures_bb:
            print(f"  [FAIL] EB and BB RDM summaries have mismatched future_id sets:")
            missing_eb = futures_bb - futures_eb
            missing_bb = futures_eb - futures_bb
            if missing_eb:
                print(f"    Missing in EB: {sorted(missing_eb)}")
            if missing_bb:
                print(f"    Missing in BB: {sorted(missing_bb)}")
            return
        
        # Check that canonical futures.csv contains exactly those future_ids
        if not allow_extra_futures:
            if futures_csv_ids != futures_eb:
                missing_in_csv = futures_eb - futures_csv_ids
                extra_in_csv = futures_csv_ids - futures_eb
                print(f"  [FAIL] Canonical futures.csv does not match RDM future_ids:")
                if missing_in_csv:
                    print(f"    Missing in futures.csv: {sorted(missing_in_csv)}")
                if extra_in_csv:
                    print(f"    Extra in futures.csv: {sorted(extra_in_csv)}")
                return
            print(f"  [OK] Paired futures validated: {len(futures_eb)} futures (EB=BB=CSV)")
        else:
            if not futures_eb.issubset(futures_csv_ids):
                missing_in_csv = futures_eb - futures_csv_ids
                print(f"  [FAIL] Canonical futures.csv missing required future_ids: {sorted(missing_in_csv)}")
                return
            print(f"  [OK] Paired futures validated: {len(futures_eb)} futures (EB=BB, CSV has {len(futures_csv_ids)})")
        
        # Check cost closure
        validate_rdm_summaries(rdm_eb, rdm_bb)
        print(f"  [OK] Cost closure validation passed")
        
        # Check multipliers
        multiplier_cols = ['P_elec_mult', 'P_biomass_mult', 'ETS_mult']
        missing = [col for col in multiplier_cols if col not in futures_df.columns]
        if missing:
            print(f"  [WARN] Missing multiplier columns: {missing}")
            print(f"  [INFO] Multipliers will default to 1.0 (degenerate robustness)")
            print(f"  [INFO] Run with --init-futures-multipliers to generate them")
        else:
            print(f"  [OK] Multiplier columns present: {multiplier_cols}")
            # Check if all are 1.0
            all_one = True
            for col in multiplier_cols:
                if not (futures_df[col].fillna(1.0) == 1.0).all():
                    all_one = False
                    break
            if all_one:
                print(f"  [WARN] All multipliers are 1.0 (degenerate robustness)")
            else:
                print(f"  [OK] Multipliers are active (non-trivial robustness)")
                # Show range
                for col in multiplier_cols:
                    vals = futures_df[col].values
                    print(f"    {col}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")
        
        print("\n[OK] All validation checks passed")
        
    except Exception as e:
        print(f"  [FAIL] Validation error: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    """CLI entrypoint for site decision robustness consolidation."""
    parser = argparse.ArgumentParser(
        description='Consolidate site dispatch costs with RDM upgrade costs for robustness analysis (PoC v2)'
    )
    parser.add_argument('--bundle', type=str, required=True,
                       help='Bundle name (e.g., poc_20260105_release02)')
    parser.add_argument('--output-root', type=str, default='Output',
                       help='Output root directory (default: Output)')
    parser.add_argument('--epoch-eb', type=str, default='2035_EB',
                       help='EB epoch tag (default: 2035_EB)')
    parser.add_argument('--epoch-bb', type=str, default='2035_BB',
                       help='BB epoch tag (default: 2035_BB)')
    parser.add_argument('--config', type=str, default=None,
                       help='Override config TOML path (default: searches Input/configs/ then configs/)')
    parser.add_argument('--futures-csv', type=str, default=None,
                       help='Override path to canonical futures CSV (default: <bundle>/rdm/futures.csv)')
    parser.add_argument('--futures-template', type=str, default=None,
                       help='Override path to futures template (default: Input/rdm/futures_2035.csv)')
    parser.add_argument('--init-futures-multipliers', action='store_true',
                       help='Create canonical futures.csv and initialize multiplier columns if missing')
    parser.add_argument('--force-init', action='store_true',
                       help='Force overwrite existing multiplier columns')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override random seed for multiplier generation')
    parser.add_argument('--validate-only', action='store_true',
                       help='Run validation checks only (no outputs)')
    
    args = parser.parse_args()
    
    # Resolve paths
    output_root = Path(resolve_path(args.output_root))
    bundle_dir = output_root / 'runs' / args.bundle
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    
    # Load config (use find_config_file if no override)
    if args.config:
        config_path = Path(resolve_path(args.config))
    else:
        config_path = None  # Will use find_config_file()
    config = load_config(config_path)
    
    # Override seed if provided
    if args.seed is not None:
        config['generation'] = config.get('generation', {})
        config['generation']['seed'] = args.seed
    
    # Determine canonical futures CSV path
    if args.futures_csv:
        futures_path = Path(resolve_path(args.futures_csv))
    else:
        futures_path = bundle_dir / 'rdm' / 'futures.csv'
    
    # Determine futures template path
    futures_template_path = None
    if args.futures_template:
        futures_template_path = Path(resolve_path(args.futures_template))
    
    # Ensure canonical futures.csv exists (create from template if needed)
    if not futures_path.exists() and args.init_futures_multipliers:
        print("[Init] Creating canonical futures.csv from template...")
        futures_path = ensure_canonical_futures(bundle_dir, futures_template_path)
    elif not futures_path.exists():
        print(f"[WARN] Canonical futures.csv not found: {futures_path}")
        print(f"[INFO] Run with --init-futures-multipliers to create it from template")
        if args.validate_only:
            # In validate-only mode, we can still check other things
            pass
        else:
            raise FileNotFoundError(
                f"Canonical futures.csv not found: {futures_path}\n"
                f"Run with --init-futures-multipliers to create it"
            )
    
    # Initialize multipliers if requested
    if args.init_futures_multipliers:
        print("[Init] Initializing site multiplier columns...")
        init_futures_multipliers(futures_path, config, force=args.force_init)
        if args.validate_only:
            return
    
    # Validate-only mode
    if args.validate_only:
        validate_only(bundle_dir, args.epoch_eb, args.epoch_bb, futures_path, config)
        return
    
    print(f"[Site Decision Robustness] Processing bundle: {args.bundle}")
    print(f"  EB epoch: {args.epoch_eb}")
    print(f"  BB epoch: {args.epoch_bb}")
    
    # Load dispatch summaries
    print("\n[1/7] Loading dispatch summaries...")
    dispatch_eb = load_dispatch_total(bundle_dir, args.epoch_eb)
    dispatch_bb = load_dispatch_total(bundle_dir, args.epoch_bb)
    
    if dispatch_eb is None or dispatch_bb is None:
        raise ValueError("Failed to load dispatch summaries")
    
    print(f"  [OK] EB dispatch summary loaded")
    print(f"  [OK] BB dispatch summary loaded")
    
    # Load RDM summaries
    print("\n[2/7] Loading RDM summaries...")
    rdm_dir = bundle_dir / 'rdm'
    rdm_eb_path = rdm_dir / f'rdm_summary_{args.epoch_eb}.csv'
    rdm_bb_path = rdm_dir / f'rdm_summary_{args.epoch_bb}.csv'
    
    rdm_eb = load_rdm_summary(rdm_eb_path)
    rdm_bb = load_rdm_summary(rdm_bb_path)
    
    print(f"  [OK] EB RDM summary loaded: {len(rdm_eb)} futures")
    print(f"  [OK] BB RDM summary loaded: {len(rdm_bb)} futures")
    
    # Validate RDM summaries
    print("\n[3/7] Validating RDM summaries...")
    validate_rdm_summaries(rdm_eb, rdm_bb)
    print(f"  [OK] Validation passed: {len(rdm_eb)} paired futures")
    
    # Load futures CSV
    print("\n[4/7] Loading canonical futures CSV...")
    futures_df_full = load_futures_csv(futures_path)
    
    # Filter futures to only those in RDM summaries (paired futures)
    rdm_future_ids = set(rdm_eb['future_id'].unique())
    futures_df = futures_df_full[futures_df_full['future_id'].isin(rdm_future_ids)].copy()
    
    if len(futures_df) != len(rdm_eb):
        raise ValueError(
            f"Canonical futures.csv does not contain all RDM future_ids. "
            f"Expected {len(rdm_eb)} futures, found {len(futures_df)} matching futures."
        )
    
    print(f"  [INFO] Canonical futures rows: {len(futures_df_full)}, using paired futures from RDM: {len(futures_df)}")
    print(f"  [OK] Loaded {len(futures_df)} futures (matched to RDM summaries)")
    
    # Check multipliers
    multiplier_cols = ['P_elec_mult', 'P_biomass_mult', 'ETS_mult']
    missing = [col for col in multiplier_cols if col not in futures_df.columns]
    if missing:
        print(f"  [WARN] Missing multiplier columns: {missing}")
        print(f"  [WARN] Defaulting to 1.0 (robustness will be degenerate)")
        print(f"  [INFO] Run with --init-futures-multipliers to generate them")
        for col in missing:
            futures_df[col] = 1.0
    else:
        print(f"  [OK] Multiplier columns present")
        # Check if multipliers are active (not all 1.0)
        all_one = True
        for col in multiplier_cols:
            if not (futures_df[col].fillna(1.0) == 1.0).all():
                all_one = False
                break
        if all_one:
            print(f"  [WARN] All multipliers are 1.0 (degenerate robustness)")
        else:
            print(f"  [OK] Multipliers are active (non-trivial robustness)")
    
    # Compute robustness table
    print("\n[5/7] Computing robustness metrics...")
    robustness_df = compute_robustness_table(dispatch_eb, dispatch_bb, rdm_eb, rdm_bb, futures_df, config)
    
    # Create archive directory
    archive_dir = bundle_dir / '_archive_overlays' / datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Write per-future table (archive old if exists)
    robustness_table_path = rdm_dir / f'site_decision_robustness_{args.epoch_eb}_vs_{args.epoch_bb}.csv'
    archive_file(robustness_table_path, archive_dir)
    safe_to_csv(robustness_df, robustness_table_path)
    print(f"  [OK] Per-future table written: {robustness_table_path.name}")
    
    # Compute summary metrics
    summary_metrics = compute_summary_metrics(robustness_df, rdm_eb, rdm_bb, config)
    summary_df = pd.DataFrame([summary_metrics])
    summary_path = rdm_dir / 'site_decision_robustness_summary_2035.csv'
    archive_file(summary_path, archive_dir)
    safe_to_csv(summary_df, summary_path)
    print(f"  [OK] Summary metrics written: {summary_path.name}")
    
    # Generate figures
    print("\n[6/7] Generating figures...")
    thesis_pack_figures_dir = bundle_dir / 'thesis_pack' / 'figures'
    thesis_pack_figures_dir.mkdir(parents=True, exist_ok=True)
    
    figure_path = thesis_pack_figures_dir / 'site_decision_regret_cdf_2035.png'
    archive_file(figure_path, archive_dir)
    plot_regret_cdf(robustness_df, figure_path)
    
    figure_path2 = thesis_pack_figures_dir / 'site_decision_system_cost_boxplot_2035.png'
    archive_file(figure_path2, archive_dir)
    plot_system_cost_boxplot(robustness_df, figure_path2)
    
    # Update pointers.md
    print("\n[7/7] Updating pointers.md...")
    tables = [
        (robustness_table_path.name, str(robustness_table_path)),
        (summary_path.name, str(summary_path))
    ]
    figures = [
        (figure_path.name, str(figure_path)),
        (figure_path2.name, str(figure_path2))
    ]
    update_pointers(bundle_dir, tables, figures)
    
    print("\n" + "="*80)
    print("Site Decision Robustness Consolidation Complete (PoC v2)")
    print("="*80)
    print(f"Per-future table: {robustness_table_path}")
    print(f"Summary metrics: {summary_path}")
    print(f"Regret CDF figure: {figure_path}")
    print(f"System cost boxplot: {figure_path2}")
    if archive_dir.exists() and any(archive_dir.iterdir()):
        print(f"Archived old outputs to: {archive_dir}")


if __name__ == '__main__':
    main()
