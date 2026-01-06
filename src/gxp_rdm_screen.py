"""
GXP RDM Screening Module

Minimal, deterministic RDM screening for regional electricity grid constraints.
Evaluates upgrade options across uncertainty futures using paired futures for EB and BB.

This module implements one-pass coupling: site dispatch → incremental electricity → 
regional screening with no iterative feedback.
"""

from __future__ import annotations

import sys
from pathlib import Path as PathlibPath

ROOT = PathlibPath(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# TOML loading
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Need tomllib (Python 3.11+) or tomli package")

from src.path_utils import repo_root, resolve_path
from src.time_utils import parse_any_timestamp


def load_futures_csv(futures_path: PathlibPath) -> pd.DataFrame:
    """
    Load futures CSV with uncertainty multipliers.
    
    Expected schema: future_id, U_headroom_mult, U_inc_mult, U_upgrade_capex_mult, U_voll, U_consents_uplift
    
    Args:
        futures_path: Path to futures CSV
        
    Returns:
        DataFrame with futures
    """
    if not futures_path.exists():
        raise FileNotFoundError(f"Futures CSV not found: {futures_path}")
    
    df = pd.read_csv(futures_path)
    
    # Validate required columns
    required_cols = ['future_id', 'U_headroom_mult', 'U_inc_mult', 'U_upgrade_capex_mult', 'U_voll', 'U_consents_uplift']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Futures CSV missing required columns: {missing}")
    
    print(f"[OK] Loaded {len(df)} futures from {futures_path}")
    return df


def load_incremental_electricity(csv_path: PathlibPath) -> pd.DataFrame:
    """
    Load incremental electricity demand CSV.
    
    Expected schema: timestamp/timestamp_utc, incremental_electricity_MW
    
    Args:
        csv_path: Path to incremental electricity CSV
        
    Returns:
        DataFrame with columns: timestamp_utc, incremental_electricity_MW
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Incremental electricity CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Find timestamp column
    ts_col = None
    for candidate in ['timestamp_utc', 'timestamp']:
        if candidate in df.columns:
            ts_col = candidate
            break
    
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {csv_path}. Expected 'timestamp_utc' or 'timestamp'")
    
    # Find incremental column
    inc_col = None
    for candidate in ['incremental_electricity_MW', 'incremental_MW']:
        if candidate in df.columns:
            inc_col = candidate
            break
    
    if inc_col is None:
        raise ValueError(f"No incremental electricity column found in {csv_path}")
    
    # Parse timestamps
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='raise')
    
    # Rename to standard names
    result = df[[ts_col, inc_col]].copy()
    result.columns = ['timestamp_utc', 'incremental_electricity_MW']
    
    # Ensure numeric
    result['incremental_electricity_MW'] = pd.to_numeric(result['incremental_electricity_MW'], errors='coerce').fillna(0.0)
    
    print(f"[OK] Loaded incremental electricity: {len(result)} rows")
    return result


def load_or_generate_headroom(timestamps: pd.DatetimeIndex, headroom_csv: Optional[PathlibPath] = None) -> pd.Series:
    """
    Load headroom from CSV or generate deterministic PoC headroom.
    
    Args:
        timestamps: DatetimeIndex for headroom series
        headroom_csv: Optional path to headroom CSV
        
    Returns:
        Series with headroom_MW indexed by timestamp
    """
    if headroom_csv and headroom_csv.exists():
        df = pd.read_csv(headroom_csv)
        
        # Find timestamp column
        ts_col = None
        for candidate in ['timestamp_utc', 'timestamp']:
            if candidate in df.columns:
                ts_col = candidate
                break
        
        if ts_col is None:
            raise ValueError(f"No timestamp column found in {headroom_csv}")
        
        # Find headroom column
        hr_col = None
        for candidate in ['headroom_MW', 'headroom_mw', 'headroom']:
            if candidate in df.columns:
                hr_col = candidate
                break
        
        if hr_col is None:
            raise ValueError(f"No headroom column found in {headroom_csv}")
        
        # Parse timestamps
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='raise')
        df = df.set_index(ts_col)
        
        # Reindex to match timestamps
        headroom = df[hr_col].reindex(timestamps, method='nearest')
        headroom = pd.to_numeric(headroom, errors='coerce').fillna(0.0)
        
        print(f"[OK] Loaded headroom from {headroom_csv}")
        return headroom
    else:
        # Generate deterministic PoC headroom with seasonal pattern
        # Base headroom: 50 MW
        # Seasonal variation: ±20 MW (winter lower, summer higher)
        # Weekly pattern: weekend slightly higher
        
        n_hours = len(timestamps)
        base_headroom = 50.0
        
        # Seasonal pattern (sine wave, peak in summer)
        day_of_year = timestamps.dayofyear
        seasonal = 20.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)  # Peak around day 80 (March)
        
        # Weekly pattern (weekend +5 MW)
        is_weekend = timestamps.dayofweek >= 5
        weekly = np.where(is_weekend, 5.0, 0.0)
        
        headroom = base_headroom + seasonal + weekly
        headroom = np.maximum(headroom, 10.0)  # Minimum 10 MW
        
        result = pd.Series(headroom, index=timestamps, name='headroom_MW')
        print(f"[WARN] Generated deterministic PoC headroom (base=50 MW, seasonal ±20 MW, weekend +5 MW)")
        return result


def load_upgrade_menu(upgrade_toml: PathlibPath) -> Tuple[List[Dict], Dict]:
    """
    Load upgrade menu from TOML.
    
    Args:
        upgrade_toml: Path to upgrade menu TOML
        
    Returns:
        Tuple of (upgrade_options list, assumptions dict)
    """
    if not upgrade_toml.exists():
        raise FileNotFoundError(f"Upgrade menu TOML not found: {upgrade_toml}")
    
    with open(upgrade_toml, 'rb') as f:
        config = tomllib.load(f)
    
    assumptions = config.get('assumptions', {})
    upgrades = config.get('upgrades', [])
    
    # Add "none" option if not present
    has_none = any(opt.get('name') == 'none' for opt in upgrades)
    if not has_none:
        upgrades.insert(0, {
            'name': 'none',
            'capacity_MW': 0,
            'capex_nzd': 0.0,
            'lead_time_months': 0,
            'security_level': 'existing'
        })
    
    print(f"[OK] Loaded {len(upgrades)} upgrade options from {upgrade_toml}")
    return upgrades, assumptions


def compute_annualised_cost(capex_nzd: float, assumptions: Dict, consents_uplift: float = 1.0) -> float:
    """
    Compute annualised cost from capex using CRF.
    
    Args:
        capex_nzd: Capital expenditure (NZD)
        assumptions: Assumptions dict with real_discount_rate, asset_life_years
        consents_uplift: Consents uplift multiplier
        
    Returns:
        Annualised cost (NZD/year)
    """
    r = assumptions.get('real_discount_rate', 0.07)
    n = assumptions.get('asset_life_years', 40)
    
    # Capital recovery factor
    if r == 0:
        crf = 1.0 / n
    else:
        crf = r * (1 + r)**n / ((1 + r)**n - 1)
    
    # Apply consents uplift
    total_capex = capex_nzd * consents_uplift
    
    annualised = total_capex * crf
    return annualised


def evaluate_upgrade_option(
    incremental: pd.Series,
    headroom: pd.Series,
    upgrade: Dict,
    assumptions: Dict,
    future_row: pd.Series,
    dt_h: float = 1.0
) -> Dict:
    """
    Evaluate a single upgrade option for a future.
    
    Args:
        incremental: Incremental electricity demand (MW) - already scaled by uncertainties
        headroom: Headroom (MW) - already scaled by uncertainties
        upgrade: Upgrade option dict
        assumptions: Assumptions dict
        future_row: Series with uncertainty multipliers
        dt_h: Timestep in hours
        
    Returns:
        Dict with metrics for this upgrade option
    """
    upgrade_name = upgrade['name']
    upgrade_capacity_MW = float(upgrade.get('capacity_MW', 0))
    base_capex = float(upgrade.get('capex_nzd', 0.0))
    capex_mult = future_row.get('U_upgrade_capex_mult', 1.0)
    voll = float(future_row.get('U_voll', 10000.0))
    consents_uplift = float(future_row.get('U_consents_uplift', 1.25))
    
    # Apply capex multiplier (except for "none")
    if upgrade_name == 'none':
        capex_nzd = 0.0
    else:
        capex_nzd = base_capex * capex_mult
    
    # Compute annualised upgrade cost
    annualised_upgrade_cost = compute_annualised_cost(capex_nzd, assumptions, consents_uplift)
    
    # Compute effective headroom (base + upgrade)
    effective_headroom = headroom + upgrade_capacity_MW
    
    # Compute exceed (incremental - effective_headroom when positive)
    exceed_MW = np.maximum(0.0, incremental - effective_headroom)
    
    # Compute shortfall (same as exceed for shed)
    shortfall_MW = exceed_MW.copy()
    
    # Compute annual metrics
    annual_incremental_MWh = float(np.sum(incremental * dt_h))
    annual_shed_MWh = float(np.sum(shortfall_MW * dt_h))
    shed_fraction = annual_shed_MWh / annual_incremental_MWh if annual_incremental_MWh > 0 else 0.0
    
    # Compute exceed metrics
    hours_exceed = int(np.sum(exceed_MW > 1e-6))
    hours_shed = int(np.sum(shortfall_MW > 1e-6))
    max_exceed_MW = float(np.max(exceed_MW)) if len(exceed_MW) > 0 else 0.0
    
    # Compute percentiles on exceed>0 distribution
    exceed_positive = exceed_MW[exceed_MW > 1e-6]
    if len(exceed_positive) > 0:
        p95_exceed_MW = float(np.percentile(exceed_positive, 95))
        p99_exceed_MW = float(np.percentile(exceed_positive, 99))
    else:
        p95_exceed_MW = 0.0
        p99_exceed_MW = 0.0
    
    unserved_peak_MW = max_exceed_MW  # Alias for backward compatibility
    
    # Compute shed cost
    annual_shed_cost = annual_shed_MWh * voll
    
    # Total cost
    total_cost = annualised_upgrade_cost + annual_shed_cost
    
    # Get menu max capacity
    # This will be computed at the matrix level, set to None for now
    menu_max_capacity_MW = None
    menu_max_selected = None
    
    return {
        'selected_upgrade_name': upgrade_name,
        'selected_capacity_MW': upgrade_capacity_MW,
        'annualised_upgrade_cost_nzd': annualised_upgrade_cost,
        'annual_incremental_MWh': annual_incremental_MWh,
        'annual_shed_MWh': annual_shed_MWh,
        'shed_fraction': shed_fraction,
        'annual_shed_cost_nzd': annual_shed_cost,
        'total_cost_nzd': total_cost,
        'n_hours_binding': hours_shed,  # Alias for backward compatibility
        'unserved_peak_MW': unserved_peak_MW,
        'max_exceed_MW': max_exceed_MW,
        'p95_exceed_MW': p95_exceed_MW,
        'p99_exceed_MW': p99_exceed_MW,
        'hours_exceed': hours_exceed,
        'hours_shed': hours_shed,
        'voll_nzd_per_MWh_used': voll
    }


def evaluate_future(
    incremental_base: pd.Series,
    headroom_base: pd.Series,
    upgrade_options: List[Dict],
    assumptions: Dict,
    future_row: pd.Series,
    dt_h: float = 1.0
) -> Dict:
    """
    Evaluate a single future across all upgrade options and return the best.
    
    Args:
        incremental_base: Base incremental electricity demand (MW)
        headroom_base: Base headroom (MW)
        upgrade_options: List of upgrade option dicts
        assumptions: Assumptions dict
        future_row: Series with uncertainty multipliers
        dt_h: Timestep in hours
        
    Returns:
        Dict with selected upgrade and metrics (best option only)
    """
    # Apply uncertainties
    inc_mult = future_row.get('U_inc_mult', 1.0)
    hr_mult = future_row.get('U_headroom_mult', 1.0)
    
    incremental = incremental_base * inc_mult
    headroom = headroom_base * hr_mult
    
    # Evaluate each upgrade option and find best
    best_option = None
    best_total_cost = float('inf')
    best_metrics = None
    
    for upgrade in upgrade_options:
        metrics = evaluate_upgrade_option(
            incremental, headroom, upgrade, assumptions, future_row, dt_h
        )
        
        if metrics['total_cost_nzd'] < best_total_cost:
            best_option = upgrade['name']
            best_total_cost = metrics['total_cost_nzd']
            best_metrics = metrics
    
    return best_metrics


def run_rdm_screen(
    incremental_csv: PathlibPath,
    futures_csv: PathlibPath,
    upgrade_toml: PathlibPath,
    headroom_csv: Optional[PathlibPath] = None,
    epoch_tag: str = '2035_EB',
    strategy_id: str = 'S_AUTO',
    strategy_label: str = 'Auto-select upgrade (min cost)'
) -> pd.DataFrame:
    """
    Run RDM screening for a single epoch.
    
    Args:
        incremental_csv: Path to incremental electricity CSV
        futures_csv: Path to futures CSV
        upgrade_toml: Path to upgrade menu TOML
        headroom_csv: Optional path to headroom CSV (if None, generates PoC headroom)
        epoch_tag: Epoch tag (e.g., '2035_EB')
        strategy_id: Strategy ID
        strategy_label: Strategy label
        
    Returns:
        DataFrame with results (one row per future)
    """
    # Load data
    print(f"[RDM] Loading data for {epoch_tag}...")
    incremental_df = load_incremental_electricity(incremental_csv)
    futures_df = load_futures_csv(futures_csv)
    upgrade_options, assumptions = load_upgrade_menu(upgrade_toml)
    
    # Get timestamps
    timestamps = incremental_df['timestamp_utc'].values
    timestamps = pd.DatetimeIndex(timestamps)
    
    # Load or generate headroom
    headroom_series = load_or_generate_headroom(timestamps, headroom_csv)
    
    # Align headroom with incremental timestamps
    headroom_aligned = headroom_series.reindex(timestamps, method='nearest').fillna(0.0)
    incremental_series = pd.Series(incremental_df['incremental_electricity_MW'].values, index=timestamps)
    
    # Validate timestep
    dt_h = 1.0  # Assume hourly
    if len(timestamps) > 1:
        dt_h = (timestamps[1] - timestamps[0]).total_seconds() / 3600.0
    
    # Evaluate each future
    print(f"[RDM] Evaluating {len(futures_df)} futures...")
    results = []
    
    for idx, future_row in futures_df.iterrows():
        future_id = int(future_row['future_id'])
        metrics = evaluate_future(
            incremental_series,
            headroom_aligned,
            upgrade_options,
            assumptions,
            future_row,
            dt_h
        )
        
        # Add metadata
        metrics['epoch_tag'] = epoch_tag
        metrics['strategy_id'] = strategy_id
        metrics['strategy_label'] = strategy_label
        metrics['future_id'] = future_id
        
        results.append(metrics)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    col_order = [
        'epoch_tag', 'strategy_id', 'strategy_label', 'future_id',
        'selected_upgrade_name', 'selected_capacity_MW', 'annualised_upgrade_cost_nzd',
        'annual_incremental_MWh', 'annual_shed_MWh', 'shed_fraction',
        'annual_shed_cost_nzd', 'total_cost_nzd',
        'n_hours_binding', 'unserved_peak_MW'
    ]
    results_df = results_df[[col for col in col_order if col in results_df.columns]]
    
    print(f"[OK] RDM screening complete: {len(results_df)} futures evaluated")
    return results_df


def run_rdm_matrix(
    incremental_csv: PathlibPath,
    futures_csv: PathlibPath,
    upgrade_toml: PathlibPath,
    headroom_csv: Optional[PathlibPath] = None,
    epoch_tag: str = '2035_EB',
    satisficing_tol_MWh: float = 1e-6
) -> pd.DataFrame:
    """
    Run RDM matrix evaluation: all strategies (upgrade options) for all futures.
    
    Generates a long-form table with one row per (future_id, strategy_id) combination.
    Includes all upgrade options as strategies plus S_AUTO (auto-select benchmark).
    
    Args:
        incremental_csv: Path to incremental electricity CSV
        futures_csv: Path to futures CSV
        upgrade_toml: Path to upgrade menu TOML
        headroom_csv: Optional path to headroom CSV (if None, generates PoC headroom)
        epoch_tag: Epoch tag (e.g., '2035_EB')
        satisficing_tol_MWh: Tolerance for satisficing check (default: 1e-6)
        
    Returns:
        DataFrame with matrix results (one row per future_id, strategy_id)
    """
    # Load data
    print(f"[RDM Matrix] Loading data for {epoch_tag}...")
    incremental_df = load_incremental_electricity(incremental_csv)
    futures_df = load_futures_csv(futures_csv)
    upgrade_options, assumptions = load_upgrade_menu(upgrade_toml)
    
    # Get timestamps
    timestamps = incremental_df['timestamp_utc'].values
    timestamps = pd.DatetimeIndex(timestamps)
    
    # Load or generate headroom
    headroom_series = load_or_generate_headroom(timestamps, headroom_csv)
    
    # Align headroom with incremental timestamps
    headroom_aligned = headroom_series.reindex(timestamps, method='nearest').fillna(0.0)
    incremental_series = pd.Series(incremental_df['incremental_electricity_MW'].values, index=timestamps)
    
    # Validate timestep
    dt_h = 1.0  # Assume hourly
    if len(timestamps) > 1:
        dt_h = (timestamps[1] - timestamps[0]).total_seconds() / 3600.0
    
    # Get menu max capacity
    menu_max_capacity_MW = max([float(opt.get('capacity_MW', 0)) for opt in upgrade_options])
    
    # Create strategy list: one per upgrade option + S_AUTO
    # Strategy IDs: S0, S1, S2, ... for upgrade options, S_AUTO for auto-select
    strategies = []
    for i, upgrade in enumerate(upgrade_options):
        upgrade_name = upgrade['name']
        strategy_id = f"S{i}"
        # Create readable label
        if upgrade_name == 'none':
            strategy_label = "No upgrade (allow shedding)"
        else:
            strategy_label = f"Force {upgrade_name}"
        
        strategies.append({
            'strategy_id': strategy_id,
            'strategy_label': strategy_label,
            'upgrade': upgrade,
            'is_benchmark': False,
            'is_policy': True
        })
    
    # Add S_AUTO strategy
    strategies.append({
        'strategy_id': 'S_AUTO',
        'strategy_label': 'Auto-select upgrade (min cost)',
        'upgrade': None,  # Will be selected per future
        'is_benchmark': True,
        'is_policy': False
    })
    
    # Evaluate all futures × all strategies
    print(f"[RDM Matrix] Evaluating {len(futures_df)} futures × {len(strategies)} strategies...")
    matrix_rows = []
    
    for idx, future_row in futures_df.iterrows():
        future_id = int(future_row['future_id'])
        
        # Apply uncertainties to base series
        inc_mult = future_row.get('U_inc_mult', 1.0)
        hr_mult = future_row.get('U_headroom_mult', 1.0)
        
        incremental = incremental_series * inc_mult
        headroom = headroom_aligned * hr_mult
        
        # Evaluate all policy strategies (upgrade options)
        policy_results = {}
        for strategy in strategies:
            if strategy['strategy_id'] == 'S_AUTO':
                continue  # Handle separately
            
            upgrade = strategy['upgrade']
            metrics = evaluate_upgrade_option(
                incremental, headroom, upgrade, assumptions, future_row, dt_h
            )
            
            # Add strategy metadata
            metrics['epoch_tag'] = epoch_tag
            metrics['strategy_id'] = strategy['strategy_id']
            metrics['strategy_label'] = strategy['strategy_label']
            metrics['future_id'] = future_id
            metrics['U_inc_mult'] = inc_mult
            metrics['U_headroom_mult'] = hr_mult
            metrics['U_upgrade_capex_mult'] = future_row.get('U_upgrade_capex_mult', 1.0)
            metrics['U_voll'] = future_row.get('U_voll', 10000.0)
            metrics['U_consents_uplift'] = future_row.get('U_consents_uplift', 1.25)
            metrics['is_benchmark'] = False
            metrics['is_policy'] = True
            metrics['menu_max_capacity_MW'] = menu_max_capacity_MW
            metrics['menu_max_selected'] = (metrics['selected_capacity_MW'] == menu_max_capacity_MW)
            
            # Satisficing check
            metrics['satisficing_pass'] = (
                metrics['annual_shed_MWh'] <= satisficing_tol_MWh and
                metrics['hours_exceed'] == 0
            )
            
            policy_results[strategy['strategy_id']] = metrics
        
        # Evaluate S_AUTO: find best policy strategy
        # If ties in total_cost_nzd, choose smallest capacity
        def sort_key(sid):
            metrics = policy_results[sid]
            return (metrics['total_cost_nzd'], metrics['selected_capacity_MW'])
        
        best_policy_id = min(policy_results.keys(), key=sort_key)
        best_policy_metrics = policy_results[best_policy_id].copy()
        
        # Create S_AUTO row
        auto_metrics = best_policy_metrics.copy()
        auto_metrics['strategy_id'] = 'S_AUTO'
        auto_metrics['strategy_label'] = 'Auto-select upgrade (min cost)'
        auto_metrics['is_benchmark'] = True
        auto_metrics['is_policy'] = False
        # Satisficing_pass is already set from best_policy_metrics, which is correct
        
        # Add all results to matrix
        for strategy in strategies:
            if strategy['strategy_id'] == 'S_AUTO':
                matrix_rows.append(auto_metrics)
            else:
                matrix_rows.append(policy_results[strategy['strategy_id']])
    
    # Create matrix DataFrame
    matrix_df = pd.DataFrame(matrix_rows)
    
    # Initialize regret columns
    matrix_df['regret_vs_benchmark_nzd'] = np.nan
    matrix_df['regret_vs_best_policy_nzd'] = np.nan
    matrix_df['regret'] = np.nan
    
    # Compute regret metrics
    # Group by future_id to compute regret within each future
    for future_id in matrix_df['future_id'].unique():
        future_mask = matrix_df['future_id'] == future_id
        future_data = matrix_df[future_mask].copy()
        
        # Get benchmark (S_AUTO) cost
        benchmark_row = future_data[future_data['strategy_id'] == 'S_AUTO']
        if len(benchmark_row) > 0:
            benchmark_cost = benchmark_row.iloc[0]['total_cost_nzd']
            
            # Get best policy cost (excluding S_AUTO)
            policy_data = future_data[future_data['is_policy'] == True]
            if len(policy_data) > 0:
                best_policy_cost = policy_data['total_cost_nzd'].min()
            else:
                best_policy_cost = benchmark_cost
            
            # Compute regret for all strategies in this future
            for idx in future_data.index:
                strategy_cost = matrix_df.loc[idx, 'total_cost_nzd']
                matrix_df.loc[idx, 'regret_vs_benchmark_nzd'] = strategy_cost - benchmark_cost
                matrix_df.loc[idx, 'regret_vs_best_policy_nzd'] = strategy_cost - best_policy_cost
                matrix_df.loc[idx, 'regret'] = strategy_cost - benchmark_cost  # Default to vs benchmark
    
    # Reorder columns
    col_order = [
        'epoch_tag', 'strategy_id', 'strategy_label', 'future_id',
        'U_inc_mult', 'U_headroom_mult', 'U_upgrade_capex_mult', 'U_voll', 'U_consents_uplift',
        'selected_upgrade_name', 'selected_capacity_MW', 'annualised_upgrade_cost_nzd',
        'annual_incremental_MWh', 'annual_shed_MWh', 'shed_fraction', 'annual_shed_cost_nzd',
        'total_cost_nzd',
        'max_exceed_MW', 'p95_exceed_MW', 'p99_exceed_MW', 'hours_exceed', 'hours_shed',
        'menu_max_capacity_MW', 'menu_max_selected',
        'satisficing_pass',
        'voll_nzd_per_MWh_used',
        'is_benchmark', 'is_policy',
        'regret_vs_benchmark_nzd', 'regret_vs_best_policy_nzd', 'regret',
        'n_hours_binding', 'unserved_peak_MW'  # Backward compatibility
    ]
    
    # Only include columns that exist
    matrix_df = matrix_df[[col for col in col_order if col in matrix_df.columns]]
    
    print(f"[OK] RDM matrix complete: {len(matrix_df)} rows ({len(futures_df)} futures × {len(strategies)} strategies)")
    return matrix_df


def main():
    """CLI entrypoint for RDM screening."""
    parser = argparse.ArgumentParser(
        description='Run RDM screening for regional electricity grid constraints'
    )
    parser.add_argument('--incremental-csv', type=PathlibPath, required=True,
                       help='Path to incremental electricity CSV')
    parser.add_argument('--futures-csv', type=PathlibPath, required=True,
                       help='Path to futures CSV')
    parser.add_argument('--upgrade-toml', type=PathlibPath, required=True,
                       help='Path to upgrade menu TOML')
    parser.add_argument('--headroom-csv', type=PathlibPath, default=None,
                       help='Optional path to headroom CSV (if missing, generates PoC headroom)')
    parser.add_argument('--epoch-tag', type=str, required=True,
                       help='Epoch tag (e.g., 2035_EB)')
    parser.add_argument('--strategy-id', type=str, default='S_AUTO',
                       help='Strategy ID (default: S_AUTO)')
    parser.add_argument('--strategy-label', type=str, default='Auto-select upgrade (min cost)',
                       help='Strategy label')
    parser.add_argument('--output-csv', type=PathlibPath, required=True,
                       help='Output CSV path')
    
    args = parser.parse_args()
    
    # Resolve paths
    incremental_csv = resolve_path(args.incremental_csv)
    futures_csv = resolve_path(args.futures_csv)
    upgrade_toml = resolve_path(args.upgrade_toml)
    headroom_csv = resolve_path(args.headroom_csv) if args.headroom_csv else None
    
    # Run screening
    results_df = run_rdm_screen(
        incremental_csv,
        futures_csv,
        upgrade_toml,
        headroom_csv,
        args.epoch_tag,
        args.strategy_id,
        args.strategy_label
    )
    
    # Write output
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n[OK] Results written to: {args.output_csv}")


if __name__ == '__main__':
    main()


