"""
RDM (Robust Decision Making) Experiment Runner

Runs regional electricity upgrade selection across multiple futures and strategies
to compute regret and robust satisficing metrics.

Syntax check: python -m py_compile src/run_rdm.py
"""

# Bootstrap: allow `python .\src\script.py` (adds repo root to sys.path)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import argparse
import tomllib
import json
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from src.path_utils import repo_root, input_root, resolve_path
from src.time_utils import parse_any_timestamp
from src.regional_electricity_poc import (
    load_grid_upgrades, select_annual_upgrade
)


def get_benchmark_strategy_id(strategies: List[Dict], metrics_config: Dict) -> str:
    """Get the benchmark strategy ID from config or auto-detect."""
    # Check config override first
    benchmark_id = metrics_config.get('benchmark_strategy_id', None)
    if benchmark_id:
        return benchmark_id
    
    # Auto-detect: S_AUTO or starts with S_AUTO
    for strategy in strategies:
        sid = str(strategy.get('strategy_id', ''))
        if sid == 'S_AUTO' or sid.startswith('S_AUTO'):
            return sid
        # Also check mode
        if strategy.get('mode', '').lower() == 'auto_select':
            return sid
    
    return None


def is_benchmark_strategy(strategy: Dict, benchmark_strategy_id: str = None) -> bool:
    """Check if a strategy is a benchmark (ex-post optimum; not deployable)."""
    if benchmark_strategy_id is None:
        return False
    sid = str(strategy.get('strategy_id', ''))
    return sid == benchmark_strategy_id or sid.startswith(benchmark_strategy_id)


def get_strategy_metadata(strategies: List[Dict], metrics_config: Dict) -> Dict[str, Dict]:
    """Get single source-of-truth strategy metadata."""
    benchmark_id = get_benchmark_strategy_id(strategies, metrics_config)
    metadata = {}
    
    for strategy in strategies:
        sid = strategy['strategy_id']
        is_benchmark = is_benchmark_strategy(strategy, benchmark_id)
        metadata[sid] = {
            'strategy_id': sid,
            'strategy_label': strategy.get('label', sid),
            'is_benchmark': is_benchmark,
            'is_policy': not is_benchmark,
            'benchmark_label': 'ex-post optimum (benchmark; not deployable)' if is_benchmark else None
        }
    
    return metadata, benchmark_id


def latin_hypercube_sample(n_samples: int, n_dims: int, bounds: List[tuple], seed: int = None) -> np.ndarray:
    """
    Generate Latin Hypercube Sample (LHS) for continuous uncertainties.
    
    Args:
        n_samples: Number of samples
        n_dims: Number of dimensions
        bounds: List of (min, max) tuples for each dimension
        seed: Random seed
    
    Returns:
        Array of shape (n_samples, n_dims) with samples in [0, 1] range
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate LHS
    samples = np.zeros((n_samples, n_dims))
    for i in range(n_dims):
        # Divide [0, 1] into n_samples intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        # Randomly sample one point from each interval
        samples[:, i] = np.random.uniform(intervals[:-1], intervals[1:])
        # Shuffle to break correlation
        np.random.shuffle(samples[:, i])
    
    # Scale to bounds
    for i, (min_val, max_val) in enumerate(bounds):
        samples[:, i] = samples[:, i] * (max_val - min_val) + min_val
    
    return samples


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load RDM experiment TOML config."""
    config_path = resolve_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    return config


def generate_futures(config: Dict[str, Any], bundle_name: str) -> pd.DataFrame:
    """
    Generate futures table from uncertainties using LHS or discrete sampling.
    
    Returns DataFrame with columns: future_id, U_<u_id>, ...
    """
    experiment = config['experiment']
    n_futures = experiment.get('n_futures', 200)
    seed = experiment.get('seed', 42)
    sampling = experiment.get('sampling', 'lhs')
    
    uncertainties = config.get('uncertainties', [])
    
    # Separate continuous and discrete uncertainties
    continuous_uncs = []
    discrete_uncs = []
    continuous_bounds = []
    
    for unc in uncertainties:
        kind = unc.get('kind', 'scalar_multiplier')
        if kind == 'discrete':
            discrete_uncs.append(unc)
        else:
            continuous_uncs.append(unc)
            if 'bounds' in unc:
                continuous_bounds.append(tuple(unc['bounds']))
            else:
                continuous_bounds.append((0.0, 1.0))  # Default
    
    # Generate continuous samples
    futures_data = {}
    if continuous_uncs:
        if sampling == 'lhs':
            continuous_samples = latin_hypercube_sample(
                n_futures, len(continuous_uncs), continuous_bounds, seed
            )
        else:  # random
            np.random.seed(seed)
            continuous_samples = np.random.uniform(
                [b[0] for b in continuous_bounds],
                [b[1] for b in continuous_bounds],
                size=(n_futures, len(continuous_uncs))
            )
        
        for i, unc in enumerate(continuous_uncs):
            futures_data[unc['u_id']] = continuous_samples[:, i]
    
    # Generate discrete samples (uniform sampling from options)
    for idx, unc in enumerate(discrete_uncs):
        options = unc.get('options', [])
        if not options:
            raise ValueError(f"Discrete uncertainty {unc['u_id']} must have 'options'")
        
        np.random.seed(seed + 1000 + idx)  # Offset seed for discrete
        futures_data[unc['u_id']] = np.random.choice(options, size=n_futures)
    
    # Create DataFrame
    futures_df = pd.DataFrame(futures_data)
    futures_df.insert(0, 'future_id', range(n_futures))
    
    return futures_df


def apply_uncertainties(headroom_base: pd.Series, incremental_base: pd.Series,
                       upgrade_options: List[Dict], assumptions: Dict,
                       uncertainties: List[Dict], future_row: pd.Series) -> tuple:
    """
    Apply uncertainty transforms to inputs.
    
    Returns: (headroom_modified, incremental_modified, upgrade_options_modified, assumptions_modified, voll)
    """
    headroom = headroom_base.copy()
    incremental = incremental_base.copy()
    
    # Deep copy upgrades and store original capex for uncertainty application
    upgrades = []
    for opt in upgrade_options:
        upgrade_copy = opt.copy()
        upgrade_copy['_original_capex_nzd'] = upgrade_copy['capex_nzd']  # Store original
        upgrades.append(upgrade_copy)
    
    assumpts = assumptions.copy()
    voll = 10000.0  # Default
    
    # First pass: collect multipliers
    capex_mult = 1.0
    consents_mult = 1.0
    
    for unc in uncertainties:
        u_id = unc['u_id']
        target = unc['target']
        value = future_row[u_id]
        
        if target == 'incremental_series':
            incremental = incremental * value
        elif target == 'headroom_series':
            headroom = headroom * value
        elif target == 'upgrade_capex_nzd':
            capex_mult = value
        elif target == 'consents_uplift_factor':
            consents_mult = value
        elif target == 'voll_nzd_per_MWh':
            voll = float(value)
    
    # Apply capex multiplier to all upgrades except "none"
    crf = assumpts.get('CRF', 0.0)
    for upgrade in upgrades:
        if upgrade['name'] != 'none' and upgrade['capacity_MW'] > 0:
            upgrade['capex_nzd'] = upgrade['_original_capex_nzd'] * capex_mult
    
    # Apply consents uplift multiplier
    assumpts['consents_uplift_factor'] = assumpts.get('consents_uplift_factor', 1.25) * consents_mult
    
    # Recompute annualised costs for all upgrades
    for upgrade in upgrades:
        upgrade['annualised_cost_nzd'] = upgrade['capex_nzd'] * crf * assumpts['consents_uplift_factor']
        # Remove temporary field
        if '_original_capex_nzd' in upgrade:
            del upgrade['_original_capex_nzd']
    
    return headroom, incremental, upgrades, assumpts, voll


def write_run_metadata(run_dir: Path, config_path: Path, experiment: Dict, epoch_tag: str,
                      bundle_name: str, headroom_csv: Path, incremental_csv: Path,
                      upgrade_toml: Path) -> None:
    """Write run_metadata.json breadcrumb file."""
    metadata = {
        'config_path': str(config_path),
        'experiment_name': experiment.get('name', 'unknown'),
        'epoch_tag': epoch_tag,
        'bundle': bundle_name,
        'headroom_csv': str(headroom_csv),
        'incremental_csv': str(incremental_csv),
        'upgrade_toml': str(upgrade_toml),
        'timestamp_utc': datetime.now(timezone.utc).isoformat()
    }
    
    metadata_path = run_dir / 'run_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Created run metadata: {metadata_path}")


def run_rdm_experiment(config_path: str, bundle_name: str, n_futures_override: int = None,
                       seed_override: int = None, clean: bool = False,
                       save_timeseries: str = 'none', dry_run: bool = False) -> None:
    """
    Run RDM experiment: generate futures, evaluate strategies, compute regret.
    
    Args:
        config_path: Path to experiment TOML
        bundle_name: Bundle name for output paths
        n_futures_override: Override n_futures from config
        seed_override: Override seed from config
        clean: Delete output folder before running
        save_timeseries: 'representative' | 'all' | 'none'
        dry_run: If True, create run_dir and metadata, then exit
    """
    print(f"[RDM] Loading experiment config from {config_path}...")
    config = load_experiment_config(config_path)
    config_path_resolved = resolve_path(config_path)
    
    experiment = config['experiment']
    exp_name = experiment['name']
    epoch_tag = experiment['epoch_tag']
    
    # Override n_futures and seed if provided
    if n_futures_override:
        experiment['n_futures'] = n_futures_override
    if seed_override:
        experiment['seed'] = seed_override
    
    # Resolve output path and create run_dir EARLY (before loading CSVs)
    output_root_str = config.get('outputs', {}).get('output_root', f'Output/runs/<BUNDLE>/epoch{epoch_tag}/dispatch_prop_v2/rdm')
    output_root = Path(output_root_str.replace('<BUNDLE>', bundle_name))
    run_dir = output_root / exp_name
    
    if clean and run_dir.exists():
        print(f"[RDM] Cleaning existing run directory: {run_dir}")
        shutil.rmtree(run_dir)
    
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Resolve paths (for metadata)
    paths = config['paths']
    headroom_csv = resolve_path(paths['headroom_csv'].replace('<BUNDLE>', bundle_name))
    incremental_csv = resolve_path(paths['incremental_csv'].replace('<BUNDLE>', bundle_name))
    upgrade_toml = resolve_path(paths['upgrade_menu_toml'])
    
    # Write metadata breadcrumb immediately
    write_run_metadata(run_dir, config_path_resolved, experiment, epoch_tag, bundle_name,
                      headroom_csv, incremental_csv, upgrade_toml)
    
    # Save config copy
    shutil.copy(config_path_resolved, run_dir / 'config_used.toml')
    
    # Dry run: exit after creating metadata
    if dry_run:
        print(f"[DRY-RUN] Resolved paths:")
        print(f"  Headroom CSV: {headroom_csv}")
        print(f"  Incremental CSV: {incremental_csv}")
        print(f"  Upgrade TOML: {upgrade_toml}")
        print(f"  Run directory: {run_dir}")
        print(f"[OK] Dry run complete. Exiting.")
        return
    
    # Load upgrade menu early for validation
    print(f"[RDM] Loading upgrade menu...")
    if not upgrade_toml.exists():
        raise FileNotFoundError(f"Upgrade menu TOML not found: {upgrade_toml}")
    
    upgrade_options, assumptions = load_grid_upgrades(upgrade_toml)
    upgrade_names = [opt['name'] for opt in upgrade_options]
    
    # Get strategies and validate upgrade names early
    strategies = config.get('strategies', [])
    if not strategies:
        if 'decision_levers' in config:
            strategies = config['decision_levers'].get('strategies', [])
    
    if not strategies:
        raise ValueError("No strategies found in config. Expected [[strategies]] entries.")
    
    # Validate force_upgrade_name for each strategy
    for strategy in strategies:
        if strategy.get('mode') == 'force':
            force_name = strategy.get('force_upgrade_name')
            if force_name and force_name not in upgrade_names:
                available_str = ', '.join(upgrade_names) if upgrade_names else '(none)'
                raise ValueError(
                    f"Strategy {strategy['strategy_id']}: Unknown upgrade name '{force_name}'. "
                    f"Available upgrade names in menu: {available_str}"
                )
    
    uncertainties = config.get('uncertainties', [])
    metrics_config = config.get('metrics', {})
    
    # Apply satisficing profile if specified
    profile = metrics_config.get('satisficing_profile', None)
    if profile:
        profiles = {
            'strict_grid': {
                'annual_shed_MWh_max': 0.0,
                'shed_fraction_max': 0.0,
                'max_exceed_MW_max': 0.1,
                'hours_exceed_max': 0,
                'p95_exceed_MW_max': 0.0
            },
            'tolerant_ops': {
                'annual_shed_MWh_max': 0.0,
                'shed_fraction_max': 0.005,
                'max_exceed_MW_max': 1.0,
                'hours_exceed_max': 10,
                'p95_exceed_MW_max': None
            },
            'demo': {
                'annual_shed_MWh_max': 0.0,
                'shed_fraction_max': 0.001,
                'max_exceed_MW_max': 0.5,
                'hours_exceed_max': 5,
                'p95_exceed_MW_max': 0.0
            }
        }
        if profile in profiles:
            for key, value in profiles[profile].items():
                if key not in metrics_config:  # Profile values only apply if not explicitly set
                    metrics_config[key] = value
    
    # Set default metrics thresholds (defensible defaults)
    if 'annual_shed_MWh_max' not in metrics_config:
        metrics_config['annual_shed_MWh_max'] = metrics_config.get('shed_MWh_max', 0.0)  # Backward compat
    if 'shed_MWh_max' not in metrics_config:
        metrics_config['shed_MWh_max'] = metrics_config.get('annual_shed_MWh_max', 0.0)
    if 'max_exceed_MW_max' not in metrics_config:
        metrics_config['max_exceed_MW_max'] = 0.1
    if 'shed_fraction_max' not in metrics_config:
        metrics_config['shed_fraction_max'] = 0.0
    if 'p95_exceed_MW_max' not in metrics_config:
        metrics_config['p95_exceed_MW_max'] = None
    if 'hours_exceed_max' not in metrics_config:
        metrics_config['hours_exceed_max'] = None
    
    # Load base data with clear error messages
    print(f"[RDM] Loading base data...")
    try:
        if not headroom_csv.exists():
            raise FileNotFoundError(
                f"Headroom CSV not found: {headroom_csv}\n"
                f"  Expected pattern: {paths['headroom_csv']} (with <BUNDLE>={bundle_name})"
            )
        headroom_df = pd.read_csv(headroom_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise
    
    try:
        if not incremental_csv.exists():
            raise FileNotFoundError(
                f"Incremental CSV not found: {incremental_csv}\n"
                f"  Expected pattern: {paths['incremental_csv']} (with <BUNDLE>={bundle_name}, epoch={epoch_tag})\n"
                f"  Ensure site dispatch has been run and incremental electricity CSV exported."
            )
        incremental_df = pd.read_csv(incremental_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise
    
    # Parse timestamps and align
    headroom_ts_col = paths['timestamp_col']
    incremental_ts_col = 'timestamp_utc' if 'timestamp_utc' in incremental_df.columns else 'timestamp'
    
    headroom_df[headroom_ts_col] = parse_any_timestamp(headroom_df[headroom_ts_col])
    incremental_df[incremental_ts_col] = parse_any_timestamp(incremental_df[incremental_ts_col])
    
    # Merge on timestamps
    merged = pd.merge(
        headroom_df[[headroom_ts_col, paths['headroom_col']]],
        incremental_df[[incremental_ts_col, paths['incremental_col']]],
        left_on=headroom_ts_col, right_on=incremental_ts_col,
        how='inner'
    )
    
    if len(merged) == 0:
        raise ValueError("No matching timestamps between headroom and incremental CSVs")
    
    headroom_base = pd.to_numeric(merged[paths['headroom_col']], errors='coerce').fillna(0.0)
    incremental_base = pd.to_numeric(merged[paths['incremental_col']], errors='coerce').fillna(0.0)
    
    # After merge, the timestamp column from the left (headroom) is kept
    # Use the headroom timestamp column (which should match the config timestamp_col)
    ts_col = headroom_ts_col
    if ts_col not in merged.columns:
        # Fallback: try config timestamp_col or common names
        for candidate in [paths.get('timestamp_col', 'timestamp_utc'), 'timestamp_utc', 'timestamp']:
            if candidate in merged.columns:
                ts_col = candidate
                break
        else:
            raise KeyError(f"Timestamp column not found in merged DataFrame. Available columns: {merged.columns.tolist()}")
    
    timestamps = merged[ts_col]
    
    # Parse timestamps to datetime (UTC) and validate
    timestamps = pd.to_datetime(timestamps, utc=True, errors='raise')
    
    # Validate: monotonic increasing
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonic increasing")
    
    # Validate: no duplicates
    if timestamps.duplicated().any():
        raise ValueError(f"Timestamps contain duplicates: {timestamps[timestamps.duplicated()].tolist()[:5]}")
    
    # Validate: constant timestep
    deltas = timestamps.diff().dropna()
    deltas_h = deltas.dt.total_seconds() / 3600.0
    
    if deltas_h.nunique() != 1:
        raise ValueError(f"Timestep is not constant. Unique timesteps (hours): {sorted(deltas_h.unique())[:10]}")
    
    dt_h = float(deltas_h.iloc[0])
    
    # Compare to expected dt_h from config
    dt_h_expected = float(experiment.get('dt_h', 1.0))
    if abs(dt_h - dt_h_expected) > 1e-6:
        raise ValueError(f"Timestep mismatch: computed dt_h={dt_h:.6f} h, expected dt_h={dt_h_expected:.6f} h")
    
    print(f"[OK] Timestamp validation passed: {len(timestamps)} timesteps, dt_h={dt_h:.6f} h")
    
    # Generate futures
    print(f"[RDM] Generating {experiment['n_futures']} futures...")
    futures_df = generate_futures(config, bundle_name)
    futures_df.to_csv(run_dir / 'futures.csv', index=False)
    print(f"[OK] Saved futures to {run_dir / 'futures.csv'}")
    
    # Core loop: futures x strategies
    print(f"[RDM] Running {len(futures_df)} futures x {len(strategies)} strategies...")
    results = []
    ledger_entries = []
    
    total_runs = len(futures_df) * len(strategies)
    run_count = 0
    
    for future_idx, future_row in futures_df.iterrows():
        # Apply uncertainties
        headroom_mod, incremental_mod, upgrades_mod, assumpts_mod, voll = apply_uncertainties(
            headroom_base, incremental_base, upgrade_options, assumptions,
            uncertainties, future_row
        )
        
        for strategy in strategies:
            strategy_id = strategy['strategy_id']
            strategy_label = strategy.get('label', strategy_id)
            mode = strategy.get('mode', 'auto_select')
            force_name = strategy.get('force_upgrade_name', None)
            
            # Filter upgrades if force mode
            upgrades_to_use = upgrades_mod
            if mode == 'force' and force_name:
                # Filter to {none, force_name}
                filtered = []
                none_found = False
                force_found = False
                for opt in upgrades_mod:
                    opt_name = opt.get('name', '')
                    opt_capacity = opt.get('capacity_MW', 0)
                    # Check if this is the "none" option
                    if opt_name == 'none' or opt_capacity == 0:
                        if force_name == 'none' or opt_name == 'none':
                            filtered.append(opt)
                            none_found = True
                            if force_name == 'none':
                                force_found = True
                    # Check if this is the forced upgrade
                    elif opt_name == force_name:
                        filtered.append(opt)
                        force_found = True
                
                if not force_found:
                    available = [o.get('name', 'unknown') for o in upgrades_mod]
                    raise ValueError(f"Forced upgrade '{force_name}' not found in menu. Available: {available}")
                if not none_found and force_name != 'none' and len(filtered) == 1:
                    print(f"[WARN] 'none' option not found in menu, using only forced upgrade '{force_name}'")
                
                upgrades_to_use = filtered
            
            # Run selection
            try:
                selected = select_annual_upgrade(
                    headroom_base=headroom_mod,
                    incremental_MW=incremental_mod,
                    upgrade_options=upgrades_to_use,
                    voll_nzd_per_MWh=voll,
                    dt_h=dt_h
                )
                
                # Compute metrics
                headroom_eff = headroom_mod + selected['capacity_MW']
                exceed_MW = (incremental_mod - headroom_eff).clip(lower=0.0)
                max_exceed_MW = float(exceed_MW.max())
                p95_exceed_MW = float(np.quantile(exceed_MW, 0.95)) if len(exceed_MW) > 0 else 0.0
                p99_exceed_MW = float(np.quantile(exceed_MW, 0.99)) if len(exceed_MW) > 0 else 0.0
                annual_shed_cost_nzd = selected['annual_shed_MWh'] * voll
                menu_max_capacity_MW = max(opt['capacity_MW'] for opt in upgrades_to_use)
                menu_max_selected = (selected['capacity_MW'] == menu_max_capacity_MW)
                
                # Derived metrics
                annual_incremental_MWh = float((incremental_mod * dt_h).sum())
                eps = 1e-6  # Small epsilon to avoid division by zero
                shed_fraction = selected['annual_shed_MWh'] / max(annual_incremental_MWh, eps)
                
                # Reliability metrics
                max_exceed_MW_tolerance = metrics_config.get('max_exceed_MW_max', 0.1)
                hours_exceed = int((exceed_MW > max_exceed_MW_tolerance).sum())
                # hours_shed: count hours where shed_MW > 0 (using exceed_MW as proxy for shed)
                hours_shed = int((exceed_MW > 0).sum())
                
                # Satisficing check (use defaults from metrics_config)
                shed_MWh_max = metrics_config.get('shed_MWh_max', 0.0)
                max_exceed_MW_max = metrics_config.get('max_exceed_MW_max', 0.01)
                shed_fraction_max = metrics_config.get('shed_fraction_max', None)
                p95_exceed_MW_max = metrics_config.get('p95_exceed_MW_max', None)
                
                satisficing_pass = (
                    selected['annual_shed_MWh'] <= shed_MWh_max and
                    max_exceed_MW <= max_exceed_MW_max
                )
                if shed_fraction_max is not None:
                    satisficing_pass = satisficing_pass and (shed_fraction <= shed_fraction_max)
                if p95_exceed_MW_max is not None:
                    satisficing_pass = satisficing_pass and (p95_exceed_MW <= p95_exceed_MW_max)
                
                # Determine if this is a benchmark strategy (will be set via metadata later)
                # Note: is_benchmark will be added to result_row from metadata
                
                # Record result
                result_row = {
                    'epoch_tag': epoch_tag,
                    'strategy_id': strategy_id,
                    'strategy_label': strategy_label,
                    'future_id': int(future_row['future_id']),
                    **{u['u_id']: float(future_row[u['u_id']]) for u in uncertainties},
                    'selected_upgrade_name': selected['name'],
                    'selected_capacity_MW': selected['capacity_MW'],
                    'annualised_upgrade_cost_nzd': selected['annualised_cost_nzd'],
                    'annual_incremental_MWh': annual_incremental_MWh,
                    'annual_shed_MWh': selected['annual_shed_MWh'],
                    'shed_fraction': shed_fraction,
                    'annual_shed_cost_nzd': annual_shed_cost_nzd,
                    'total_cost_nzd': selected['total_cost_nzd'],
                    'max_exceed_MW': max_exceed_MW,
                    'p95_exceed_MW': p95_exceed_MW,
                    'p99_exceed_MW': p99_exceed_MW,
                    'hours_exceed': hours_exceed,
                    'hours_shed': hours_shed,
                    'menu_max_capacity_MW': menu_max_capacity_MW,
                    'menu_max_selected': menu_max_selected,
                    'satisficing_pass': bool(satisficing_pass),
                    'voll_nzd_per_MWh_used': voll
                }
                results.append(result_row)
                
                # Ledger entry
                ledger_entry = {
                    'future_id': int(future_row['future_id']),
                    'strategy_id': strategy_id,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'selected_upgrade': selected['name'],
                    'total_cost_nzd': float(selected['total_cost_nzd']),
                    'dt_h': float(dt_h),
                    'status': 'success'
                }
                ledger_entries.append(ledger_entry)
                
                run_count += 1
                if run_count % 50 == 0:
                    print(f"  Progress: {run_count}/{total_runs} runs completed...")
                
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] Future {future_row['future_id']}, Strategy {strategy_id}: {error_msg}")
                # Record error row
                result_row = {
                    'epoch_tag': epoch_tag,
                    'strategy_id': strategy_id,
                    'strategy_label': strategy_label,
                    'future_id': int(future_row['future_id']),
                    **{u['u_id']: float(future_row[u['u_id']]) for u in uncertainties},
                    'error': error_msg
                }
                results.append(result_row)
                
                # Ledger entry for error
                ledger_entry = {
                    'future_id': int(future_row['future_id']),
                    'strategy_id': strategy_id,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'error': error_msg,
                    'status': 'error'
                }
                ledger_entries.append(ledger_entry)
                
                run_count += 1
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Get strategy metadata for regret computation
    strategy_metadata, benchmark_strategy_id = get_strategy_metadata(strategies, metrics_config)
    
    # Add strategy metadata columns
    if len(summary_df) > 0:
        summary_df['is_benchmark'] = summary_df['strategy_id'].map(lambda sid: strategy_metadata.get(sid, {}).get('is_benchmark', False))
        summary_df['is_policy'] = summary_df['strategy_id'].map(lambda sid: strategy_metadata.get(sid, {}).get('is_policy', True))
    
    # Compute regret columns (robust to NaNs)
    if metrics_config.get('compute_regret', True) and len(summary_df) > 0:
        # Regret vs benchmark (if benchmark exists)
        if benchmark_strategy_id and benchmark_strategy_id in summary_df['strategy_id'].values:
            benchmark_costs = summary_df[summary_df['strategy_id'] == benchmark_strategy_id].set_index('future_id')['total_cost_nzd']
            summary_df['regret_vs_benchmark_nzd'] = summary_df.apply(
                lambda row: row['total_cost_nzd'] - benchmark_costs.get(row['future_id'], np.nan)
                if pd.notna(row['total_cost_nzd']) else np.nan,
                axis=1
            )
        else:
            summary_df['regret_vs_benchmark_nzd'] = np.nan
        
        # Regret vs best policy (minimum cost among non-benchmark strategies per future)
        policy_df = summary_df[summary_df['is_policy']].copy()
        if len(policy_df) > 0:
            policy_min_costs = policy_df.groupby('future_id')['total_cost_nzd'].min()
            summary_df['regret_vs_best_policy_nzd'] = summary_df.apply(
                lambda row: row['total_cost_nzd'] - policy_min_costs.get(row['future_id'], np.nan)
                if pd.notna(row['total_cost_nzd']) and row['is_policy'] else np.nan,
                axis=1
            )
        else:
            summary_df['regret_vs_best_policy_nzd'] = np.nan
        
        # Legacy regret column (alias to regret_vs_best_policy_nzd for ranking)
        summary_df['regret'] = summary_df['regret_vs_best_policy_nzd']
    
    # Save summary CSV (use run_dir, not a subdirectory)
    summary_csv_name = config.get('outputs', {}).get('summary_csv_name', 'rdm_summary.csv')
    summary_path = run_dir / summary_csv_name
    summary_df.to_csv(summary_path, index=False)
    print(f"[OK] Saved summary to {summary_path}")
    
    # Save run ledger (JSONL)
    if config.get('outputs', {}).get('save_run_ledger', True):
        ledger_path = run_dir / 'run_ledger.jsonl'
        with open(ledger_path, 'w') as f:
            for entry in ledger_entries:
                f.write(json.dumps(entry) + '\n')
        print(f"[OK] Saved run ledger to {ledger_path}")
    
    # Compute robust summary
    robust_summary = compute_robust_summary(
        summary_df, strategies, metrics_config, 
        experiment, epoch_tag, config_path_resolved, len(futures_df)
    )
    
    # Save robust summary JSON
    robust_path = run_dir / 'robust_summary.json'
    with open(robust_path, 'w') as f:
        json.dump(robust_summary, f, indent=2)
    print(f"[OK] Saved robust summary to {robust_path}")
    
    # Generate robust summary table CSV
    print(f"[RDM] Generating robust summary table...")
    summary_table_df = generate_robust_summary_table(summary_df, strategies, robust_summary, metrics_config)
    table_path = run_dir / 'robust_summary_table.csv'
    summary_table_df.to_csv(table_path, index=False)
    print(f"[OK] Saved robust summary table to {table_path}")
    
    # Generate figures
    print(f"[RDM] Generating figures...")
    generate_rdm_figures(summary_df, strategies, figures_dir, metrics_config)
    
    # Save README
    write_rdm_readme(run_dir, exp_name, epoch_tag, len(futures_df), len(strategies))
    
    # Generate thesis figures index
    write_thesis_figures_md(run_dir, figures_dir, experiment, epoch_tag, robust_summary, metrics_config)
    
    # Console summary
    print(f"\n[SUMMARY] RDM experiment completed: {exp_name}")
    print(f"  Futures: {len(futures_df)}, Strategies: {len(strategies)}")
    
    # Count valid rows
    if 'total_cost_nzd' in summary_df.columns:
        valid_count = summary_df['total_cost_nzd'].notna().sum()
        print(f"  Valid rows: {valid_count}/{len(summary_df)}")
    else:
        valid_count = len(summary_df)
        print(f"  Valid rows: {valid_count}/{len(summary_df)}")
    
    # Best strategies (from ranking pool)
    exclude_auto = metrics_config.get('exclude_auto_from_ranking', True)
    if exclude_auto and robust_summary.get('benchmark_strategy_ids'):
        print(f"  Note: AUTO strategies excluded from ranking (benchmark only)")
    
    if 'best_strategy_by_minimax_regret' in robust_summary:
        best_sid = robust_summary['best_strategy_by_minimax_regret']
        best_label = robust_summary.get('strategy_labels', {}).get(best_sid, best_sid)
        max_regret = robust_summary.get('regret_by_strategy', {}).get(best_sid, {}).get('max_regret', np.nan)
        failure_rate = robust_summary.get('failure_rates', {}).get(best_sid, 0.0)
        print(f"  Best by minimax regret: {best_label} (max_regret={max_regret/1e9:.2f}B NZD, failure_rate={failure_rate:.1%})")
    
    if 'best_strategy_by_max_satisficing' in robust_summary:
        best_sid = robust_summary['best_strategy_by_max_satisficing']
        best_label = robust_summary.get('strategy_labels', {}).get(best_sid, best_sid)
        print(f"  Best by max satisficing: {best_label}")
    
    # Top-2 strategies by minimax regret
    if 'regret_by_strategy' in robust_summary:
        ranking_ids = robust_summary.get('ranking_strategy_ids', [])
        regret_ranking = [
            (sid, robust_summary['regret_by_strategy'][sid]['max_regret'])
            for sid in ranking_ids
            if sid in robust_summary['regret_by_strategy']
        ]
        regret_ranking.sort(key=lambda x: x[1])
        if len(regret_ranking) >= 2:
            print(f"  Top-2 by minimax regret:")
            for i, (sid, max_regret) in enumerate(regret_ranking[:2], 1):
                label = robust_summary.get('strategy_labels', {}).get(sid, sid)
                failure_rate = robust_summary.get('failure_rates', {}).get(sid, 0.0)
                print(f"    {i}. {label}: max_regret={max_regret/1e9:.2f}B NZD, failure_rate={failure_rate:.1%}")
    
    # Failure rates
    if 'failure_rates' in robust_summary:
        print(f"  Failure rates by strategy:")
        for sid, rate in robust_summary['failure_rates'].items():
            strategy_label = robust_summary.get('strategy_labels', {}).get(sid, sid)
            is_auto = sid in robust_summary.get('benchmark_strategy_ids', [])
            if is_auto:
                strategy_label += ' (benchmark)'
            print(f"    {strategy_label}: {rate:.1%}")
    
    print(f"  Output directory: {run_dir}")


def compute_robust_summary(summary_df: pd.DataFrame, strategies: List[Dict],
                          metrics_config: Dict, experiment: Dict = None,
                          epoch_tag: str = None, config_path: Path = None,
                          n_futures: int = None) -> Dict[str, Any]:
    """Compute robust summary statistics."""
    robust = {}
    
    # Filter out error rows (robust filtering)
    if 'total_cost_nzd' in summary_df.columns:
        valid_df = summary_df.dropna(subset=['total_cost_nzd']).copy()
    else:
        valid_df = summary_df.copy()
    
    if len(valid_df) == 0:
        print("[WARN] No valid results (all rows have NaN total_cost_nzd)")
        return {'error': 'No valid results'}
    
    # Get strategy metadata
    strategy_metadata, benchmark_strategy_id = get_strategy_metadata(strategies, metrics_config)
    
    # Identify benchmark and ranking strategies
    exclude_benchmark = metrics_config.get('exclude_benchmark_from_ranking', metrics_config.get('exclude_auto_from_ranking', True))
    benchmark_strategy_ids = []
    ranking_strategy_ids = []
    
    for strategy in strategies:
        sid = strategy['strategy_id']
        is_benchmark = strategy_metadata.get(sid, {}).get('is_benchmark', False)
        if is_benchmark:
            benchmark_strategy_ids.append(sid)
        else:
            ranking_strategy_ids.append(sid)
    
    benchmark_strategy_ids = sorted(benchmark_strategy_ids)
    ranking_strategy_ids = sorted(ranking_strategy_ids)
    
    # Store strategy classification
    robust['exclude_benchmark_from_ranking'] = exclude_benchmark
    robust['exclude_auto_from_ranking'] = exclude_benchmark  # Backward compat alias
    robust['benchmark_strategy_id'] = benchmark_strategy_id
    robust['ranking_strategy_ids'] = ranking_strategy_ids
    robust['benchmark_strategy_ids'] = benchmark_strategy_ids
    
    # Determine regret reference
    if benchmark_strategy_id and benchmark_strategy_id in valid_df['strategy_id'].values:
        regret_reference_for_reporting = 'benchmark'
    else:
        regret_reference_for_reporting = 'best_policy'
    robust['regret_reference_for_ranking'] = 'best_policy'
    robust['regret_reference_for_reporting'] = regret_reference_for_reporting
    
    # Create filtered dataframe for ranking (exclude benchmark if flag is set)
    benchmark_set = set(benchmark_strategy_ids)
    ranking_df = valid_df[~valid_df['strategy_id'].isin(benchmark_set)].copy() if exclude_benchmark and benchmark_set else valid_df.copy()
    
    # Quantiles by strategy
    quantiles = metrics_config.get('report_quantiles', [0.05, 0.50, 0.95])
    robust['quantiles_by_strategy'] = {}
    
    for strategy in strategies:
        sid = strategy['strategy_id']
        strat_df = valid_df[valid_df['strategy_id'] == sid]
        if len(strat_df) == 0:
            continue
        
        robust['quantiles_by_strategy'][sid] = {
            'total_cost_nzd': {
                f'p{int(q*100):02d}': float(np.quantile(strat_df['total_cost_nzd'], q))
                for q in quantiles
            },
            'annual_shed_MWh': {
                f'p{int(q*100):02d}': float(np.quantile(strat_df['annual_shed_MWh'], q))
                for q in quantiles
            }
        }
    
    # Regret metrics (use regret_vs_best_policy_nzd for ranking)
    regret_col_policy = 'regret_vs_best_policy_nzd' if 'regret_vs_best_policy_nzd' in valid_df.columns else 'regret'
    regret_col_benchmark = 'regret_vs_benchmark_nzd' if 'regret_vs_benchmark_nzd' in valid_df.columns else None
    
    if regret_col_policy in valid_df.columns:
        robust['regret_by_strategy'] = {}
        robust['regret_vs_benchmark_by_strategy'] = {}
        
        for strategy in strategies:
            sid = strategy['strategy_id']
            strat_df = valid_df[valid_df['strategy_id'] == sid]
            if len(strat_df) == 0:
                continue
            
            # Policy regret (for ranking)
            regret_vals_policy = strat_df[regret_col_policy].dropna()
            if len(regret_vals_policy) > 0:
                robust['regret_by_strategy'][sid] = {
                    'max_regret': float(regret_vals_policy.max()),
                    'p95_regret': float(np.quantile(regret_vals_policy, 0.95))
                }
            
            # Benchmark regret (for reporting)
            if regret_col_benchmark and regret_col_benchmark in strat_df.columns:
                regret_vals_benchmark = strat_df[regret_col_benchmark].dropna()
                if len(regret_vals_benchmark) > 0:
                    robust['regret_vs_benchmark_by_strategy'][sid] = {
                        'max_regret': float(regret_vals_benchmark.max()),
                        'p95_regret': float(np.quantile(regret_vals_benchmark, 0.95))
                    }
        
        # Best by minimax regret (EXCLUDE benchmark if flag is set, use policy regret)
        benchmark_set = set(benchmark_strategy_ids)
        if exclude_benchmark and benchmark_set:
            ranking_regret = {
                sid: robust['regret_by_strategy'][sid]['max_regret']
                for sid in robust['regret_by_strategy']
                if sid not in benchmark_set
            }
        else:
            ranking_regret = {
                sid: robust['regret_by_strategy'][sid]['max_regret']
                for sid in robust['regret_by_strategy']
            }
        if ranking_regret:
            robust['best_strategy_by_minimax_regret'] = min(ranking_regret, key=ranking_regret.get)
    
    # Satisficing rates
    if 'satisficing_pass' in valid_df.columns:
        robust['satisficing_by_strategy'] = {}
        for strategy in strategies:
            sid = strategy['strategy_id']
            strat_df = valid_df[valid_df['strategy_id'] == sid]
            if len(strat_df) == 0:
                continue
            
            pass_rate = strat_df['satisficing_pass'].mean()
            robust['satisficing_by_strategy'][sid] = {
                'pass_rate': float(pass_rate),
                'n_pass': int(strat_df['satisficing_pass'].sum()),
                'n_total': len(strat_df)
            }
        
        # Best by max satisficing, then min p95 cost (EXCLUDE benchmark if flag is set)
        if robust['satisficing_by_strategy']:
            # Sort by pass rate (desc), then p95 cost (asc)
            if exclude_benchmark and benchmark_set:
                candidates = [
                    (sid, robust['satisficing_by_strategy'][sid]['pass_rate'],
                     robust['quantiles_by_strategy'].get(sid, {}).get('total_cost_nzd', {}).get('p95', float('inf')))
                    for sid in robust['satisficing_by_strategy']
                    if sid not in benchmark_set
                ]
            else:
                candidates = [
                    (sid, robust['satisficing_by_strategy'][sid]['pass_rate'],
                     robust['quantiles_by_strategy'].get(sid, {}).get('total_cost_nzd', {}).get('p95', float('inf')))
                    for sid in robust['satisficing_by_strategy']
                ]
            candidates.sort(key=lambda x: (-x[1], x[2]))
            if candidates:
                robust['best_strategy_by_max_satisficing'] = candidates[0][0]
    
    # Failure rates (based on satisficing)
    robust['failure_rates'] = {}
    for strategy in strategies:
        sid = strategy['strategy_id']
        strat_df = valid_df[valid_df['strategy_id'] == sid]
        if len(strat_df) == 0:
            continue
        
        # Count failures (not satisficing)
        if 'satisficing_pass' in strat_df.columns:
            failures = (~strat_df['satisficing_pass']).sum()
        else:
            # Fallback: compute from thresholds
            failures = (
                (strat_df.get('annual_shed_MWh', pd.Series([0.0] * len(strat_df))) > metrics_config.get('shed_MWh_max', 0.1)) |
                (strat_df.get('max_exceed_MW', pd.Series([0.0] * len(strat_df))) > metrics_config.get('max_exceed_MW_max', 0.01))
            ).sum()
        robust['failure_rates'][sid] = float(failures / len(strat_df)) if len(strat_df) > 0 else 0.0
    
    # Add alias keys for backward compatibility
    if 'best_strategy_by_minimax_regret' in robust:
        robust['best_by_minimax_regret'] = robust['best_strategy_by_minimax_regret']
    if 'best_strategy_by_max_satisficing' in robust:
        robust['best_by_max_satisficing'] = robust['best_strategy_by_max_satisficing']
    
    # Add strategy_labels mapping
    robust['strategy_labels'] = {
        strategy['strategy_id']: strategy.get('label', strategy['strategy_id'])
        for strategy in strategies
    }
    
    # Add satisficing thresholds used
    robust['satisficing_thresholds'] = {
        'shed_MWh_max': metrics_config.get('shed_MWh_max', 0.0),
        'max_exceed_MW_max': metrics_config.get('max_exceed_MW_max', 0.01),
        'shed_fraction_max': metrics_config.get('shed_fraction_max', None),
        'p95_exceed_MW_max': metrics_config.get('p95_exceed_MW_max', None)
    }
    
    # Add provenance metadata
    if experiment is not None:
        robust['experiment_name'] = experiment.get('name', 'unknown')
        robust['epoch_tag'] = epoch_tag
        robust['n_futures'] = n_futures
        robust['seed'] = experiment.get('seed', None)
        robust['dt_h'] = experiment.get('dt_h', 1.0)
        if config_path is not None:
            robust['config_path'] = str(config_path)
        robust['timestamp'] = pd.Timestamp.now().isoformat()
    
    return robust


def generate_robust_summary_table(summary_df: pd.DataFrame, strategies: List[Dict],
                                   robust_summary: Dict[str, Any], metrics_config: Dict) -> pd.DataFrame:
    """Generate robust summary table CSV with one row per strategy."""
    # Filter out error rows
    if 'total_cost_nzd' in summary_df.columns:
        valid_df = summary_df.dropna(subset=['total_cost_nzd']).copy()
    else:
        valid_df = summary_df.copy()
    
    if len(valid_df) == 0:
        return pd.DataFrame()
    
    # Get strategy metadata
    strategy_metadata, benchmark_strategy_id = get_strategy_metadata(strategies, metrics_config)
    benchmark_set = set(robust_summary.get('benchmark_strategy_ids', []))
    
    rows = []
    for strategy in strategies:
        sid = strategy['strategy_id']
        strat_df = valid_df[valid_df['strategy_id'] == sid]
        if len(strat_df) == 0:
            continue
        
        is_benchmark = sid in benchmark_set
        metadata = strategy_metadata.get(sid, {})
        
        # Get metrics from robust_summary
        failure_rate = robust_summary.get('failure_rates', {}).get(sid, 0.0)
        satisficing_rate = robust_summary.get('satisficing_by_strategy', {}).get(sid, {}).get('pass_rate', 0.0)
        
        # Cost quantiles
        cost_quantiles = robust_summary.get('quantiles_by_strategy', {}).get(sid, {}).get('total_cost_nzd', {})
        p50_total_cost_nzd = cost_quantiles.get('p50', np.nan)
        p95_total_cost_nzd = cost_quantiles.get('p95', np.nan)
        
        # Regret metrics (policy regret for ranking)
        regret_metrics = robust_summary.get('regret_by_strategy', {}).get(sid, {})
        max_regret_policy_nzd = regret_metrics.get('max_regret', np.nan)
        p95_regret_policy_nzd = regret_metrics.get('p95_regret', np.nan)
        
        # Benchmark regret (if available)
        regret_benchmark_metrics = robust_summary.get('regret_vs_benchmark_by_strategy', {}).get(sid, {})
        max_regret_benchmark_nzd = regret_benchmark_metrics.get('max_regret', np.nan)
        p95_regret_benchmark_nzd = regret_benchmark_metrics.get('p95_regret', np.nan)
        
        # Shed fraction quantiles (from per-run data)
        if 'shed_fraction' in strat_df.columns:
            p50_shed_fraction = float(np.quantile(strat_df['shed_fraction'], 0.50))
            p95_shed_fraction = float(np.quantile(strat_df['shed_fraction'], 0.95))
        else:
            p50_shed_fraction = np.nan
            p95_shed_fraction = np.nan
        
        # p95_exceed_MW (aggregate per-run p95_exceed_MW)
        if 'p95_exceed_MW' in strat_df.columns:
            median_p95_exceed_MW = float(np.median(strat_df['p95_exceed_MW']))
            p95_p95_exceed_MW = float(np.quantile(strat_df['p95_exceed_MW'], 0.95))
        else:
            median_p95_exceed_MW = np.nan
            p95_p95_exceed_MW = np.nan
        
        # hours_exceed metrics
        if 'hours_exceed' in strat_df.columns:
            median_hours_exceed = float(np.median(strat_df['hours_exceed']))
            p95_hours_exceed = float(np.quantile(strat_df['hours_exceed'], 0.95))
        else:
            median_hours_exceed = np.nan
            p95_hours_exceed = np.nan
        
        rows.append({
            'strategy_id': sid,
            'strategy_label': metadata.get('strategy_label', strategy.get('label', sid)),
            'is_benchmark': bool(is_benchmark),
            'benchmark_label': metadata.get('benchmark_label', None),
            'failure_rate': failure_rate,
            'satisficing_rate': satisficing_rate,
            'p50_total_cost_nzd': p50_total_cost_nzd,
            'p95_total_cost_nzd': p95_total_cost_nzd,
            'max_regret_policy_nzd': max_regret_policy_nzd,
            'p95_regret_policy_nzd': p95_regret_policy_nzd,
            'max_regret_benchmark_nzd': max_regret_benchmark_nzd,
            'p95_regret_benchmark_nzd': p95_regret_benchmark_nzd,
            'p50_shed_fraction': p50_shed_fraction,
            'p95_shed_fraction': p95_shed_fraction,
            'median_p95_exceed_MW': median_p95_exceed_MW,
            'p95_p95_exceed_MW': p95_p95_exceed_MW,
            'median_hours_exceed': median_hours_exceed,
            'p95_hours_exceed': p95_hours_exceed
        })
    
    return pd.DataFrame(rows)


def generate_rdm_figures(summary_df: pd.DataFrame, strategies: List[Dict],
                         figures_dir: Path, metrics_config: Dict = None) -> None:
    """Generate RDM figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping figures")
        return
    
    # Filter out error rows (robust filtering)
    if 'total_cost_nzd' in summary_df.columns:
        valid_df = summary_df.dropna(subset=['total_cost_nzd']).copy()
    else:
        valid_df = summary_df.copy()
    
    if len(valid_df) == 0:
        print("[WARN] No valid results for figures (all rows have NaN total_cost_nzd)")
        return
    
    # Get strategy metadata
    if metrics_config is None:
        metrics_config = {}
    strategy_metadata, benchmark_strategy_id = get_strategy_metadata(strategies, metrics_config)
    
    # Figure 1: Regret ECDF (step plot, rescaled to billions)
    if 'regret' in valid_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_auto = False
        for strategy in strategies:
            sid = strategy['strategy_id']
            strat_df = valid_df[valid_df['strategy_id'] == sid]
            if len(strat_df) == 0:
                continue
            
            # Filter NaNs from regret
            regret_vals = strat_df['regret'].dropna()
            if len(regret_vals) == 0:
                continue
            
            sorted_regret = np.sort(regret_vals)
            y = np.arange(1, len(sorted_regret) + 1) / len(sorted_regret)
            
            # Rescale to billions NZD
            regret_billions = sorted_regret / 1e9
            
            # Use step plot
            label = strategy.get('label', sid)
            is_benchmark = strategy_metadata.get(sid, {}).get('is_benchmark', False)
            if is_benchmark:
                has_auto = True
                label += ' (benchmark; ex-post optimum)'
            elif sid.startswith('S0_NONE'):
                label += ' (baseline)'
            
            ax.step(regret_billions, y, label=label, linewidth=2, where='post')
        
        ax.set_xlabel('Regret (NZD billions)')
        ax.set_ylabel('Cumulative Probability')
        title = 'Regret ECDF by Strategy'
        if has_auto:
            title += '\n(AUTO = ex-post optimum benchmark)'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'regret_cdf.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {figures_dir / 'regret_cdf.png'}")
        
        # Figure 1b: Regret ECDF with log-x scaling
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in strategies:
            sid = strategy['strategy_id']
            strat_df = valid_df[valid_df['strategy_id'] == sid]
            if len(strat_df) == 0:
                continue
            
            regret_vals = strat_df['regret'].dropna()
            if len(regret_vals) == 0:
                continue
            
            sorted_regret = np.sort(regret_vals)
            y = np.arange(1, len(sorted_regret) + 1) / len(sorted_regret)
            
            # Add small epsilon to avoid log(0)
            regret_log = sorted_regret + 1e-6
            
            label = strategy.get('label', sid)
            is_benchmark = strategy_metadata.get(sid, {}).get('is_benchmark', False)
            if is_benchmark:
                label += ' (benchmark; ex-post optimum)'
            elif sid.startswith('S0_NONE'):
                label += ' (baseline)'
            
            ax.step(regret_log, y, label=label, linewidth=2, where='post')
        
        ax.set_xscale('log')
        ax.set_xlabel('Regret (NZD, log scale)')
        ax.set_ylabel('Cumulative Probability')
        title = 'Regret ECDF by Strategy (Log Scale)'
        if has_auto:
            title += '\n(AUTO = ex-post optimum benchmark)'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'regret_cdf_logx.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {figures_dir / 'regret_cdf_logx.png'}")
    
    # Figure 2: Total cost boxplot (rescaled to billions)
    # Order strategies consistently: S0_NONE, OPT1, OPT2, OPT3, OPT4, AUTO
    strategy_order = []
    for prefix in ['S0_NONE', 'S1_OPT1', 'S2_OPT2', 'S3_OPT3', 'S4_OPT4']:
        matching = [s for s in strategies if s['strategy_id'].startswith(prefix)]
        strategy_order.extend(matching)
    # Add any remaining strategies (including AUTO)
    remaining = [s for s in strategies if s not in strategy_order]
    strategy_order.extend(remaining)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    labels = []
    for s in strategy_order:
        sid = s['strategy_id']
        strat_df = valid_df[valid_df['strategy_id'] == sid]
        if len(strat_df) == 0:
            continue
        cost_vals = strat_df['total_cost_nzd'].values
        # Rescale to billions NZD
        data.append(cost_vals / 1e9)
        label = s.get('label', sid)
        is_benchmark = strategy_metadata.get(sid, {}).get('is_benchmark', False)
        if is_benchmark:
            label += ' (benchmark; ex-post optimum)'
        labels.append(label)
    
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.set_ylabel('Total cost (NZD, billions)')
        ax.set_title('Total Cost Distribution by Strategy')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(figures_dir / 'total_cost_boxplot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {figures_dir / 'total_cost_boxplot.png'}")
    
    # Figure 2b: Total cost boxplot excluding S0_NONE (to show discrimination among upgrades)
    fig, ax = plt.subplots(figsize=(10, 6))
    data_excl_none = []
    labels_excl_none = []
    for s in strategy_order:
        sid = s['strategy_id']
        if sid.startswith('S0_NONE'):
            continue
        strat_df = valid_df[valid_df['strategy_id'] == sid]
        if len(strat_df) == 0:
            continue
        cost_vals = strat_df['total_cost_nzd'].values
        data_excl_none.append(cost_vals / 1e9)
        label = s.get('label', sid)
        is_benchmark = strategy_metadata.get(sid, {}).get('is_benchmark', False)
        if is_benchmark:
            label += ' (benchmark; ex-post optimum)'
        labels_excl_none.append(label)
    
    if data_excl_none:
        bp = ax.boxplot(data_excl_none, tick_labels=labels_excl_none, patch_artist=True)
        ax.set_ylabel('Total cost (NZD, billions)')
        ax.set_title('Total Cost Distribution by Strategy (excluding S0_NONE)')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(figures_dir / 'total_cost_boxplot_excl_none.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {figures_dir / 'total_cost_boxplot_excl_none.png'}")
    
    # Figure 3: Shed histogram (annual_shed_MWh)
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy in strategies:
        sid = strategy['strategy_id']
        strat_df = valid_df[valid_df['strategy_id'] == sid]
        if len(strat_df) == 0:
            continue
        
        label = strategy.get('label', sid)
        is_benchmark = strategy_metadata.get(sid, {}).get('is_benchmark', False)
        if is_benchmark:
            label += ' (benchmark; ex-post optimum)'
        ax.hist(strat_df['annual_shed_MWh'], bins=30, alpha=0.6, label=label)
    
    ax.set_xlabel('Annual Shed (MWh)')
    ax.set_ylabel('Frequency')
    ax.set_title('Annual Shed Distribution by Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'shed_hist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {figures_dir / 'shed_hist.png'}")
    
    # Figure 3b: Shed fraction histogram (if computed)
    if 'shed_fraction' in valid_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in strategies:
            sid = strategy['strategy_id']
            strat_df = valid_df[valid_df['strategy_id'] == sid]
            if len(strat_df) == 0:
                continue
            
            label = strategy.get('label', sid)
            is_benchmark = strategy_metadata.get(sid, {}).get('is_benchmark', False)
            if is_benchmark:
                label += ' (benchmark; ex-post optimum)'
            ax.hist(strat_df['shed_fraction'], bins=30, alpha=0.6, label=label)
        
        ax.set_xlabel('Shed Fraction (annual_shed_MWh / annual_incremental_MWh)')
        ax.set_ylabel('Frequency')
        ax.set_title('Shed Fraction Distribution by Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'shed_fraction_hist.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {figures_dir / 'shed_fraction_hist.png'}")
    
    # Figure 4: Failure rate bar chart
    if 'satisficing_pass' in valid_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        strategy_ids = []
        failure_rates = []
        labels = []
        is_auto_flags = []
        
        for strategy in strategies:
            sid = strategy['strategy_id']
            strat_df = valid_df[valid_df['strategy_id'] == sid]
            if len(strat_df) == 0:
                continue
            
            is_auto = strategy.get('mode') == 'auto_select'
            failure_rate = (~strat_df['satisficing_pass']).mean() if 'satisficing_pass' in strat_df.columns else 0.0
            
            strategy_ids.append(sid)
            failure_rates.append(failure_rate)
            is_auto_flags.append(is_auto)
            
            label = strategy.get('label', sid)
            if is_auto:
                label += ' (benchmark)'
            labels.append(label)
        
        # Create bar chart
        bars = ax.bar(range(len(strategy_ids)), failure_rates, alpha=0.7)
        
        # Color AUTO strategies differently
        for i, is_auto in enumerate(is_auto_flags):
            if is_auto:
                bars[i].set_color('gray')
                bars[i].set_alpha(0.5)
        
        ax.set_xlabel('Failure Rate')
        ax.set_ylabel('Strategy')
        ax.set_title('Failure Rate by Strategy')
        ax.set_yticks(range(len(strategy_ids)))
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(figures_dir / 'failure_rate_bar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {figures_dir / 'failure_rate_bar.png'}")


def write_thesis_figures_md(run_dir: Path, figures_dir: Path, experiment: Dict,
                           epoch_tag: str, robust_summary: Dict,
                           metrics_config: Dict) -> None:
    """Generate THESIS_FIGURES.md with figure index and descriptions."""
    exp_name = experiment.get('name', 'unknown')
    
    # Extract benchmark info
    benchmark_strategy_id = robust_summary.get('benchmark_strategy_id', None)
    has_benchmark = benchmark_strategy_id is not None
    
    # Extract regret references
    regret_ref_ranking = robust_summary.get('regret_reference_for_ranking', 'best_policy')
    regret_ref_reporting = robust_summary.get('regret_reference_for_reporting', 'best_policy')
    
    # Extract satisficing thresholds
    thresholds = robust_summary.get('satisficing_thresholds', {})
    annual_shed_max = thresholds.get('annual_shed_MWh_max', metrics_config.get('annual_shed_MWh_max', 0.0))
    shed_fraction_max = thresholds.get('shed_fraction_max', metrics_config.get('shed_fraction_max', 0.0))
    max_exceed_max = thresholds.get('max_exceed_MW_max', metrics_config.get('max_exceed_MW_max', 0.1))
    hours_exceed_max = thresholds.get('hours_exceed_max', metrics_config.get('hours_exceed_max', None))
    p95_exceed_max = thresholds.get('p95_exceed_MW_max', metrics_config.get('p95_exceed_MW_max', None))
    
    # Build threshold list
    threshold_items = [
        f"annual_shed_MWh_max = {annual_shed_max:.3f} MWh",
        f"shed_fraction_max = {shed_fraction_max:.4f}",
        f"max_exceed_MW_max = {max_exceed_max:.2f} MW"
    ]
    if hours_exceed_max is not None:
        threshold_items.append(f"hours_exceed_max = {hours_exceed_max}")
    if p95_exceed_max is not None:
        threshold_items.append(f"p95_exceed_MW_max = {p95_exceed_max:.2f} MW")
    
    # Check which figures exist
    figure_files = {
        'regret_quantiles': figures_dir / 'regret_quantiles.png',
        'regret_ecdf_policy': figures_dir / 'regret_ecdf_policy.png',
        'regret_ecdf_vs_benchmark': figures_dir / 'regret_ecdf_vs_benchmark.png',
        'failure_satisficing_barh': figures_dir / 'failure_satisficing_barh.png',
        'failure_rate_bar': figures_dir / 'failure_rate_bar.png',
        'exceed_quantiles': figures_dir / 'exceed_quantiles.png',
        'shed_fraction_boxplot': figures_dir / 'shed_fraction_boxplot.png',
        'shed_hist': figures_dir / 'shed_hist.png'
    }
    
    figure_exists = {name: path.exists() for name, path in figure_files.items()}
    
    # Build markdown content
    lines = [
        f"# Thesis-ready figures for {exp_name} ({epoch_tag})",
        "",
        "## Preamble",
        ""
    ]
    
    # Benchmark and regret info
    if has_benchmark:
        lines.append(f"- **Benchmark strategy**: {benchmark_strategy_id} (ex-post optimum; not deployable ex-ante)")
    else:
        lines.append("- **Benchmark strategy**: None (no ex-post optimum available)")
    
    lines.append(f"- **Regret baseline for ranking**: {regret_ref_ranking}")
    if has_benchmark:
        lines.append(f"- **Regret baseline for reporting**: {regret_ref_reporting}")
    
    lines.append("- **Satisficing thresholds**:")
    for item in threshold_items:
        lines.append(f"  - {item}")
    
    lines.extend([
        "",
        "## Figures to cite (required)",
        ""
    ])
    
    # A) regret_quantiles.png
    if figure_exists['regret_quantiles']:
        lines.append("**A) [regret_quantiles.png](figures/regret_quantiles.png)**")
        if has_benchmark:
            lines.append("Robust tail-regret comparison across policy strategies, with benchmark framing showing the value-of-information gap relative to the ex-post optimum.")
        else:
            lines.append("Robust tail-regret comparison across policy strategies, highlighting robustness across futures.")
    else:
        lines.append("**A) regret_quantiles.png** (missing)")
        lines.append("Robust tail-regret comparison across policy strategies (benchmark framing if benchmark exists).")
    
    lines.append("")
    
    # B) regret_ecdf_policy.png
    if figure_exists['regret_ecdf_policy']:
        lines.append("**B) [regret_ecdf_policy.png](figures/regret_ecdf_policy.png)**")
        lines.append("Distribution of policy regret vs best-policy baseline, highlighting robustness across futures.")
    else:
        lines.append("**B) regret_ecdf_policy.png** (missing)")
        lines.append("Distribution of policy regret vs best-policy baseline, highlighting robustness across futures.")
    
    lines.append("")
    
    # C) regret_ecdf_vs_benchmark.png (only if benchmark exists)
    if has_benchmark:
        if figure_exists['regret_ecdf_vs_benchmark']:
            lines.append("**C) [regret_ecdf_vs_benchmark.png](figures/regret_ecdf_vs_benchmark.png)**")
            lines.append("Policy regret relative to AUTO ex-post benchmark to show value-of-information gap and benchmark framing.")
        else:
            lines.append("**C) regret_ecdf_vs_benchmark.png** (missing)")
            lines.append("Policy regret relative to AUTO ex-post benchmark to show value-of-information gap / benchmark framing.")
        lines.append("")
    
    # D) failure_satisficing_barh.png or failure_rate_bar.png
    if figure_exists['failure_satisficing_barh']:
        lines.append("**D) [failure_satisficing_barh.png](figures/failure_satisficing_barh.png)**")
        lines.append("Failure/satisficing rates under configured thresholds (reliability framing).")
    elif figure_exists['failure_rate_bar']:
        lines.append("**D) [failure_rate_bar.png](figures/failure_rate_bar.png)**")
        lines.append("Failure/satisficing rates under configured thresholds (reliability framing).")
    else:
        lines.append("**D) failure_satisficing_barh.png / failure_rate_bar.png** (missing)")
        lines.append("Failure/satisficing rates under configured thresholds (reliability framing).")
    
    lines.append("")
    
    # E) exceed_quantiles.png
    if figure_exists['exceed_quantiles']:
        lines.append("**E) [exceed_quantiles.png](figures/exceed_quantiles.png)**")
        lines.append("Exceedance risk summary (p50/p95/p99/max exceed_MW) linking to headroom constraint and satisficing.")
    else:
        lines.append("**E) exceed_quantiles.png** (missing)")
        lines.append("Exceedance risk summary (p50/p95/p99/max exceed_MW) linking to headroom constraint and satisficing.")
    
    lines.append("")
    
    # F) shed_fraction_boxplot.png or shed_hist.png
    if figure_exists['shed_fraction_boxplot']:
        lines.append("**F) [shed_fraction_boxplot.png](figures/shed_fraction_boxplot.png)**")
        lines.append("Curtailment/shedding severity distribution across strategies (operational feasibility).")
    elif figure_exists['shed_hist']:
        lines.append("**F) [shed_hist.png](figures/shed_hist.png)**")
        lines.append("Curtailment/shedding severity distribution across strategies (operational feasibility).")
    else:
        lines.append("**F) shed_fraction_boxplot.png / shed_hist.png** (missing)")
        lines.append("Curtailment/shedding severity distribution across strategies (operational feasibility).")
    
    lines.extend([
        "",
        "## Notes for captioning",
        "",
        "- **Benchmark definition**: AUTO = ex-post optimum (not deployable ex-ante)",
        f"- **Regret baselines**: Policy regret computed vs best-policy baseline for ranking; " +
        (f"benchmark regret computed vs {benchmark_strategy_id} ex-post optimum for reporting." if has_benchmark else "no benchmark available."),
        f"- **Satisficing constraints**: Strategy passes if annual_shed_MWh <= {annual_shed_max:.3f} MWh, " +
        f"shed_fraction <= {shed_fraction_max:.4f}, max_exceed_MW <= {max_exceed_max:.2f} MW" +
        (f", hours_exceed <= {hours_exceed_max}" if hours_exceed_max is not None else "") +
        (f", p95_exceed_MW <= {p95_exceed_max:.2f} MW" if p95_exceed_max is not None else "") + "."
    ])
    
    # Write file
    md_path = run_dir / 'THESIS_FIGURES.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[OK] Saved thesis figures index to {md_path}")


def write_rdm_readme(run_dir: Path, exp_name: str, epoch_tag: str,
                    n_futures: int, n_strategies: int) -> None:
    """Write README.md explaining RDM outputs."""
    readme_content = f"""# RDM Experiment: {exp_name}

**Epoch**: {epoch_tag}  
**Futures**: {n_futures}  
**Strategies**: {n_strategies}

## Files

- `config_used.toml`: Copy of experiment configuration
- `run_metadata.json`: Run metadata (config paths, timestamps)
- `futures.csv`: Sampled uncertainty futures (one row per future)
- `rdm_summary.csv`: Results for all future-strategy combinations
- `robust_summary.json`: Aggregated statistics (quantiles, regret, satisficing)
- `run_ledger.jsonl`: Minimal provenance per run
- `figures/`: Generated plots (regret CDF, cost boxplot, shed histogram)

## Key Metrics

- **Regret**: Difference between strategy cost and minimum cost in each future
- **Satisficing**: Whether strategy meets constraints (shed_MWh <= threshold, max_exceed_MW <= threshold)
- **Quantiles**: p05/p50/p95 of total_cost and annual_shed_MWh by strategy

## PoC Assumptions

- Upgrades available instantly upon selection (lead_time_months retained for reporting only)
- Annualised costs computed using CRF from assumptions in upgrade menu TOML
"""
    
    with open(run_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    print(f"  Saved {run_dir / 'README.md'}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='RDM Experiment Runner: Evaluate grid upgrade strategies across uncertain futures'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment TOML config')
    parser.add_argument('--bundle', type=str, required=True,
                       help='Bundle name for output paths (replaces <BUNDLE> in config)')
    parser.add_argument('--clean', action='store_true',
                       help='Delete output directory before running')
    parser.add_argument('--n-futures', type=int, default=None,
                       help='Override n_futures from config')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override seed from config')
    parser.add_argument('--save-timeseries', type=str, choices=['representative', 'all', 'none'],
                       default='none', help='Save timeseries outputs (default: none)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Create run directory and metadata, then exit without loading CSVs')
    
    args = parser.parse_args()
    
    try:
        run_rdm_experiment(
            config_path=args.config,
            bundle_name=args.bundle,
            n_futures_override=args.n_futures,
            seed_override=args.seed,
            clean=args.clean,
            save_timeseries=args.save_timeseries,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

