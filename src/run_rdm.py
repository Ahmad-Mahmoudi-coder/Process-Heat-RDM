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
                raise ValueError(
                    f"Strategy {strategy['strategy_id']}: Unknown upgrade name '{force_name}'. "
                    f"Available: {upgrade_names}"
                )
    
    uncertainties = config.get('uncertainties', [])
    metrics_config = config.get('metrics', {})
    
    # Set default metrics thresholds (numerically robust)
    if 'shed_MWh_max' not in metrics_config:
        metrics_config['shed_MWh_max'] = 0.1
    if 'max_exceed_MW_max' not in metrics_config:
        metrics_config['max_exceed_MW_max'] = 0.01
    
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
    
    # Use config-driven timestamp selection
    ts_col = paths.get('timestamp_col', 'timestamp_utc')
    if ts_col not in merged.columns:
        # Try incremental timestamp column
        if incremental_ts_col in merged.columns:
            ts_col = incremental_ts_col
        else:
            raise KeyError(f"timestamp_col '{paths.get('timestamp_col', 'timestamp_utc')}' not found in merged columns: {merged.columns.tolist()}")
    
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
                annual_shed_cost_nzd = selected['annual_shed_MWh'] * voll
                menu_max_capacity_MW = max(opt['capacity_MW'] for opt in upgrades_to_use)
                menu_max_selected = (selected['capacity_MW'] == menu_max_capacity_MW)
                
                # Satisficing check (use defaults from metrics_config)
                shed_MWh_max = metrics_config.get('shed_MWh_max', 0.1)
                max_exceed_MW_max = metrics_config.get('max_exceed_MW_max', 0.01)
                satisficing_pass = (
                    selected['annual_shed_MWh'] <= shed_MWh_max and
                    max_exceed_MW <= max_exceed_MW_max
                )
                
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
                    'annual_shed_MWh': selected['annual_shed_MWh'],
                    'annual_shed_cost_nzd': annual_shed_cost_nzd,
                    'total_cost_nzd': selected['total_cost_nzd'],
                    'max_exceed_MW': max_exceed_MW,
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
    
    # Compute regret (robust to NaNs)
    if metrics_config.get('compute_regret', True):
        regret_baseline = metrics_config.get('regret_baseline', 'per_future_min')
        
        if regret_baseline == 'per_future_min':
            # For each future, compute min cost across strategies (ignoring NaNs)
            summary_df['min_cost_in_future'] = summary_df.groupby('future_id')['total_cost_nzd'].transform(lambda x: x.min(skipna=True))
            summary_df['regret'] = summary_df['total_cost_nzd'] - summary_df['min_cost_in_future']
            # If all strategies are NaN for a future, regret will be NaN (which is correct)
            summary_df = summary_df.drop(columns=['min_cost_in_future'])
        # TODO: Add fixed_strategy_id baseline if needed
    
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
    robust_summary = compute_robust_summary(summary_df, strategies, metrics_config)
    
    # Save robust summary JSON
    robust_path = run_dir / 'robust_summary.json'
    with open(robust_path, 'w') as f:
        json.dump(robust_summary, f, indent=2)
    print(f"[OK] Saved robust summary to {robust_path}")
    
    # Generate figures
    print(f"[RDM] Generating figures...")
    generate_rdm_figures(summary_df, strategies, figures_dir)
    
    # Save README
    write_rdm_readme(run_dir, exp_name, epoch_tag, len(futures_df), len(strategies))
    
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
    
    # Best strategies
    if 'best_strategy_by_minimax_regret' in robust_summary:
        print(f"  Best by minimax regret: {robust_summary['best_strategy_by_minimax_regret']}")
    if 'best_strategy_by_max_satisficing' in robust_summary:
        print(f"  Best by max satisficing: {robust_summary['best_strategy_by_max_satisficing']}")
    
    # Failure rates
    if 'failure_rates' in robust_summary:
        print(f"  Failure rates by strategy:")
        for sid, rate in robust_summary['failure_rates'].items():
            strategy_label = next((s.get('label', sid) for s in strategies if s['strategy_id'] == sid), sid)
            print(f"    {strategy_label}: {rate:.1%}")
    
    print(f"  Output directory: {run_dir}")


def compute_robust_summary(summary_df: pd.DataFrame, strategies: List[Dict],
                          metrics_config: Dict) -> Dict[str, Any]:
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
    
    # Regret metrics (robust to NaNs)
    if 'regret' in valid_df.columns:
        robust['regret_by_strategy'] = {}
        for strategy in strategies:
            sid = strategy['strategy_id']
            strat_df = valid_df[valid_df['strategy_id'] == sid]
            if len(strat_df) == 0:
                continue
            
            # Filter NaNs from regret
            regret_vals = strat_df['regret'].dropna()
            if len(regret_vals) == 0:
                continue
            
            robust['regret_by_strategy'][sid] = {
                'max_regret': float(regret_vals.max()),
                'p95_regret': float(np.quantile(regret_vals, 0.95))
            }
        
        # Best by minimax regret
        max_regrets = {
            sid: robust['regret_by_strategy'][sid]['max_regret']
            for sid in robust['regret_by_strategy']
        }
        if max_regrets:
            robust['best_strategy_by_minimax_regret'] = min(max_regrets, key=max_regrets.get)
    
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
        
        # Best by max satisficing, then min p95 cost
        if robust['satisficing_by_strategy']:
            # Sort by pass rate (desc), then p95 cost (asc)
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
    
    return robust


def generate_rdm_figures(summary_df: pd.DataFrame, strategies: List[Dict],
                         figures_dir: Path) -> None:
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
    
    # Figure 1: Regret CDF
    if 'regret' in valid_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
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
            ax.plot(sorted_regret, y, label=strategy.get('label', sid), linewidth=2)
        
        ax.set_xlabel('Regret (NZD)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Regret ECDF by Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'regret_cdf.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {figures_dir / 'regret_cdf.png'}")
    
    # Figure 2: Total cost boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [valid_df[valid_df['strategy_id'] == s['strategy_id']]['total_cost_nzd'].values
            for s in strategies
            if len(valid_df[valid_df['strategy_id'] == s['strategy_id']]) > 0]
    labels = [s.get('label', s['strategy_id']) for s in strategies
              if len(valid_df[valid_df['strategy_id'] == s['strategy_id']]) > 0]
    
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.set_ylabel('Total Cost (NZD)')
        ax.set_title('Total Cost Distribution by Strategy')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(figures_dir / 'total_cost_boxplot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {figures_dir / 'total_cost_boxplot.png'}")
    
    # Figure 3: Shed histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy in strategies:
        sid = strategy['strategy_id']
        strat_df = valid_df[valid_df['strategy_id'] == sid]
        if len(strat_df) == 0:
            continue
        
        ax.hist(strat_df['annual_shed_MWh'], bins=30, alpha=0.6, label=strategy.get('label', sid))
    
    ax.set_xlabel('Annual Shed (MWh)')
    ax.set_ylabel('Frequency')
    ax.set_title('Annual Shed Distribution by Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'shed_hist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {figures_dir / 'shed_hist.png'}")


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

