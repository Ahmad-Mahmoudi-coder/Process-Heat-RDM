"""
Regional Electricity PoC Module

Minimal, deterministic electricity "regional signalling" PoC that:
- reads incremental electricity demand (from site dispatch export)
- applies a simple headroom constraint
- produces headroom + tariff signals (SignalsPack-like)
- writes outputs into the same run folder (no external downloads)
"""

from __future__ import annotations

import sys
from pathlib import Path as PathlibPath

ROOT = PathlibPath(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import pandas as pd
import numpy as np
from typing import Dict, Optional

# TOML loading (compatible with Python 3.11+ tomllib or tomli fallback)
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Need tomllib (Python 3.11+) or tomli package")

from src.path_utils import repo_root, resolve_path
from src.time_utils import parse_any_timestamp, to_iso_z


def load_gxp_config(config_path: Optional[PathlibPath], epoch_tag: str) -> Dict:
    """
    Load GXP PoC config from TOML file.
    
    Args:
        config_path: Path to GXP config TOML (optional)
        epoch_tag: Epoch tag (e.g., "2035_EB")
        
    Returns:
        Dictionary with keys: gxp_capacity_MW, baseline_import_MW, 
        tariff_base_nzd_per_MWh, scarcity_adder_nzd_per_MWh
    """
    defaults = {
        'gxp_capacity_MW': 120.0,
        'baseline_import_MW': 70.0,
        'tariff_base_nzd_per_MWh': 120.0,
        'scarcity_adder_nzd_per_MWh': 200.0
    }
    
    if config_path is None or not config_path.exists():
        print(f"[WARN] GXP config TOML not found: {config_path}, using defaults")
        return defaults
    
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
        
        # Look for epoch-specific section (e.g., [2035_EB] or [2035_BB])
        epoch_section = config.get(epoch_tag, {})
        if not epoch_section:
            # Try base year (e.g., [2035] if epoch_tag is "2035_EB")
            base_year = epoch_tag.split('_')[0] if '_' in epoch_tag else epoch_tag
            epoch_section = config.get(base_year, {})
        
        # Merge defaults with config values
        result = defaults.copy()
        for key in defaults.keys():
            if key in epoch_section:
                result[key] = float(epoch_section[key])
        
        print(f"[OK] Loaded GXP config for {epoch_tag} from {config_path}")
        return result
    except Exception as e:
        print(f"[WARN] Failed to load GXP config: {e}, using defaults")
        return defaults


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
    
    # Normalize timestamp column
    ts_col = None
    for col in ['timestamp_utc', 'timestamp']:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col is None:
        raise ValueError(f"Incremental CSV must have 'timestamp' or 'timestamp_utc' column. Found: {df.columns.tolist()}")
    
    # Normalize incremental column
    inc_col = None
    for col in ['incremental_electricity_MW', 'incremental_MW']:
        if col in df.columns:
            inc_col = col
            break
    
    if inc_col is None:
        raise ValueError(f"Incremental CSV must have 'incremental_electricity_MW' or 'incremental_MW' column. Found: {df.columns.tolist()}")
    
    # Parse timestamps
    df['timestamp_utc'] = parse_any_timestamp(df[ts_col])
    if df['timestamp_utc'].dt.tz is None:
        df['timestamp_utc'] = df['timestamp_utc'].dt.tz_localize('UTC')
    else:
        df['timestamp_utc'] = df['timestamp_utc'].dt.tz_convert('UTC')
    
    # Rename incremental column to standard name
    df['incremental_electricity_MW'] = df[inc_col]
    
    # Sort by timestamp
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Validate timestamps
    if not df['timestamp_utc'].is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonic increasing")
    
    if df['timestamp_utc'].duplicated().any():
        raise ValueError("Duplicate timestamps found")
    
    # Validate incremental values
    if (df['incremental_electricity_MW'] < 0).any():
        print(f"[WARN] Negative incremental electricity values found, setting to 0")
        df['incremental_electricity_MW'] = df['incremental_electricity_MW'].clip(lower=0.0)
    
    return df[['timestamp_utc', 'incremental_electricity_MW']]


def compute_regional_signals(incremental_df: pd.DataFrame, gxp_config: Dict) -> pd.DataFrame:
    """
    Compute regional electricity signals (headroom, overload, shed, tariff).
    
    Logic:
    1) headroom_MW = max(0, gxp_capacity_MW - baseline_import_MW) (constant)
    2) overload_MW = max(0, incremental_MW - headroom_MW)
    3) shed_MW = overload_MW
    4) tariff_nzd_per_MWh = tariff_base + scarcity_adder * (overload_MW > 0 ? 1 : 0)
    
    Args:
        incremental_df: DataFrame with timestamp_utc, incremental_electricity_MW
        gxp_config: Dictionary with GXP configuration
        
    Returns:
        DataFrame with columns: timestamp_utc, headroom_MW, overload_MW, 
        shed_MW, tariff_nzd_per_MWh
    """
    result = incremental_df.copy()
    
    # Extract config values
    gxp_capacity_MW = gxp_config['gxp_capacity_MW']
    baseline_import_MW = gxp_config['baseline_import_MW']
    tariff_base = gxp_config['tariff_base_nzd_per_MWh']
    scarcity_adder = gxp_config['scarcity_adder_nzd_per_MWh']
    
    # 1) Compute headroom (constant across hours)
    headroom_MW = max(0.0, gxp_capacity_MW - baseline_import_MW)
    result['headroom_MW'] = headroom_MW
    
    # 2) Compute overload
    result['overload_MW'] = np.maximum(0.0, result['incremental_electricity_MW'] - headroom_MW)
    
    # 3) Compute shed (equal to overload in PoC)
    result['shed_MW'] = result['overload_MW']
    
    # 4) Compute tariff (binary scarcity adder)
    result['tariff_nzd_per_MWh'] = tariff_base + scarcity_adder * (result['overload_MW'] > 0).astype(float)
    
    return result


def compute_summary(signals_df: pd.DataFrame, gxp_config: Dict, epoch_tag: str) -> Dict:
    """
    Compute annual summary statistics.
    
    Args:
        signals_df: DataFrame with computed signals
        gxp_config: Dictionary with GXP configuration
        epoch_tag: Epoch tag
        
    Returns:
        Dictionary with summary metrics
    """
    # Estimate dt_h from timestamps
    if len(signals_df) > 1:
        dt_h = (signals_df['timestamp_utc'].iloc[1] - signals_df['timestamp_utc'].iloc[0]).total_seconds() / 3600.0
    else:
        dt_h = 1.0
    
    # Compute annual totals
    annual_incremental_MWh = (signals_df['incremental_electricity_MW'] * dt_h).sum()
    annual_shed_MWh = (signals_df['shed_MW'] * dt_h).sum()
    max_overload_MW = signals_df['overload_MW'].max()
    
    summary = {
        'epoch_tag': epoch_tag,
        'gxp_capacity_MW': float(gxp_config['gxp_capacity_MW']),
        'baseline_import_MW': float(gxp_config['baseline_import_MW']),
        'annual_incremental_MWh': float(annual_incremental_MWh),
        'annual_shed_MWh': float(annual_shed_MWh),
        'max_overload_MW': float(max_overload_MW),
        'notes': 'PoC deterministic regional signalling (no external data)'
    }
    
    return summary


def write_outputs(signals_df: pd.DataFrame, summary: Dict, output_dir: PathlibPath, epoch_tag: str) -> None:
    """
    Write outputs to electricity_poc directory.
    
    Outputs:
    - signals/headroom_MW_<epoch_tag>.csv
    - signals/tariff_nzd_per_MWh_<epoch_tag>.csv
    - summary_<epoch_tag>.json
    
    Args:
        signals_df: DataFrame with computed signals
        summary: Summary dictionary
        output_dir: Base output directory (e.g., Output/runs/<bundle>/epoch<epoch_tag>)
        epoch_tag: Epoch tag
    """
    # Create electricity_poc directory structure
    poc_dir = output_dir / 'electricity_poc'
    signals_dir = poc_dir / 'signals'
    signals_dir.mkdir(parents=True, exist_ok=True)
    
    # Write headroom CSV
    headroom_df = signals_df[['timestamp_utc', 'headroom_MW']].copy()
    headroom_df['timestamp_utc'] = to_iso_z(headroom_df['timestamp_utc'])
    headroom_path = signals_dir / f'headroom_MW_{epoch_tag}.csv'
    headroom_df.to_csv(headroom_path, index=False)
    print(f"[OK] Wrote headroom signals to {headroom_path}")
    
    # Write tariff CSV
    tariff_df = signals_df[['timestamp_utc', 'tariff_nzd_per_MWh']].copy()
    tariff_df['timestamp_utc'] = to_iso_z(tariff_df['timestamp_utc'])
    tariff_path = signals_dir / f'tariff_nzd_per_MWh_{epoch_tag}.csv'
    tariff_df.to_csv(tariff_path, index=False)
    print(f"[OK] Wrote tariff signals to {tariff_path}")
    
    # Write summary JSON
    summary_path = poc_dir / f'summary_{epoch_tag}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Wrote summary to {summary_path}")


def main():
    """CLI entrypoint for regional electricity PoC."""
    parser = argparse.ArgumentParser(
        description='Regional Electricity PoC: Generate headroom and tariff signals from incremental electricity demand'
    )
    parser.add_argument('--epoch-tag', type=str, required=True,
                       help='Epoch tag (e.g., 2035_EB, 2035_BB)')
    parser.add_argument('--incremental-csv', type=str, required=True,
                       help='Path to incremental_electricity_MW_<epoch_tag>.csv')
    parser.add_argument('--gxp-config-toml', type=str, default=None,
                       help='Path to GXP PoC config TOML (optional, defaults used if missing)')
    parser.add_argument('--output-root', type=str, default='Output',
                       help='Output root directory (default: Output)')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run ID for output path resolution (e.g., <bundle>/epoch<epoch_tag>)')
    
    args = parser.parse_args()
    
    # Resolve paths
    ROOT = repo_root()
    incremental_path = PathlibPath(resolve_path(args.incremental_csv))
    
    # Load GXP config
    gxp_config_path = None
    if args.gxp_config_toml:
        gxp_config_path = PathlibPath(resolve_path(args.gxp_config_toml))
    else:
        # Try default location
        default_config = ROOT / 'Input' / 'gxp' / 'poc_gxp_config.toml'
        if default_config.exists():
            gxp_config_path = default_config
    
    gxp_config = load_gxp_config(gxp_config_path, args.epoch_tag)
    
    # Load incremental electricity
    print(f"Loading incremental electricity from {incremental_path}...")
    incremental_df = load_incremental_electricity(incremental_path)
    print(f"[OK] Loaded {len(incremental_df)} timesteps")
    
    # Compute regional signals
    print("Computing regional electricity signals...")
    signals_df = compute_regional_signals(incremental_df, gxp_config)
    
    # Compute summary
    summary = compute_summary(signals_df, gxp_config, args.epoch_tag)
    
    # Resolve output directory
    output_root = PathlibPath(resolve_path(args.output_root))
    if args.run_id:
        # Parse run_id to extract bundle and epoch structure
        # Format: <bundle>/epoch<epoch_tag> or just <bundle>
        run_id_parts = args.run_id.split('/')
        if len(run_id_parts) >= 2 and run_id_parts[1].startswith('epoch'):
            # Format: <bundle>/epoch<epoch_tag>
            output_dir = output_root / 'runs' / run_id_parts[0] / run_id_parts[1]
        else:
            # Format: <bundle> or <bundle>/...
            output_dir = output_root / 'runs' / args.run_id
    else:
        # Fallback: create epoch-specific directory
        output_dir = output_root / 'runs' / f'epoch{args.epoch_tag}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write outputs
    print(f"Writing outputs to {output_dir}...")
    write_outputs(signals_df, summary, output_dir, args.epoch_tag)
    
    # Print summary
    print("\n" + "="*80)
    print("Regional Electricity PoC Summary")
    print("="*80)
    print(f"Epoch tag: {summary['epoch_tag']}")
    print(f"GXP capacity: {summary['gxp_capacity_MW']:.1f} MW")
    print(f"Baseline import: {summary['baseline_import_MW']:.1f} MW")
    print(f"Annual incremental: {summary['annual_incremental_MWh']:.2f} MWh")
    print(f"Annual shed: {summary['annual_shed_MWh']:.2f} MWh")
    print(f"Max overload: {summary['max_overload_MW']:.2f} MW")
    print("="*80)


if __name__ == '__main__':
    main()
