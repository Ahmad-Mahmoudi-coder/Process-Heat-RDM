"""
Regional Electricity PoC Module

Computes effective electricity tariffs from GXP capacity constraints based on
incremental site demand. Supports two engines:
- simple: Direct calculation without PyPSA (default)
- pypsa: PyPSA-based optimization (requires PyPSA installation)

Consumes SignalsPack GXP hourly data directly from Edendale_GXP submodule.

Syntax check: python -m py_compile src/regional_electricity_poc.py
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

# Try to import PyPSA (optional)
try:
    import pypsa
    HAVE_PYPSA = True
except ImportError:
    HAVE_PYPSA = False

from src.time_utils import parse_any_timestamp, to_iso_z


def load_gxp_hourly(gxp_csv_path: str, for_simple_engine: bool = False) -> pd.DataFrame:
    """
    Load GXP hourly data from SignalsPack.
    
    Expected columns from gxp_hourly_<epoch>.csv:
    - timestamp_utc: UTC ISO-8601 with Z suffix
    - capacity_mw: GXP capacity available
    - baseline_import_mw: Baseline import at GXP
    - headroom_mw: Available headroom (for simple engine)
    - reserve_margin_mw: Reserve margin (for simple engine)
    - tariff_nzd_per_mwh: Base tariff
    - epoch: Epoch year (for simple engine)
    
    Args:
        gxp_csv_path: Path to GXP hourly CSV
        for_simple_engine: If True, expects additional columns (headroom_mw, reserve_margin_mw, epoch)
    """
    df = pd.read_csv(gxp_csv_path)
    
    # Parse timestamp_utc to UTC datetime
    if 'timestamp_utc' not in df.columns:
        raise ValueError(f"timestamp_utc column missing in {gxp_csv_path}")
    
    df['timestamp_utc'] = parse_any_timestamp(df['timestamp_utc'])
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Verify required columns exist
    if for_simple_engine:
        required_cols = ['capacity_mw', 'baseline_import_mw', 'headroom_mw', 
                        'reserve_margin_mw', 'tariff_nzd_per_mwh', 'epoch']
    else:
        required_cols = ['capacity_mw', 'baseline_import_mw', 'tariff_nzd_per_mwh']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {gxp_csv_path}: {missing_cols}")
    
    # Rename to match expected names
    rename_dict = {
        'capacity_mw': 'gxp_capacity_MW',
        'baseline_import_mw': 'baseline_import_MW',
        'tariff_nzd_per_mwh': 'tariff_base_nzd_per_MWh',
        'timestamp_utc': 'timestamp'
    }
    
    if for_simple_engine:
        rename_dict.update({
            'headroom_mw': 'headroom_mw',
            'reserve_margin_mw': 'reserve_margin_mw',
            'epoch': 'epoch'
        })
    
    df = df.rename(columns=rename_dict)
    
    # Select columns to return
    if for_simple_engine:
        return df[['timestamp', 'gxp_capacity_MW', 'baseline_import_MW', 'headroom_mw',
                   'reserve_margin_mw', 'tariff_base_nzd_per_MWh', 'epoch']]
    else:
        return df[['timestamp', 'gxp_capacity_MW', 'baseline_import_MW', 'tariff_base_nzd_per_MWh']]


def load_incremental_demand(incremental_path: str, gxp_timestamps: pd.Series, 
                            strict_alignment: bool = False) -> pd.DataFrame:
    """
    Load hourly incremental electricity demand from site.
    
    If incremental_path is None or file doesn't exist, returns zero series aligned to GXP timestamps.
    
    Expected columns: timestamp or timestamp_utc, incremental_demand_MW (or similar)
    
    Args:
        incremental_path: Path to incremental demand CSV (optional)
        gxp_timestamps: Series of GXP timestamps to align to
        strict_alignment: If True, require exact timestamp match (for simple engine)
    """
    if incremental_path is None or not Path(incremental_path).exists():
        print(f"[WARNING] Incremental demand file not found: {incremental_path}")
        print(f"  Using zero incremental demand series aligned to GXP timestamps")
        return pd.DataFrame({
            'timestamp': gxp_timestamps,
            'incremental_demand_MW': 0.0
        })
    
    df = pd.read_csv(incremental_path)
    
    # Support both timestamp and timestamp_utc
    if 'timestamp_utc' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp_utc'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp'])
    else:
        raise ValueError(f"Incremental demand CSV {incremental_path} must have either 'timestamp' or 'timestamp_utc' column")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Try common column names
    demand_col = None
    for col in df.columns:
        if 'incremental' in col.lower() and 'mw' in col.lower():
            demand_col = col
            break
        elif 'demand' in col.lower() and 'mw' in col.lower() and 'incremental' not in col.lower():
            demand_col = col
            break
    
    if demand_col is None:
        raise ValueError(f"No incremental demand column found in {incremental_path}. Expected column with 'incremental' and 'MW' in name.")
    
    result = df[['timestamp', demand_col]].rename(columns={demand_col: 'incremental_demand_MW'})
    
    if strict_alignment:
        # For simple engine: require exact timestamp match
        result = result.set_index('timestamp')
        gxp_indexed = pd.Series(gxp_timestamps.values, index=gxp_timestamps)
        result = result.reindex(gxp_indexed.index)
        
        if result['incremental_demand_MW'].isna().any():
            raise ValueError(
                f"Incremental demand timestamps do not exactly match GXP timestamps. "
                f"Missing timestamps: {result[result['incremental_demand_MW'].isna()].index.tolist()[:5]}..."
            )
        
        result = result.reset_index()
        result = result.rename(columns={'index': 'timestamp'})
    else:
        # For PyPSA engine: allow reindexing with forward fill
        result = result.set_index('timestamp').reindex(gxp_timestamps, method='ffill').fillna(0.0).reset_index()
    
    return result


# load_tariff is no longer needed - tariff comes from GXP file


def run_simple_engine(gxp_csv_path: str, incremental_path: str = None,
                     output_path: str = None, epoch: str = '2020') -> pd.DataFrame:
    """
    Run simple engine: direct calculation without PyPSA.
    
    Args:
        gxp_csv_path: Path to GXP hourly CSV from SignalsPack
        incremental_path: Optional path to incremental demand CSV
        output_path: Path to output CSV
        epoch: Epoch label
    
    Returns DataFrame with results.
    """
    print("Loading GXP hourly data from SignalsPack...")
    gxp_df = load_gxp_hourly(gxp_csv_path, for_simple_engine=True)
    
    print("Loading incremental demand...")
    incremental_df = load_incremental_demand(incremental_path, gxp_df['timestamp'], strict_alignment=True)
    
    # Merge GXP data with incremental demand
    df = gxp_df.merge(incremental_df, on='timestamp', how='inner')
    
    if len(df) == 0:
        raise ValueError("No matching timestamps found after merging GXP and incremental data")
    
    print(f"Processing {len(df)} hours with simple engine...")
    
    # Simple dispatch logic
    df['inc_served_mw'] = df[['incremental_demand_MW', 'headroom_mw']].min(axis=1)
    df['shed_mw'] = (df['incremental_demand_MW'] - df['headroom_mw']).clip(lower=0.0)
    df['grid_import_mw'] = df['baseline_import_MW'] + df['inc_served_mw']
    df['headroom_remaining_mw'] = df['headroom_mw'] - df['inc_served_mw']
    df['tariff_effective_nzd_per_mwh'] = df['tariff_base_nzd_per_MWh']  # Simple: no VOLL pricing
    
    # Build output DataFrame
    output_df = pd.DataFrame({
        'timestamp_utc': to_iso_z(df['timestamp']),
        'epoch': df['epoch'],
        'baseline_import_mw': df['baseline_import_MW'],
        'incremental_demand_mw': df['incremental_demand_MW'],
        'grid_import_mw': df['grid_import_mw'],
        'shed_mw': df['shed_mw'],
        'headroom_mw': df['headroom_mw'],
        'headroom_remaining_mw': df['headroom_remaining_mw'],
        'tariff_nzd_per_mwh': df['tariff_base_nzd_per_MWh'],
        'tariff_effective_nzd_per_mwh': df['tariff_effective_nzd_per_mwh'],
        'capacity_mw': df['gxp_capacity_MW'],
        'reserve_margin_mw': df['reserve_margin_mw']
    })
    
    # Save output
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
    
    # Summary statistics
    total_shed_mwh = df['shed_mw'].sum()
    max_shed_mw = df['shed_mw'].max()
    min_headroom_remaining_mw = df['headroom_remaining_mw'].min()
    
    print(f"[OK] Simple engine completed")
    print(f"  Total hours: {len(output_df)}")
    print(f"  Total shed: {total_shed_mwh:.2f} MWh")
    print(f"  Max shed: {max_shed_mw:.2f} MW")
    print(f"  Min headroom remaining: {min_headroom_remaining_mw:.2f} MW")
    if output_path:
        print(f"  Saved results to {output_path}")
    
    return output_df


def solve_hourly_opf(baseline_MW: float, incremental_MW: float, capacity_MW: float,
                     tariff_base: float, voll: float = 10000.0) -> dict:
    """
    Solve single-hour OPF for GXP import with capacity constraint.
    
    Args:
        baseline_MW: Baseline import (must be served)
        incremental_MW: Incremental demand from site
        capacity_MW: Maximum GXP capacity
        tariff_base: Base tariff (NZD per MWh)
        voll: Value of lost load (NZD per MWh) for shedding generator
        
    Returns:
        dict with: grid_import_MW, shed_MW, headroom_MW, tariff_effective
    """
    total_demand = baseline_MW + incremental_MW
    
    # Create minimal PyPSA network (only called if HAVE_PYPSA is True)
    if not HAVE_PYPSA:
        raise ImportError("PyPSA is required for PyPSA engine. Install with: pip install pypsa")
    
    n = pypsa.Network()
    
    # Add bus
    n.add("Bus", "bus0")
    
    # Add load
    n.add("Load", "load0", bus="bus0", p_set=total_demand)
    
    # Add import generator (capacity limited)
    max_import = max(0.0, capacity_MW - baseline_MW)  # Available capacity for incremental
    n.add("Generator", "import", bus="bus0", 
          p_min_pu=0.0, p_max_pu=1.0,
          p_nom=max(0.1, max_import),  # At least 0.1 MW to avoid zero capacity
          marginal_cost=tariff_base)
    
    # Add shedding generator (high cost, unlimited capacity)
    n.add("Generator", "shed", bus="bus0",
          p_min_pu=0.0, p_max_pu=1.0,
          p_nom=total_demand,  # Enough to meet all demand
          marginal_cost=voll)
    
    # Solve linear OPF
    try:
        n.lopf(pyomo=False, solver_name='glpk')
        
        # Extract results
        grid_import = n.generators_t.p['import'].iloc[0] if len(n.generators_t.p) > 0 else 0.0
        shed = n.generators_t.p['shed'].iloc[0] if len(n.generators_t.p) > 0 else 0.0
        
        # Ensure grid_import doesn't exceed capacity
        grid_import = min(grid_import, max_import)
        
        # If shedding occurred, effective tariff = VOLL, else = base tariff
        tariff_effective = voll if shed > 1e-6 else tariff_base
        
        # Headroom = capacity - total import
        total_import = baseline_MW + grid_import
        headroom = max(0.0, capacity_MW - total_import)
        
    except Exception as e:
        # Fallback: if solver fails, use simple logic
        print(f"Warning: OPF solver failed: {e}. Using simple dispatch logic.")
        if total_demand <= capacity_MW:
            grid_import = incremental_MW
            shed = 0.0
            tariff_effective = tariff_base
        else:
            grid_import = max(0.0, capacity_MW - baseline_MW)
            shed = total_demand - capacity_MW
            tariff_effective = voll
        
        total_import = baseline_MW + grid_import
        headroom = max(0.0, capacity_MW - total_import)
    
    return {
        'grid_import_MW': grid_import,
        'shed_MW': shed,
        'headroom_MW': headroom,
        'tariff_effective_nzd_per_MWh': tariff_effective
    }


def run_regional_poc_pypsa(gxp_csv_path: str, incremental_path: str = None,
                           output_path: str = None, voll: float = 10000.0) -> pd.DataFrame:
    """
    Run regional electricity PoC using PyPSA engine.
    
    Args:
        gxp_csv_path: Path to GXP hourly CSV from SignalsPack
        incremental_path: Optional path to incremental demand CSV (defaults to zero series)
        output_path: Path to output CSV
        voll: Value of lost load (NZD per MWh)
    
    Returns DataFrame with results.
    """
    if not HAVE_PYPSA:
        raise ImportError("PyPSA is required for PyPSA engine. Install with: pip install pypsa")
    
    print("Loading GXP hourly data from SignalsPack...")
    gxp_df = load_gxp_hourly(gxp_csv_path, for_simple_engine=False)
    
    print("Loading incremental demand...")
    incremental_df = load_incremental_demand(incremental_path, gxp_df['timestamp'], strict_alignment=False)
    
    # Merge GXP data with incremental demand
    df = gxp_df.merge(incremental_df, on='timestamp', how='inner')
    
    if len(df) == 0:
        raise ValueError("No matching timestamps found after merging GXP and incremental data")
    
    print(f"Processing {len(df)} hours with PyPSA engine...")
    
    # Solve OPF for each hour
    results = []
    for idx, row in df.iterrows():
        result = solve_hourly_opf(
            baseline_MW=row['baseline_import_MW'],
            incremental_MW=row['incremental_demand_MW'],
            capacity_MW=row['gxp_capacity_MW'],
            tariff_base=row['tariff_base_nzd_per_MWh'],
            voll=voll
        )
        results.append(result)
    
    # Combine results
    results_df = pd.DataFrame(results)
    output_df = pd.concat([df[['timestamp', 'gxp_capacity_MW', 'baseline_import_MW', 
                               'incremental_demand_MW', 'tariff_base_nzd_per_MWh']], 
                           results_df], axis=1)
    
    # Rename for clarity
    output_df = output_df.rename(columns={
        'incremental_demand_MW': 'incremental_import_MW'
    })
    
    # Convert timestamp to ISO Z format for output
    output_df['timestamp_utc'] = to_iso_z(output_df['timestamp'])
    output_df = output_df.drop(columns=['timestamp'])
    
    # Reorder columns with timestamp_utc first
    output_df = output_df[[
        'timestamp_utc',
        'gxp_capacity_MW',
        'baseline_import_MW',
        'incremental_import_MW',
        'grid_import_MW',
        'shed_MW',
        'headroom_MW',
        'tariff_base_nzd_per_MWh',
        'tariff_effective_nzd_per_MWh'
    ]]
    
    # Save output
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
    
    # Summary statistics
    total_shed_mwh = output_df['shed_MW'].sum()
    max_shed_mw = output_df['shed_MW'].max()
    min_headroom_mw = output_df['headroom_MW'].min()
    
    print(f"[OK] PyPSA engine completed")
    print(f"  Total hours: {len(output_df)}")
    print(f"  Total shed: {total_shed_mwh:.2f} MWh")
    print(f"  Max shed: {max_shed_mw:.2f} MW")
    print(f"  Min headroom: {min_headroom_mw:.2f} MW")
    if output_path:
        print(f"  Saved results to {output_path}")
    
    return output_df


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='Regional Electricity PoC: Compute effective tariffs from GXP capacity constraints'
    )
    parser.add_argument('--epoch', type=str, default='2020',
                       help='Epoch label (default: 2020)')
    parser.add_argument('--gxp-csv', type=str, required=True,
                       help='Path to GXP hourly CSV from SignalsPack (e.g., modules/edendale_gxp/outputs_latest/gxp_hourly_2020.csv)')
    parser.add_argument('--incremental-csv', type=str, default=None,
                       dest='incremental',  # Keep internal name for backward compatibility
                       help='Optional path to incremental demand CSV (defaults to zero series if not provided)')
    parser.add_argument('--out', type=str,
                       default='Output/regional_electricity_signals_2020.csv',
                       help='Path to output SignalsPack CSV')
    parser.add_argument('--engine', type=str, choices=['simple', 'pypsa'], default='simple',
                       help='Engine to use: simple (default, no PyPSA) or pypsa (requires PyPSA)')
    parser.add_argument('--voll', type=float, default=10000.0,
                       help='Value of lost load (NZD per MWh) for shedding - PyPSA engine only (default: 10000.0)')
    
    args = parser.parse_args()
    
    # Check if PyPSA is required but not available
    if args.engine == 'pypsa' and not HAVE_PYPSA:
        print("[ERROR] PyPSA engine requested but PyPSA is not installed.", file=sys.stderr)
        print("  Install PyPSA with: pip install pypsa", file=sys.stderr)
        print("  Or use --engine simple (default) which does not require PyPSA", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.engine == 'simple':
            run_simple_engine(
                gxp_csv_path=args.gxp_csv,
                incremental_path=args.incremental,
                output_path=args.out,
                epoch=args.epoch
            )
        else:  # pypsa
            run_regional_poc_pypsa(
                gxp_csv_path=args.gxp_csv,
                incremental_path=args.incremental,
                output_path=args.out,
                voll=args.voll
            )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

