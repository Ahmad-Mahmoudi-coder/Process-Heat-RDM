"""
Regional Electricity PoC Module

Uses PyPSA core to model Edendale GXP capacity constraints and compute
effective electricity tariffs based on incremental site demand.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

try:
    import pypsa
except ImportError:
    raise ImportError("PyPSA is required. Install with: pip install pypsa")


def load_baseline_import(baseline_path: str) -> pd.DataFrame:
    """
    Load hourly baseline import data for Edendale GXP.
    
    Expected columns: timestamp, baseline_import_MW
    """
    df = pd.read_csv(baseline_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    if 'baseline_import_MW' not in df.columns:
        raise ValueError(f"baseline_import_MW column missing in {baseline_path}")
    
    return df[['timestamp', 'baseline_import_MW']]


def load_gxp_capacity(capacity_path: str) -> pd.DataFrame:
    """
    Load hourly GXP capacity data.
    
    Expected columns: timestamp, gxp_capacity_MW
    """
    df = pd.read_csv(capacity_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    if 'gxp_capacity_MW' not in df.columns:
        raise ValueError(f"gxp_capacity_MW column missing in {capacity_path}")
    
    return df[['timestamp', 'gxp_capacity_MW']]


def load_incremental_demand(incremental_path: str) -> pd.DataFrame:
    """
    Load hourly incremental electricity demand from site.
    
    Expected columns: timestamp, incremental_demand_MW (or similar)
    """
    df = pd.read_csv(incremental_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
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
    
    return df[['timestamp', demand_col]].rename(columns={demand_col: 'incremental_demand_MW'})


def load_tariff(tariff_path: str) -> pd.DataFrame:
    """
    Load hourly base tariff data.
    
    Expected columns: timestamp, tariff_base_nzd_per_MWh
    """
    df = pd.read_csv(tariff_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    if 'tariff_base_nzd_per_MWh' not in df.columns:
        raise ValueError(f"tariff_base_nzd_per_MWh column missing in {tariff_path}")
    
    return df[['timestamp', 'tariff_base_nzd_per_MWh']]


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
    
    # Create minimal PyPSA network
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


def run_regional_poc(baseline_path: str, capacity_path: str, incremental_path: str,
                     tariff_path: str, output_path: str, voll: float = 10000.0) -> pd.DataFrame:
    """
    Run regional electricity PoC for all hours.
    
    Returns DataFrame with results.
    """
    print("Loading input data...")
    baseline_df = load_baseline_import(baseline_path)
    capacity_df = load_gxp_capacity(capacity_path)
    incremental_df = load_incremental_demand(incremental_path)
    tariff_df = load_tariff(tariff_path)
    
    # Merge all dataframes on timestamp
    df = baseline_df.merge(capacity_df, on='timestamp', how='inner')
    df = df.merge(incremental_df, on='timestamp', how='inner')
    df = df.merge(tariff_df, on='timestamp', how='inner')
    
    if len(df) == 0:
        raise ValueError("No matching timestamps found after merging input files")
    
    print(f"Processing {len(df)} hours...")
    
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
    
    # Reorder columns
    output_df = output_df[[
        'timestamp',
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
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print(f"[OK] Saved results to {output_path}")
    print(f"  Total hours: {len(output_df)}")
    print(f"  Hours with shedding: {(output_df['shed_MW'] > 1e-6).sum()}")
    print(f"  Max shedding: {output_df['shed_MW'].max():.2f} MW")
    print(f"  Min headroom: {output_df['headroom_MW'].min():.2f} MW")
    
    return output_df


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='Regional Electricity PoC: Compute effective tariffs from GXP capacity constraints'
    )
    parser.add_argument('--epoch', type=str, default='2020',
                       help='Epoch label (default: 2020)')
    parser.add_argument('--baseline', type=str,
                       default='Input/gxp_baseline_import_hourly_2020.csv',
                       help='Path to baseline import CSV')
    parser.add_argument('--capacity', type=str,
                       default='Input/gxp_headroom_hourly_2020.csv',
                       help='Path to GXP capacity CSV')
    parser.add_argument('--incremental', type=str,
                       default='Output/site_electricity_incremental_2020.csv',
                       help='Path to incremental demand CSV')
    parser.add_argument('--tariff', type=str,
                       default='Input/gxp_tariff_hourly_2020.csv',
                       help='Path to base tariff CSV')
    parser.add_argument('--out', type=str,
                       default='Output/regional_electricity_signals_2020.csv',
                       help='Path to output SignalsPack CSV')
    parser.add_argument('--voll', type=float, default=10000.0,
                       help='Value of lost load (NZD per MWh) for shedding (default: 10000.0)')
    
    args = parser.parse_args()
    
    try:
        run_regional_poc(
            baseline_path=args.baseline,
            capacity_path=args.capacity,
            incremental_path=args.incremental,
            tariff_path=args.tariff,
            output_path=args.out,
            voll=args.voll
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

