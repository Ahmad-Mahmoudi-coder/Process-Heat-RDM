"""
Site Utility Dispatch Module for 2020 Baseline

Allocates hourly heat demand across site utilities (coal boilers) using
proportional capacity dispatch and computes fuel consumption and CO2 emissions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
import sys


def load_hourly_demand(path: str) -> pd.DataFrame:
    """
    Load hourly demand CSV, parse timestamp, sort by time, and validate.
    
    Args:
        path: Path to hourly demand CSV file
        
    Returns:
        DataFrame with timestamp and heat_demand_MW columns
        
    Raises:
        ValueError: If number of rows is not 8760 (or 8784 for leap year)
    """
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Check row count (2020 is a leap year, so 8784 hours)
    if len(df) not in [8760, 8784]:
        raise ValueError(f"Expected 8760 or 8784 rows (non-leap/leap year), got {len(df)}")
    
    return df[['timestamp', 'heat_demand_MW']]


def load_utilities(path: str) -> pd.DataFrame:
    """
    Load site utilities CSV and validate required fields.
    
    Args:
        path: Path to site utilities CSV file
        
    Returns:
        DataFrame with utility information
        
    Raises:
        ValueError: If validation checks fail
    """
    df = pd.read_csv(path)
    
    # Check all status_2020 == "existing"
    if not (df['status_2020'] == 'existing').all():
        invalid = df[df['status_2020'] != 'existing']
        raise ValueError(f"Found units with status_2020 != 'existing': {invalid[['unit_id', 'status_2020']].to_dict('records')}")
    
    # Check max_heat_MW > 0
    if not (df['max_heat_MW'] > 0).all():
        invalid = df[df['max_heat_MW'] <= 0]
        raise ValueError(f"Found units with max_heat_MW <= 0: {invalid[['unit_id', 'max_heat_MW']].to_dict('records')}")
    
    # Check 0 < efficiency_th <= 1
    if not ((df['efficiency_th'] > 0) & (df['efficiency_th'] <= 1)).all():
        invalid = df[~((df['efficiency_th'] > 0) & (df['efficiency_th'] <= 1))]
        raise ValueError(f"Found units with efficiency_th outside (0, 1]: {invalid[['unit_id', 'efficiency_th']].to_dict('records')}")
    
    # Check 0 < availability_factor <= 1
    if not ((df['availability_factor'] > 0) & (df['availability_factor'] <= 1)).all():
        invalid = df[~((df['availability_factor'] > 0) & (df['availability_factor'] <= 1))]
        raise ValueError(f"Found units with availability_factor outside (0, 1]: {invalid[['unit_id', 'availability_factor']].to_dict('records')}")
    
    return df


def allocate_baseline_dispatch(demand_df: pd.DataFrame, util_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Allocate hourly heat demand across utilities proportionally by capacity.
    
    For each hour, allocates heat to each unit proportional to its capacity:
    q_u_t = Q_t * (max_heat_MW_u / P_total)
    
    Args:
        demand_df: DataFrame with timestamp and heat_demand_MW columns
        util_df: DataFrame with utility information including max_heat_MW, 
                 efficiency_th, co2_factor_t_per_MWh_fuel
        
    Returns:
        Tuple of (long-form DataFrame, wide-form DataFrame)
        Long-form columns: timestamp, unit_id, heat_MW, fuel_MWh, co2_tonnes
        Wide-form columns: timestamp, total_heat_MW, CB1_MW, CB2_MW, ...
    """
    # Compute total installed capacity
    P_total = util_df['max_heat_MW'].sum()
    
    # Create allocation factors (capacity share for each unit)
    util_df = util_df.copy()
    util_df['capacity_share'] = util_df['max_heat_MW'] / P_total
    
    # Initialize long-form results
    long_results = []
    
    # For each hour, allocate demand proportionally
    for _, hour_row in demand_df.iterrows():
        timestamp = hour_row['timestamp']
        total_demand = hour_row['heat_demand_MW']
        
        for _, util_row in util_df.iterrows():
            unit_id = util_row['unit_id']
            capacity_share = util_row['capacity_share']
            efficiency = util_row['efficiency_th']
            co2_factor = util_row['co2_factor_t_per_MWh_fuel']
            
            # Allocate heat proportionally
            heat_MW = total_demand * capacity_share
            
            # Compute fuel consumption
            fuel_MWh = heat_MW / efficiency if efficiency > 0 else 0.0
            
            # Compute CO2 emissions
            co2_tonnes = fuel_MWh * co2_factor
            
            long_results.append({
                'timestamp': timestamp,
                'unit_id': unit_id,
                'heat_MW': heat_MW,
                'fuel_MWh': fuel_MWh,
                'co2_tonnes': co2_tonnes,
            })
    
    dispatch_long = pd.DataFrame(long_results)
    
    # Create wide-form pivot
    dispatch_wide = dispatch_long.pivot_table(
        index='timestamp',
        columns='unit_id',
        values='heat_MW',
        aggfunc='sum'
    ).reset_index()
    
    # Rename unit columns to include _MW suffix
    unit_columns = [col for col in dispatch_wide.columns if col != 'timestamp']
    rename_dict = {col: f"{col}_MW" for col in unit_columns}
    dispatch_wide = dispatch_wide.rename(columns=rename_dict)
    
    # Add total_heat_MW column
    dispatch_wide['total_heat_MW'] = dispatch_wide[[col for col in dispatch_wide.columns if col.endswith('_MW') and col != 'total_heat_MW']].sum(axis=1)
    
    # Reorder columns: timestamp, total_heat_MW, then unit columns
    unit_cols_sorted = sorted([col for col in dispatch_wide.columns if col.endswith('_MW') and col != 'total_heat_MW'])
    dispatch_wide = dispatch_wide[['timestamp', 'total_heat_MW'] + unit_cols_sorted]
    
    return dispatch_long, dispatch_wide


def plot_dispatch_stack(dispatch_wide_path: str, output_path: str):
    """
    Generate stacked area plot of unit dispatch over time.
    
    Args:
        dispatch_wide_path: Path to wide-form dispatch CSV
        output_path: Path to save the figure
    """
    df = pd.read_csv(dispatch_wide_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Get unit columns (all columns ending in _MW except total_heat_MW)
    unit_cols = [col for col in df.columns if col.endswith('_MW') and col != 'total_heat_MW']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create stacked area plot
    ax.stackplot(df['timestamp'], 
                 [df[col] for col in unit_cols],
                 labels=[col.replace('_MW', '') for col in unit_cols],
                 alpha=0.7)
    
    # Overlay thin black line for total demand
    ax.plot(df['timestamp'], df['total_heat_MW'],
            color='black', linewidth=1.0, linestyle='-', label='Total demand', zorder=10)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Heat Demand (MW)', fontsize=12)
    ax.set_title('Unit Dispatch Stack - 2020', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def main():
    """CLI entrypoint for site dispatch computation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute 2020 baseline site utility dispatch')
    parser.add_argument('--demand-csv', default='Output/hourly_heat_demand_2020.csv',
                       help='Path to hourly demand CSV')
    parser.add_argument('--utilities-csv', default='Input/site_utilities_2020.csv',
                       help='Path to site utilities CSV')
    parser.add_argument('--out-dispatch-long', default='Output/site_dispatch_2020_long.csv',
                       help='Output path for long-form dispatch CSV')
    parser.add_argument('--out-dispatch-wide', default='Output/site_dispatch_2020_wide.csv',
                       help='Output path for wide-form dispatch CSV')
    parser.add_argument('--plot', action='store_true',
                       help='Generate dispatch stack plot')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading hourly demand from {args.demand_csv}...")
    demand_df = load_hourly_demand(args.demand_csv)
    
    print(f"Loading utilities from {args.utilities_csv}...")
    util_df = load_utilities(args.utilities_csv)
    
    print(f"Found {len(util_df)} utilities with total capacity {util_df['max_heat_MW'].sum():.2f} MW")
    
    # Compute dispatch
    print("Computing baseline dispatch...")
    dispatch_long, dispatch_wide = allocate_baseline_dispatch(demand_df, util_df)
    
    # Save outputs
    print(f"Saving long-form dispatch to {args.out_dispatch_long}...")
    dispatch_long.to_csv(args.out_dispatch_long, index=False)
    
    print(f"Saving wide-form dispatch to {args.out_dispatch_wide}...")
    dispatch_wide.to_csv(args.out_dispatch_wide, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("Annual Summary by Unit")
    print("="*60)
    
    # Aggregate by unit
    annual_by_unit = dispatch_long.groupby('unit_id').agg({
        'heat_MW': 'sum',
        'fuel_MWh': 'sum',
        'co2_tonnes': 'sum'
    })
    
    # Convert MW to GWh
    annual_by_unit['heat_GWh'] = annual_by_unit['heat_MW'] / 1000.0
    annual_by_unit['fuel_GWh'] = annual_by_unit['fuel_MWh'] / 1000.0
    
    for unit_id in annual_by_unit.index:
        row = annual_by_unit.loc[unit_id]
        print(f"\n{unit_id}:")
        print(f"  Annual heat:     {row['heat_GWh']:8.2f} GWh")
        print(f"  Annual fuel:     {row['fuel_GWh']:8.2f} GWh_fuel")
        print(f"  Annual CO2:      {row['co2_tonnes']:8.2f} tCO2")
    
    total_co2 = annual_by_unit['co2_tonnes'].sum()
    print(f"\n{'Total CO2:':<20} {total_co2:8.2f} tCO2")
    print("="*60)
    
    # Generate plot if requested
    if args.plot:
        print("\nGenerating dispatch stack plot...")
        plot_path = 'Output/Figures/heat_2020_unit_stack.png'
        plot_dispatch_stack(args.out_dispatch_wide, plot_path)


if __name__ == '__main__':
    main()

