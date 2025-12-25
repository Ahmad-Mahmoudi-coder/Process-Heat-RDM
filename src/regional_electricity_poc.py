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
import tomllib

# Try to import PyPSA (optional)
try:
    import pypsa
    HAVE_PYPSA = True
except ImportError:
    HAVE_PYPSA = False

from src.time_utils import parse_any_timestamp, to_iso_z
from src.path_utils import repo_root, input_root, resolve_path


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


def load_grid_upgrades(upgrades_config_path: str = None) -> list[dict]:
    """
    Load grid upgrade options from TOML config.
    
    Expected format in grid_upgrades.toml:
    [[upgrades]]
    capacity_MW = 0
    annual_cost_nzd = 0.0
    
    [[upgrades]]
    capacity_MW = 10
    annual_cost_nzd = 50000.0
    
    Returns list of upgrade options sorted by capacity_MW (ascending).
    """
    if upgrades_config_path is None:
        # Default to Input/signals/grid_upgrades.toml
        upgrades_config_path = input_root() / 'signals' / 'grid_upgrades.toml'
    else:
        upgrades_config_path = resolve_path(upgrades_config_path)
    
    if not upgrades_config_path.exists():
        # Return default: no upgrade option
        return [{'capacity_MW': 0, 'annual_cost_nzd': 0.0}]
    
    with open(upgrades_config_path, 'rb') as f:
        config = tomllib.load(f)
    
    upgrades = config.get('upgrades', [])
    if not upgrades:
        return [{'capacity_MW': 0, 'annual_cost_nzd': 0.0}]
    
    # Validate and sort by capacity
    validated = []
    for upgrade in upgrades:
        if 'capacity_MW' not in upgrade or 'annual_cost_nzd' not in upgrade:
            print(f"[WARN] Skipping invalid upgrade option: {upgrade}")
            continue
        validated.append({
            'capacity_MW': float(upgrade['capacity_MW']),
            'annual_cost_nzd': float(upgrade['annual_cost_nzd'])
        })
    
    # Sort by capacity (ascending)
    validated.sort(key=lambda x: x['capacity_MW'])
    
    # Ensure zero option exists
    if validated[0]['capacity_MW'] != 0:
        validated.insert(0, {'capacity_MW': 0, 'annual_cost_nzd': 0.0})
    
    return validated


def choose_optimal_upgrade(incremental_demand_MW: float, headroom_MW: float,
                           upgrade_options: list[dict], voll: float = 10000.0) -> dict:
    """
    Choose the cheapest upgrade option that minimizes total cost (upgrade + shed*VOLL).
    
    Args:
        incremental_demand_MW: Incremental demand for this hour
        headroom_MW: Available headroom before upgrade
        upgrade_options: List of upgrade options (from load_grid_upgrades)
        voll: Value of lost load (NZD per MWh)
    
    Returns:
        dict with keys: upgrade_selected_MW, upgrade_annual_cost_nzd, headroom_effective_MW,
                        shed_MW, total_cost_nzd
    """
    if incremental_demand_MW <= headroom_MW:
        # No shed needed, no upgrade needed
        return {
            'upgrade_selected_MW': 0.0,
            'upgrade_annual_cost_nzd': 0.0,
            'headroom_effective_MW': headroom_MW,
            'shed_MW': 0.0,
            'total_cost_nzd': 0.0
        }
    
    # Calculate shed without upgrade
    shed_no_upgrade = incremental_demand_MW - headroom_MW
    cost_no_upgrade = shed_no_upgrade * voll  # Per-hour cost (VOLL is per MWh)
    
    best_option = {
        'upgrade_selected_MW': 0.0,
        'upgrade_annual_cost_nzd': 0.0,
        'headroom_effective_MW': headroom_MW,
        'shed_MW': shed_no_upgrade,
        'total_cost_nzd': cost_no_upgrade
    }
    
    # Evaluate each upgrade option
    for upgrade in upgrade_options:
        upgrade_capacity = upgrade['capacity_MW']
        upgrade_annual_cost = upgrade['annual_cost_nzd']
        
        # Effective headroom with this upgrade
        headroom_effective = headroom_MW + upgrade_capacity
        
        # Shed with this upgrade
        shed = max(0.0, incremental_demand_MW - headroom_effective)
        
        # Total cost: upgrade (prorated to hourly) + shed*VOLL
        # Prorate annual cost to hourly (assume 8760 hours per year)
        upgrade_hourly_cost = upgrade_annual_cost / 8760.0
        shed_cost = shed * voll
        total_cost = upgrade_hourly_cost + shed_cost
        
        # Choose option with lowest total cost
        if total_cost < best_option['total_cost_nzd']:
            best_option = {
                'upgrade_selected_MW': upgrade_capacity,
                'upgrade_annual_cost_nzd': upgrade_annual_cost,
                'headroom_effective_MW': headroom_effective,
                'shed_MW': shed,
                'total_cost_nzd': total_cost
            }
    
    return best_option


def run_simple_engine(gxp_csv_path: str, incremental_path: str = None,
                     output_path: str = None, epoch: str = '2020',
                     upgrades_config_path: str = None, voll: float = 10000.0) -> pd.DataFrame:
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
    
    # Load grid upgrade options if available
    upgrade_options = load_grid_upgrades(upgrades_config_path)
    if len(upgrade_options) > 1:
        print(f"[INFO] Loaded {len(upgrade_options)} grid upgrade options")
    else:
        print(f"[INFO] No grid upgrade options configured (using default: no upgrade)")
    
    # Simple dispatch logic with upgrade selection
    upgrade_results = []
    for idx, row in df.iterrows():
        result = choose_optimal_upgrade(
            incremental_demand_MW=row['incremental_demand_MW'],
            headroom_MW=row['headroom_mw'],
            upgrade_options=upgrade_options,
            voll=voll
        )
        upgrade_results.append(result)
    
    upgrade_df = pd.DataFrame(upgrade_results)
    
    # Use effective headroom (after upgrade) for dispatch
    df['headroom_effective_mw'] = upgrade_df['headroom_effective_MW']
    df['inc_served_mw'] = df[['incremental_demand_MW', 'headroom_effective_mw']].min(axis=1)
    df['shed_mw'] = upgrade_df['shed_MW']
    df['grid_import_mw'] = df['baseline_import_MW'] + df['inc_served_mw']
    df['headroom_remaining_mw'] = df['headroom_effective_mw'] - df['inc_served_mw']
    df['upgrade_selected_MW'] = upgrade_df['upgrade_selected_MW']
    df['upgrade_annual_cost_nzd'] = upgrade_df['upgrade_annual_cost_nzd']
    df['tariff_effective_nzd_per_mwh'] = df['tariff_base_nzd_per_MWh']  # Simple: no VOLL pricing
    
    # Build output DataFrame
    output_df = pd.DataFrame({
        'timestamp_utc': to_iso_z(df['timestamp']),
        'epoch': df['epoch'],
        'baseline_import_mw': df['baseline_import_MW'],
        'incremental_demand_mw': df['incremental_demand_MW'],
        'grid_import_mw': df['grid_import_mw'],
        'shed_mw': df['shed_mw'],
        'headroom_mw': df['headroom_mw'],  # Original headroom (before upgrade)
        'headroom_effective_MW': df['headroom_effective_mw'],  # After upgrade
        'headroom_remaining_mw': df['headroom_remaining_mw'],
        'upgrade_selected_MW': df['upgrade_selected_MW'],
        'upgrade_annual_cost_nzd': df['upgrade_annual_cost_nzd'],
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
    total_shed_mwh = output_df['shed_mw'].sum()
    max_shed_mw = output_df['shed_mw'].max()
    min_headroom_mw = output_df['headroom_mw'].min()
    min_headroom_effective_mw = output_df['headroom_effective_MW'].min()
    total_upgrade_cost_nzd = output_df['upgrade_annual_cost_nzd'].sum() / 8760.0 * len(output_df)  # Prorate to actual hours
    hours_with_upgrade = (output_df['upgrade_selected_MW'] > 0).sum()
    
    print(f"[OK] Simple engine completed")
    print(f"  Total hours: {len(output_df)}")
    print(f"  Total shed: {total_shed_mwh:.2f} MWh")
    print(f"  Max shed: {max_shed_mw:.2f} MW")
    print(f"  Min headroom (original): {min_headroom_mw:.2f} MW")
    print(f"  Min headroom (effective, after upgrade): {min_headroom_effective_mw:.2f} MW")
    if hours_with_upgrade > 0:
        print(f"  Hours with upgrade selected: {hours_with_upgrade} ({100.0 * hours_with_upgrade / len(output_df):.1f}%)")
        print(f"  Total upgrade cost (prorated): {total_upgrade_cost_nzd:.2f} NZD")
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
                       help='Value of lost load (NZD per MWh) for shedding and upgrade decisions (default: 10000.0)')
    parser.add_argument('--upgrades-config', type=str, default=None,
                       help='Path to grid_upgrades.toml config (default: Input/signals/grid_upgrades.toml)')
    
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
                epoch=args.epoch,
                upgrades_config_path=args.upgrades_config,
                voll=args.voll
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

