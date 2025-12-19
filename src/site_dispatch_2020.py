"""
Site Utility Dispatch Module for 2020 Baseline

Allocates hourly heat demand across site utilities (coal boilers) using
proportional capacity dispatch and computes fuel consumption and CO2 emissions.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import sys

from src.load_signals import load_signals_config, get_signals_for_epoch
from src.time_utils import parse_any_timestamp, to_iso_z
from typing import Dict


def load_hourly_demand(path: str) -> pd.DataFrame:
    """
    Load hourly demand CSV, parse timestamp, sort by time, and validate.
    
    Supports both 'timestamp' and 'timestamp_utc' columns for backward compatibility.
    Normalizes to timestamp_utc with UTC timezone.
    
    Args:
        path: Path to hourly demand CSV file
        
    Returns:
        DataFrame with timestamp_utc (UTC datetime) and heat_demand_MW columns
        
    Raises:
        ValueError: If number of rows is not 8760 (or 8784 for leap year)
    """
    df = pd.read_csv(path)
    
    # Normalize timestamp column (support both 'timestamp' and 'timestamp_utc')
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = parse_any_timestamp(df['timestamp_utc'])
    elif 'timestamp' in df.columns:
        # Backward compatibility: normalize old 'timestamp' column
        df['timestamp_utc'] = parse_any_timestamp(df['timestamp'])
    else:
        raise ValueError(f"CSV file {path} must have either 'timestamp' or 'timestamp_utc' column")
    
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Check row count (2020 is a leap year, so 8784 hours)
    if len(df) not in [8760, 8784]:
        raise ValueError(f"Expected 8760 or 8784 rows (non-leap/leap year), got {len(df)}")
    
    return df[['timestamp_utc', 'heat_demand_MW']]


def load_utilities(path: str) -> pd.DataFrame:
    """
    Load site utilities CSV and validate required fields.
    
    Required columns: unit_id, site_id, tech_type, fuel, status_2020, max_heat_MW,
                     min_load_frac, efficiency_th, availability_factor, co2_factor_t_per_MWh_fuel
    
    Optional columns (defaults if missing):
    - fixed_on_cost_nzd_per_h (default 0.0)
    - startup_cost_nzd (default 0.0)
    - var_om_nzd_per_MWh_heat (default 0.0)
    - min_up_time_h (default 0)
    - min_down_time_h (default 0)
    
    Args:
        path: Path to site utilities CSV file
        
    Returns:
        DataFrame with utility information (all columns, with defaults filled)
        
    Raises:
        ValueError: If validation checks fail
    """
    df = pd.read_csv(path)
    
    # Required columns
    required_cols = ['unit_id', 'site_id', 'tech_type', 'fuel', 'status_2020', 
                     'max_heat_MW', 'min_load_frac', 'efficiency_th', 
                     'availability_factor', 'co2_factor_t_per_MWh_fuel']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {path}: {missing_cols}")
    
    # Check unit_id unique
    if df['unit_id'].duplicated().any():
        duplicates = df[df['unit_id'].duplicated(keep=False)]['unit_id'].unique()
        raise ValueError(f"Duplicate unit_id values in {path}: {list(duplicates)}")
    
    # Check all status_2020 == "existing"
    if not (df['status_2020'] == 'existing').all():
        invalid = df[df['status_2020'] != 'existing']
        raise ValueError(f"Found units with status_2020 != 'existing' in {path}: {invalid[['unit_id', 'status_2020']].to_dict('records')}")
    
    # Check max_heat_MW > 0
    if not (df['max_heat_MW'] > 0).all():
        invalid = df[df['max_heat_MW'] <= 0]
        raise ValueError(f"Found units with max_heat_MW <= 0 in {path}: {invalid[['unit_id', 'max_heat_MW']].to_dict('records')}")
    
    # Check 0 < efficiency_th <= 1
    if not ((df['efficiency_th'] > 0) & (df['efficiency_th'] <= 1)).all():
        invalid = df[~((df['efficiency_th'] > 0) & (df['efficiency_th'] <= 1))]
        raise ValueError(f"Found units with efficiency_th outside (0, 1] in {path}: {invalid[['unit_id', 'efficiency_th']].to_dict('records')}")
    
    # Check 0 <= min_load_frac < 1
    if not ((df['min_load_frac'] >= 0) & (df['min_load_frac'] < 1)).all():
        invalid = df[~((df['min_load_frac'] >= 0) & (df['min_load_frac'] < 1))]
        raise ValueError(f"Found units with min_load_frac outside [0, 1) in {path}: {invalid[['unit_id', 'min_load_frac']].to_dict('records')}")
    
    # Check availability_factor in (0, 1]
    if not ((df['availability_factor'] > 0) & (df['availability_factor'] <= 1)).all():
        invalid = df[~((df['availability_factor'] > 0) & (df['availability_factor'] <= 1))]
        raise ValueError(f"Found units with availability_factor outside (0, 1] in {path}: {invalid[['unit_id', 'availability_factor']].to_dict('records')}")
    
    # Set defaults for optional columns
    optional_defaults = {
        'fixed_on_cost_nzd_per_h': 0.0,
        'startup_cost_nzd': 0.0,
        'var_om_nzd_per_MWh_heat': 0.0,
        'min_up_time_h': 0,
        'min_down_time_h': 0,
    }
    
    for col, default_val in optional_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            # Fill NaN values with defaults
            df[col] = df[col].fillna(default_val)
    
    return df


def allocate_baseline_dispatch(demand_df: pd.DataFrame, util_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Allocate hourly heat demand across utilities proportionally by capacity.
    
    For each hour, allocates heat to each unit proportional to its capacity:
    q_u_t = Q_t * (max_heat_MW_u / P_total)
    
    Args:
        demand_df: DataFrame with timestamp_utc and heat_demand_MW columns
        util_df: DataFrame with utility information including max_heat_MW, 
                 efficiency_th, co2_factor_t_per_MWh_fuel
        
    Returns:
        Tuple of (long-form DataFrame, wide-form DataFrame)
        Long-form columns: timestamp_utc, unit_id, heat_MW, fuel_MWh, co2_tonnes
        Wide-form columns: timestamp_utc, total_heat_MW, CB1_MW, CB2_MW, ...
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
        timestamp = hour_row['timestamp_utc']
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
                'timestamp_utc': timestamp,
                'unit_id': unit_id,
                'heat_MW': heat_MW,
                'fuel_MWh': fuel_MWh,
                'co2_tonnes': co2_tonnes,
            })
    
    dispatch_long = pd.DataFrame(long_results)
    
    # Create wide-form pivot
    dispatch_wide = dispatch_long.pivot_table(
        index='timestamp_utc',
        columns='unit_id',
        values='heat_MW',
        aggfunc='sum'
    ).reset_index()
    
    # Rename unit columns to include _MW suffix
    unit_columns = [col for col in dispatch_wide.columns if col != 'timestamp_utc']
    rename_dict = {col: f"{col}_MW" for col in unit_columns}
    dispatch_wide = dispatch_wide.rename(columns=rename_dict)
    
    # Add total_heat_MW column
    dispatch_wide['total_heat_MW'] = dispatch_wide[[col for col in dispatch_wide.columns if col.endswith('_MW') and col != 'total_heat_MW']].sum(axis=1)
    
    # Reorder columns: timestamp_utc, total_heat_MW, then unit columns
    unit_cols_sorted = sorted([col for col in dispatch_wide.columns if col.endswith('_MW') and col != 'total_heat_MW'])
    dispatch_wide = dispatch_wide[['timestamp_utc', 'total_heat_MW'] + unit_cols_sorted]
    
    return dispatch_long, dispatch_wide


def compute_marginal_cost_heat(util_row: pd.Series, signals: Dict[str, float]) -> float:
    """
    Compute marginal cost of heat for a unit.
    
    mc_heat = (fuel_price / efficiency) + (ets_price * co2_factor / efficiency) + var_om
    
    Args:
        util_row: Single row from utilities DataFrame
        signals: Signals dict from get_signals_for_epoch
        
    Returns:
        Marginal cost in NZD per MWh_heat
    """
    efficiency = util_row['efficiency_th']
    co2_factor = util_row['co2_factor_t_per_MWh_fuel']
    var_om = util_row.get('var_om_nzd_per_MWh_heat', 0.0)
    
    # Determine fuel price based on fuel type
    fuel_type = util_row['fuel']
    if fuel_type in ['lignite', 'coal']:
        fuel_price = signals['coal_price_nzd_per_MWh_fuel']
    elif fuel_type == 'biomass':
        fuel_price = signals['biomass_price_nzd_per_MWh_fuel']
    else:
        fuel_price = 0.0  # Unknown fuel type
    
    ets_price = signals['ets_price_nzd_per_tCO2']
    
    # Marginal cost calculation
    mc_heat = (fuel_price / efficiency) + (ets_price * co2_factor / efficiency) + var_om
    
    return mc_heat


def allocate_dispatch_optimal_subset(
    demand_df: pd.DataFrame,
    util_df: pd.DataFrame,
    signals: Dict[str, float],
    commitment_block_hours: int = 168,
    unserved_penalty_nzd_per_MWh: float = 50000.0,
    reserve_frac: float = 0.0,
    reserve_penalty_nzd_per_MWh: float = 0.0,
    no_load_cost_nzd_per_h: float = 50.0,
    online_cost_applies_when: str = 'online_only'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Allocate dispatch using optimal subset selection per commitment block.
    
    For each commitment block:
    1. Enumerate all subsets of available units
    2. For each subset, compute hourly dispatch and costs
    3. Choose subset with minimum total cost
    4. Enforce min up/down time constraints
    
    Args:
        demand_df: DataFrame with timestamp_utc and heat_demand_MW columns
        util_df: DataFrame with utility information
        signals: Signals dict from get_signals_for_epoch
        commitment_block_hours: Hours per commitment block (default 168 = weekly)
        unserved_penalty_nzd_per_MWh: Penalty for unserved demand
        
    Returns:
        Tuple of (long-form DataFrame, wide-form DataFrame)
        Long-form includes: timestamp_utc, unit_id, heat_MW, unit_on, fuel_MWh, co2_tonnes,
                           fuel_cost_nzd, carbon_cost_nzd, var_om_cost_nzd,
                           fixed_on_cost_nzd, startup_cost_nzd, total_cost_nzd, unserved_MW
    """
    import itertools
    
    # Filter to existing units only
    existing_units = util_df[util_df['status_2020'] == 'existing'].copy()
    unit_ids = existing_units['unit_id'].tolist()
    n_units = len(unit_ids)
    
    # Pre-compute marginal costs for all units
    mc_map = {}
    for _, util_row in existing_units.iterrows():
        mc_map[util_row['unit_id']] = compute_marginal_cost_heat(util_row, signals)
    
    # Sort units by marginal cost (for dispatch order)
    existing_units = existing_units.copy()
    existing_units['mc_heat'] = existing_units['unit_id'].map(mc_map)
    existing_units = existing_units.sort_values('mc_heat').reset_index(drop=True)
    
    # Split demand into blocks
    n_hours = len(demand_df)
    n_blocks = int(np.ceil(n_hours / commitment_block_hours))
    
    # Initialize results
    long_results = []
    unit_state = {unit_id: 0 for unit_id in unit_ids}  # 0=OFF, 1=ON
    unit_blocks_remaining = {unit_id: 0 for unit_id in unit_ids}  # Blocks remaining in current state
    
    # Process each block
    for block_idx in range(n_blocks):
        start_hour = block_idx * commitment_block_hours
        end_hour = min((block_idx + 1) * commitment_block_hours, n_hours)
        block_demand = demand_df.iloc[start_hour:end_hour].copy()
        block_hours = len(block_demand)
        
        # Update min up/down time constraints
        # Check if units can change state
        can_turn_on = {}
        can_turn_off = {}
        
        for unit_id in unit_ids:
            util_row = existing_units[existing_units['unit_id'] == unit_id].iloc[0]
            min_up_blocks = int(np.ceil(util_row['min_up_time_h'] / commitment_block_hours))
            min_down_blocks = int(np.ceil(util_row['min_down_time_h'] / commitment_block_hours))
            
            if unit_state[unit_id] == 1:  # Currently ON
                can_turn_off[unit_id] = (unit_blocks_remaining[unit_id] <= 0)
                can_turn_on[unit_id] = True
            else:  # Currently OFF
                can_turn_on[unit_id] = (unit_blocks_remaining[unit_id] <= 0)
                can_turn_off[unit_id] = True
        
        # Enumerate all possible subsets (power set, but respect min up/down)
        best_subset = None
        best_cost = float('inf')
        best_block_results = None
        
        # Generate candidate subsets
        # Start with all units that can be ON
        available_units = [uid for uid in unit_ids if can_turn_on.get(uid, True)]
        
        # Try all subsets of available units (but respect constraints)
        for subset_size in range(len(available_units) + 1):
            for subset in itertools.combinations(available_units, subset_size):
                # Check if this subset respects min up/down constraints
                valid_subset = True
                for unit_id in unit_ids:
                    if unit_id in subset and unit_state[unit_id] == 0:
                        if not can_turn_on.get(unit_id, False):
                            valid_subset = False
                            break
                    elif unit_id not in subset and unit_state[unit_id] == 1:
                        if not can_turn_off.get(unit_id, False):
                            valid_subset = False
                            break
                
                if not valid_subset:
                    continue
                
                # Try this subset
                subset_units = list(subset)
                block_results = []
                block_total_cost = 0.0
                
                # Compute startup costs (units turning ON)
                startup_cost = 0.0
                for unit_id in subset_units:
                    if unit_state[unit_id] == 0:
                        util_row = existing_units[existing_units['unit_id'] == unit_id].iloc[0]
                        startup_cost += util_row['startup_cost_nzd']
                
                # Dispatch each hour in the block
                subset_infeasible = False
                for hour_idx, hour_row in block_demand.iterrows():
                    timestamp = hour_row['timestamp_utc']
                    demand_MW = hour_row['heat_demand_MW']
                    
                    # Initialize unit outputs
                    unit_outputs = {uid: 0.0 for uid in unit_ids}
                    unserved_MW = 0.0
                    
                    if len(subset_units) == 0:
                        # Empty subset: all demand is unserved
                        unserved_MW = demand_MW
                    else:
                        # Compute effective capacities using availability
                        min_cap = 0.0
                        max_cap = 0.0
                        cap_map = {}
                        min_map = {}
                        
                        for unit_id in subset_units:
                            util_row = existing_units[existing_units['unit_id'] == unit_id].iloc[0]
                            cap_u = util_row['max_heat_MW'] * util_row['availability_factor']
                            min_u = util_row['min_load_frac'] * cap_u
                            cap_map[unit_id] = cap_u
                            min_map[unit_id] = min_u
                            min_cap += min_u
                            max_cap += cap_u
                        
                        # Feasibility check: demand < min_cap (oversupply)
                        if demand_MW < min_cap - 1e-6:
                            # This subset is infeasible for this hour (oversupply)
                            subset_infeasible = True
                            break
                        
                        # Feasibility check: demand > max_cap (undersupply)
                        if demand_MW > max_cap + 1e-6:
                            # Dispatch all units at capacity, penalize unserved
                            for unit_id in subset_units:
                                unit_outputs[unit_id] = cap_map[unit_id]
                            unserved_MW = demand_MW - max_cap
                        else:
                            # Normal dispatch: set minimums first
                            total_min_output = 0.0
                            for unit_id in subset_units:
                                unit_outputs[unit_id] = min_map[unit_id]
                                total_min_output += min_map[unit_id]
                            
                            # Remaining demand to allocate
                            remaining_demand = max(0.0, demand_MW - total_min_output)
                            
                            # Allocate remaining demand to units in order of marginal cost
                            if remaining_demand > 0:
                                # Sort subset by marginal cost
                                subset_sorted = sorted(subset_units, key=lambda uid: mc_map[uid])
                                
                                for unit_id in subset_sorted:
                                    available_capacity = cap_map[unit_id] - unit_outputs[unit_id]
                                    
                                    if remaining_demand > 0 and available_capacity > 0:
                                        additional_output = min(remaining_demand, available_capacity)
                                        unit_outputs[unit_id] += additional_output
                                        remaining_demand -= additional_output
                            
                            # Check if demand is fully met (should be, but verify)
                            total_output = sum(unit_outputs.values())
                            if total_output < demand_MW - 1e-6:
                                unserved_MW = demand_MW - total_output
                    
                    # If subset is infeasible, break and skip this subset
                    if subset_infeasible:
                        block_total_cost = float('inf')
                        break
                    
                    # Compute reserve shortfall penalty
                    # Reserve requirement uses online derated capacity; dispatch can be below capacity
                    reserve_shortfall_MW = 0.0
                    if len(subset_units) > 0:
                        # Compute online capacity (derated) - sum of online units only
                        online_cap_MW = 0.0
                        for unit_id in subset_units:
                            util_row = existing_units[existing_units['unit_id'] == unit_id].iloc[0]
                            cap_u = util_row['max_heat_MW'] * util_row['availability_factor']
                            online_cap_MW += cap_u
                        
                        # Total dispatch across all units (may include units not in subset)
                        total_dispatch_MW = sum(unit_outputs.values())
                        
                        # Headroom = online capacity minus total dispatch
                        headroom_MW = online_cap_MW - total_dispatch_MW
                        
                        # Reserve requirement
                        reserve_req_MW = reserve_frac * demand_MW
                        
                        # Reserve shortfall = max(0, reserve_req - headroom)
                        reserve_shortfall_MW = max(0.0, reserve_req_MW - headroom_MW)
                    else:
                        # Empty subset: no reserve available, all reserve requirement is shortfall
                        reserve_req_MW = reserve_frac * demand_MW
                        reserve_shortfall_MW = reserve_req_MW
                    
                    # Compute costs for this hour
                    hour_fuel_cost = 0.0
                    hour_carbon_cost = 0.0
                    hour_var_om_cost = 0.0
                    hour_fixed_on_cost = 0.0
                    
                    for unit_id in subset_units:
                        util_row = existing_units[existing_units['unit_id'] == unit_id].iloc[0]
                        heat_MW = unit_outputs[unit_id]
                        
                        if heat_MW > 0:
                            efficiency = util_row['efficiency_th']
                            co2_factor = util_row['co2_factor_t_per_MWh_fuel']
                            var_om = util_row.get('var_om_nzd_per_MWh_heat', 0.0)
                            fixed_on = util_row.get('fixed_on_cost_nzd_per_h', 0.0)
                            
                            fuel_MWh = heat_MW / efficiency
                            co2_tonnes = fuel_MWh * co2_factor
                            
                            # Fuel cost
                            fuel_type = util_row['fuel']
                            if fuel_type in ['lignite', 'coal']:
                                fuel_price = signals['coal_price_nzd_per_MWh_fuel']
                            elif fuel_type == 'biomass':
                                fuel_price = signals['biomass_price_nzd_per_MWh_fuel']
                            else:
                                fuel_price = 0.0
                            
                            hour_fuel_cost += fuel_MWh * fuel_price
                            hour_carbon_cost += co2_tonnes * signals['ets_price_nzd_per_tCO2']
                            hour_var_om_cost += heat_MW * var_om
                            hour_fixed_on_cost += fixed_on
                    
                    # Unserved penalty
                    unserved_cost = unserved_MW * unserved_penalty_nzd_per_MWh
                    
                    # Reserve shortfall penalty (per MWh)
                    reserve_penalty_cost = reserve_shortfall_MW * reserve_penalty_nzd_per_MWh
                    
                    # Compute no-load/hot-standby cost
                    # Count online units (in subset)
                    n_online_units = len(subset_units)
                    if online_cost_applies_when == 'firing_only':
                        # Only count units that are actually firing (heat_MW > 0)
                        n_online_units = sum(1 for uid in subset_units if unit_outputs.get(uid, 0.0) > 0)
                    online_cost = no_load_cost_nzd_per_h * n_online_units
                    
                    # Store results for this hour (for all units)
                    for unit_id in unit_ids:
                        util_row = existing_units[existing_units['unit_id'] == unit_id].iloc[0]
                        heat_MW = unit_outputs.get(unit_id, 0.0)
                        
                        # unit_online: 1 if unit is in selected subset (can be online with heat_MW = 0)
                        unit_online = 1 if unit_id in subset_units else 0
                        
                        # unit_on: 1 if unit is firing (heat_MW > 0), else 0
                        unit_on = 1 if heat_MW > 0 else 0
                        
                        # Compute fuel and emissions
                        efficiency = util_row['efficiency_th']
                        co2_factor = util_row['co2_factor_t_per_MWh_fuel']
                        fuel_MWh = heat_MW / efficiency if efficiency > 0 and heat_MW > 0 else 0.0
                        co2_tonnes = fuel_MWh * co2_factor
                        
                        # Compute unit-specific costs
                        if unit_on and heat_MW > 0:
                            var_om = util_row.get('var_om_nzd_per_MWh_heat', 0.0)
                            fixed_on = util_row.get('fixed_on_cost_nzd_per_h', 0.0)
                            
                            fuel_type = util_row['fuel']
                            if fuel_type in ['lignite', 'coal']:
                                fuel_price = signals['coal_price_nzd_per_MWh_fuel']
                            elif fuel_type == 'biomass':
                                fuel_price = signals['biomass_price_nzd_per_MWh_fuel']
                            else:
                                fuel_price = 0.0
                            
                            unit_fuel_cost = fuel_MWh * fuel_price
                            unit_carbon_cost = co2_tonnes * signals['ets_price_nzd_per_tCO2']
                            unit_var_om_cost = heat_MW * var_om
                            unit_fixed_on_cost = fixed_on
                        else:
                            unit_fuel_cost = 0.0
                            unit_carbon_cost = 0.0
                            unit_var_om_cost = 0.0
                            unit_fixed_on_cost = 0.0
                        
                        # Startup cost allocation (only for units turning ON)
                        unit_startup_cost = 0.0
                        if unit_on and unit_state[unit_id] == 0:
                            # Allocate startup cost across block hours
                            unit_startup_cost = startup_cost / block_hours if len(subset_units) > 0 else 0.0
                        
                        # Store unserved_MW, reserve_shortfall_MW, and online_cost_nzd only once per hour (on first unit)
                        unserved_value = unserved_MW if unit_id == unit_ids[0] else 0.0
                        reserve_shortfall_value = reserve_shortfall_MW if unit_id == unit_ids[0] else 0.0
                        reserve_penalty_value = reserve_penalty_cost if unit_id == unit_ids[0] else 0.0
                        unserved_cost_value = unserved_cost if unit_id == unit_ids[0] else 0.0
                        online_cost_value = online_cost if unit_id == unit_ids[0] else 0.0
                        
                        # Always append results for all units
                        block_results.append({
                            'timestamp_utc': timestamp,
                            'unit_id': unit_id,
                            'heat_MW': heat_MW,
                            'unit_online': unit_online,  # 1 if in subset (can be online with heat_MW = 0)
                            'unit_on': unit_on,  # 1 if firing (heat_MW > 0), else 0
                            'fuel_MWh': fuel_MWh,
                            'co2_tonnes': co2_tonnes,
                            'fuel_cost_nzd': unit_fuel_cost,
                            'carbon_cost_nzd': unit_carbon_cost,
                            'var_om_cost_nzd': unit_var_om_cost,
                            'fixed_on_cost_nzd': unit_fixed_on_cost,
                            'startup_cost_nzd': unit_startup_cost,
                            'unserved_MW': unserved_value,
                            'unserved_cost_nzd': unserved_cost_value,
                            'reserve_shortfall_MW': reserve_shortfall_value,
                            'reserve_penalty_cost_nzd': reserve_penalty_value,
                            'online_cost_nzd': online_cost_value,
                        })
                    
                    # Accumulate costs (sum across all units for this hour)
                    hour_total = hour_fuel_cost + hour_carbon_cost + hour_var_om_cost + hour_fixed_on_cost + unserved_cost + reserve_penalty_cost + online_cost
                    block_total_cost += hour_total
                
                # Skip infeasible subsets
                if subset_infeasible:
                    continue
                
                # Add startup cost once per block
                if len(subset_units) > 0:
                    block_total_cost += startup_cost
                
                # Check if this is the best subset so far
                if block_total_cost < best_cost:
                    best_cost = block_total_cost
                    best_subset = subset_units
                    best_block_results = block_results
        
        # Apply best subset for this block
        if best_block_results is None:
            # Fallback: use all units if no valid subset found
            best_subset = available_units
            print(f"[WARNING] Block {block_idx}: No valid subset found, using all available units")
        
        # Update unit states
        for unit_id in unit_ids:
            if unit_id in best_subset:
                if unit_state[unit_id] == 0:
                    # Unit turning ON
                    util_row = existing_units[existing_units['unit_id'] == unit_id].iloc[0]
                    min_up_blocks = int(np.ceil(util_row['min_up_time_h'] / commitment_block_hours))
                    unit_blocks_remaining[unit_id] = min_up_blocks
                else:
                    # Unit staying ON, decrement counter
                    unit_blocks_remaining[unit_id] = max(0, unit_blocks_remaining[unit_id] - 1)
                unit_state[unit_id] = 1
            else:
                if unit_state[unit_id] == 1:
                    # Unit turning OFF
                    util_row = existing_units[existing_units['unit_id'] == unit_id].iloc[0]
                    min_down_blocks = int(np.ceil(util_row['min_down_time_h'] / commitment_block_hours))
                    unit_blocks_remaining[unit_id] = min_down_blocks
                else:
                    # Unit staying OFF, decrement counter
                    unit_blocks_remaining[unit_id] = max(0, unit_blocks_remaining[unit_id] - 1)
                unit_state[unit_id] = 0
        
        # Add block results to overall results
        if best_block_results:
            long_results.extend(best_block_results)
    
    # Create DataFrame
    if not long_results:
        raise ValueError("No dispatch results generated - check demand and utilities")
    
    dispatch_long = pd.DataFrame(long_results)
    
    # Recompute costs properly (the above was approximate for subset selection)
    # Now compute exact costs based on final dispatch
    dispatch_long = recompute_costs_optimal(dispatch_long, existing_units, signals, commitment_block_hours,
                                           unserved_penalty_nzd_per_MWh=unserved_penalty_nzd_per_MWh,
                                           reserve_penalty_nzd_per_MWh=reserve_penalty_nzd_per_MWh,
                                           no_load_cost_nzd_per_h=no_load_cost_nzd_per_h,
                                           online_cost_applies_when=online_cost_applies_when)
    
    # Create wide-form pivot
    dispatch_wide = dispatch_long.pivot_table(
        index='timestamp_utc',
        columns='unit_id',
        values='heat_MW',
        aggfunc='sum'
    ).reset_index()
    
    # Rename unit columns
    unit_columns = [col for col in dispatch_wide.columns if col != 'timestamp_utc']
    rename_dict = {col: f"{col}_MW" for col in unit_columns}
    dispatch_wide = dispatch_wide.rename(columns=rename_dict)
    
    # Add total_heat_MW
    unit_cols = [col for col in dispatch_wide.columns if col.endswith('_MW') and col != 'total_heat_MW']
    dispatch_wide['total_heat_MW'] = dispatch_wide[unit_cols].sum(axis=1)
    
    # Reorder columns
    unit_cols_sorted = sorted(unit_cols)
    dispatch_wide = dispatch_wide[['timestamp_utc', 'total_heat_MW'] + unit_cols_sorted]
    
    # Energy closure sanity check
    # Parse timestamp_utc if needed for grouping
    dispatch_long_for_closure = dispatch_long.copy()
    if not pd.api.types.is_datetime64_any_dtype(dispatch_long_for_closure['timestamp_utc']):
        dispatch_long_for_closure['timestamp_utc'] = parse_any_timestamp(dispatch_long_for_closure['timestamp_utc'])
    annual_demand_GWh = demand_df['heat_demand_MW'].sum() / 1000.0
    annual_dispatch_GWh = dispatch_wide['total_heat_MW'].sum() / 1000.0
    annual_unserved_GWh = dispatch_long_for_closure.groupby('timestamp_utc')['unserved_MW'].first().sum() / 1000.0 if 'unserved_MW' in dispatch_long_for_closure.columns else 0.0
    
    print(f"\n[Energy Closure Check]")
    print(f"  Annual demand:    {annual_demand_GWh:.6f} GWh")
    print(f"  Annual served:    {annual_dispatch_GWh:.6f} GWh")
    print(f"  Annual unserved:  {annual_unserved_GWh:.6f} GWh")
    print(f"  Difference:       {abs(annual_demand_GWh - annual_dispatch_GWh - annual_unserved_GWh):.6f} GWh")
    
    # Assert closure (allow small tolerance)
    closure_diff = abs(annual_demand_GWh - annual_dispatch_GWh - annual_unserved_GWh)
    if closure_diff > 0.1:
        print(f"[WARNING] Energy closure mismatch > 0.1 GWh (diff = {closure_diff:.6f} GWh)")
    else:
        print(f"[OK] Energy closure verified (diff = {closure_diff:.6f} GWh)")
    
    return dispatch_long, dispatch_wide


def recompute_costs_optimal(dispatch_long: pd.DataFrame, util_df: pd.DataFrame,
                            signals: Dict[str, float], commitment_block_hours: int,
                            unserved_penalty_nzd_per_MWh: float = 50000.0,
                            reserve_penalty_nzd_per_MWh: float = 2000.0,
                            no_load_cost_nzd_per_h: float = 50.0,
                            online_cost_applies_when: str = 'online_only') -> pd.DataFrame:
    """
    Recompute costs for optimal dispatch with proper allocation.
    
    Args:
        dispatch_long: Long-form dispatch DataFrame
        util_df: Utilities DataFrame
        signals: Signals dict
        commitment_block_hours: Block size for startup cost allocation
        unserved_penalty_nzd_per_MWh: Penalty for unserved demand
        reserve_penalty_nzd_per_MWh: Penalty for reserve shortfall
        
    Returns:
        DataFrame with properly computed costs
    """
    dispatch_long = dispatch_long.copy()
    
    # Initialize cost columns
    dispatch_long['fuel_cost_nzd'] = 0.0
    dispatch_long['carbon_cost_nzd'] = 0.0
    dispatch_long['var_om_cost_nzd'] = 0.0
    dispatch_long['fixed_on_cost_nzd'] = 0.0
    dispatch_long['startup_cost_nzd'] = 0.0
    
    # Ensure unserved and reserve columns exist
    if 'unserved_MW' not in dispatch_long.columns:
        dispatch_long['unserved_MW'] = 0.0
    if 'unserved_cost_nzd' not in dispatch_long.columns:
        dispatch_long['unserved_cost_nzd'] = 0.0
    if 'reserve_shortfall_MW' not in dispatch_long.columns:
        dispatch_long['reserve_shortfall_MW'] = 0.0
    if 'reserve_penalty_cost_nzd' not in dispatch_long.columns:
        dispatch_long['reserve_penalty_cost_nzd'] = 0.0
    
    # Merge unit info
    for _, util_row in util_df.iterrows():
        unit_id = util_row['unit_id']
        mask = dispatch_long['unit_id'] == unit_id
        
        efficiency = util_row['efficiency_th']
        co2_factor = util_row['co2_factor_t_per_MWh_fuel']
        var_om = util_row.get('var_om_nzd_per_MWh_heat', 0.0)
        fixed_on = util_row.get('fixed_on_cost_nzd_per_h', 0.0)
        startup_cost = util_row.get('startup_cost_nzd', 0.0)
        
        # Fuel type
        fuel_type = util_row['fuel']
        if fuel_type in ['lignite', 'coal']:
            fuel_price = signals['coal_price_nzd_per_MWh_fuel']
        elif fuel_type == 'biomass':
            fuel_price = signals['biomass_price_nzd_per_MWh_fuel']
        else:
            fuel_price = 0.0
        
        ets_price = signals['ets_price_nzd_per_tCO2']
        
        # Compute costs for this unit
        unit_data = dispatch_long[mask].copy()
        unit_data['fuel_MWh'] = unit_data['heat_MW'] / efficiency
        unit_data['co2_tonnes'] = unit_data['fuel_MWh'] * co2_factor
        
        dispatch_long.loc[mask, 'fuel_cost_nzd'] = unit_data['fuel_MWh'] * fuel_price
        dispatch_long.loc[mask, 'carbon_cost_nzd'] = unit_data['co2_tonnes'] * ets_price
        dispatch_long.loc[mask, 'var_om_cost_nzd'] = unit_data['heat_MW'] * var_om
        dispatch_long.loc[mask, 'fixed_on_cost_nzd'] = unit_data['unit_on'] * fixed_on
        
        # Unserved cost (if not already computed)
        if 'unserved_cost_nzd' in dispatch_long.columns:
            # Recompute unserved cost from unserved_MW
            unserved_mask = mask & (dispatch_long['unserved_MW'] > 0)
            dispatch_long.loc[unserved_mask, 'unserved_cost_nzd'] = (
                dispatch_long.loc[unserved_mask, 'unserved_MW'] * unserved_penalty_nzd_per_MWh
            )
        
        # Reserve penalty cost (if not already computed)
        if 'reserve_penalty_cost_nzd' in dispatch_long.columns:
            # Recompute reserve penalty from reserve_shortfall_MW
            reserve_mask = mask & (dispatch_long['reserve_shortfall_MW'] > 0)
            dispatch_long.loc[reserve_mask, 'reserve_penalty_cost_nzd'] = (
                dispatch_long.loc[reserve_mask, 'reserve_shortfall_MW'] * reserve_penalty_nzd_per_MWh
            )
        
        # Startup costs: detect transitions OFF->ON
        unit_data = unit_data.sort_values('timestamp_utc')
        unit_data['prev_unit_on'] = unit_data['unit_on'].shift(1, fill_value=0)
        startup_mask = (unit_data['unit_on'] == 1) & (unit_data['prev_unit_on'] == 0)
        
        # Allocate startup cost across block hours
        startup_hours = unit_data[startup_mask]
        if len(startup_hours) > 0:
            for _, startup_row in startup_hours.iterrows():
                # Find block containing this hour
                timestamp = startup_row['timestamp_utc']
                # Calculate block start: round down to nearest block boundary
                # Use UTC timestamp for 2020-01-01
                year_start = pd.Timestamp('2020-01-01', tz='UTC')
                hours_since_start = (timestamp - year_start).total_seconds() / 3600
                block_idx = int(hours_since_start // commitment_block_hours)
                block_start = year_start + pd.Timedelta(hours=block_idx * commitment_block_hours)
                block_end = block_start + pd.Timedelta(hours=commitment_block_hours)
                block_mask = (dispatch_long['unit_id'] == unit_id) & \
                            (dispatch_long['timestamp_utc'] >= block_start) & \
                            (dispatch_long['timestamp_utc'] < block_end) & \
                            (dispatch_long['unit_on'] == 1)
                n_block_hours = block_mask.sum()
                if n_block_hours > 0:
                    dispatch_long.loc[block_mask, 'startup_cost_nzd'] = startup_cost / n_block_hours
    
    # Recompute online cost if needed (store once per hour)
    if 'online_cost_nzd' in dispatch_long.columns:
        # Group by timestamp_utc and compute online cost once per hour
        for timestamp in dispatch_long['timestamp_utc'].unique():
            hour_mask = dispatch_long['timestamp_utc'] == timestamp
            # Count online units for this hour
            n_online = dispatch_long.loc[hour_mask, 'unit_online'].sum() if 'unit_online' in dispatch_long.columns else 0
            if n_online > 0:
                # Store on first unit only
                first_unit_id = dispatch_long.loc[hour_mask, 'unit_id'].iloc[0]
                first_unit_mask = hour_mask & (dispatch_long['unit_id'] == first_unit_id)
                dispatch_long.loc[first_unit_mask, 'online_cost_nzd'] = n_online * no_load_cost_nzd_per_h
    
    # Total cost (includes all components)
    cost_components = ['fuel_cost_nzd', 'carbon_cost_nzd', 'var_om_cost_nzd', 
                      'fixed_on_cost_nzd', 'startup_cost_nzd']
    if 'unserved_cost_nzd' in dispatch_long.columns:
        cost_components.append('unserved_cost_nzd')
    if 'reserve_penalty_cost_nzd' in dispatch_long.columns:
        cost_components.append('reserve_penalty_cost_nzd')
    if 'online_cost_nzd' in dispatch_long.columns:
        cost_components.append('online_cost_nzd')
    
    dispatch_long['total_cost_nzd'] = dispatch_long[cost_components].sum(axis=1)
    
    return dispatch_long


def add_costs_to_dispatch(dispatch_long: pd.DataFrame, util_df: pd.DataFrame, 
                          signals: Dict[str, float]) -> pd.DataFrame:
    """
    Add cost columns to dispatch DataFrame based on fuel type and signals.
    
    For 2020, all units are coal-fired. Future versions can branch on fuel_type.
    
    Args:
        dispatch_long: Long-form dispatch DataFrame with heat_MW, fuel_MWh, co2_tonnes
        util_df: Utilities DataFrame with unit_id and fuel columns
        signals: Signals dict from get_signals_for_epoch
        
    Returns:
        DataFrame with added columns: fuel_cost_nzd, carbon_cost_nzd, total_cost_nzd
    """
    dispatch_long = dispatch_long.copy()
    
    # Merge fuel type from utilities
    fuel_map = dict(zip(util_df['unit_id'], util_df['fuel']))
    dispatch_long['fuel_type'] = dispatch_long['unit_id'].map(fuel_map)
    
    # Initialize cost columns
    dispatch_long['fuel_cost_nzd'] = 0.0
    dispatch_long['carbon_cost_nzd'] = 0.0
    
    # Get ETS price (same for all units)
    ets_price = signals['ets_price_nzd_per_tCO2']
    
    # Process by fuel type
    for fuel_type in dispatch_long['fuel_type'].unique():
        mask = dispatch_long['fuel_type'] == fuel_type
        
        if fuel_type == 'lignite' or fuel_type == 'coal':
            # Use coal price and ETS
            coal_price = signals['coal_price_nzd_per_MWh_fuel']
            dispatch_long.loc[mask, 'fuel_cost_nzd'] = (
                dispatch_long.loc[mask, 'fuel_MWh'] * coal_price
            )
            dispatch_long.loc[mask, 'carbon_cost_nzd'] = (
                dispatch_long.loc[mask, 'co2_tonnes'] * ets_price
            )
        elif fuel_type == 'biomass':
            # Use biomass price and minimal ETS (biomass has low emissions factor)
            biomass_price = signals['biomass_price_nzd_per_MWh_fuel']
            dispatch_long.loc[mask, 'fuel_cost_nzd'] = (
                dispatch_long.loc[mask, 'fuel_MWh'] * biomass_price
            )
            # Biomass CO2 is mostly biogenic, but may still have small ETS cost
            # For now, apply ETS to the small non-biogenic portion
            dispatch_long.loc[mask, 'carbon_cost_nzd'] = (
                dispatch_long.loc[mask, 'co2_tonnes'] * ets_price
            )
        else:
            # Unknown fuel type - warn but don't fail
            print(f"[WARNING] Unknown fuel type '{fuel_type}', costs set to zero")
    
    # Total cost (includes all components)
    cost_components = ['fuel_cost_nzd', 'carbon_cost_nzd']
    if 'var_om_cost_nzd' in dispatch_long.columns:
        cost_components.append('var_om_cost_nzd')
    if 'fixed_on_cost_nzd' in dispatch_long.columns:
        cost_components.append('fixed_on_cost_nzd')
    if 'startup_cost_nzd' in dispatch_long.columns:
        cost_components.append('startup_cost_nzd')
    if 'unserved_cost_nzd' in dispatch_long.columns:
        cost_components.append('unserved_cost_nzd')
    if 'reserve_penalty_cost_nzd' in dispatch_long.columns:
        cost_components.append('reserve_penalty_cost_nzd')
    
    dispatch_long['total_cost_nzd'] = dispatch_long[cost_components].sum(axis=1)
    
    # Drop fuel_type column (it was just for internal use)
    dispatch_long = dispatch_long.drop(columns=['fuel_type'])
    
    return dispatch_long


def aggregate_system_costs(dispatch_long: pd.DataFrame) -> dict:
    """
    Aggregate system-level costs (stored once per hour on first unit row).
    These must NOT be summed by unit to avoid attributing all penalties to CB1.
    
    Returns dict with system cost totals.
    """
    system_costs = {}
    # System penalties are stored once per hour (on first unit row), so aggregate by timestamp_utc first
    if 'unserved_MW' in dispatch_long.columns:
        system_costs['unserved_MW'] = dispatch_long.groupby('timestamp_utc')['unserved_MW'].first().sum()
    if 'unserved_cost_nzd' in dispatch_long.columns:
        system_costs['unserved_cost_nzd'] = dispatch_long.groupby('timestamp_utc')['unserved_cost_nzd'].first().sum()
    if 'reserve_shortfall_MW' in dispatch_long.columns:
        system_costs['reserve_shortfall_MW'] = dispatch_long.groupby('timestamp_utc')['reserve_shortfall_MW'].first().sum()
    if 'reserve_penalty_cost_nzd' in dispatch_long.columns:
        system_costs['reserve_penalty_cost_nzd'] = dispatch_long.groupby('timestamp_utc')['reserve_penalty_cost_nzd'].first().sum()
    if 'online_cost_nzd' in dispatch_long.columns:
        system_costs['online_cost_nzd'] = dispatch_long.groupby('timestamp_utc')['online_cost_nzd'].first().sum()
    return system_costs


def compute_annual_summary(dispatch_long: pd.DataFrame, reserve_frac: float = 0.0) -> pd.DataFrame:
    """
    Compute annual totals and key indicators by unit, plus SYSTEM and TOTAL rows.
    
    System-level costs (unserved, reserve, online) are stored once per hour on the
    first unit row to avoid multiplication. They are aggregated separately and appear
    in a SYSTEM row, not attributed to individual units.
    
    Args:
        dispatch_long: Long-form dispatch DataFrame with all cost columns
        
    Returns:
        DataFrame with one row per unit_id, one SYSTEM row, and one TOTAL row
    """
    # Initialize penalty variables (always defined, even when reserve_frac=0)
    unserved_energy_mwh = 0.0
    unserved_penalty_cost_nzd = 0.0
    reserve_shortfall_mwh = 0.0
    reserve_penalty_cost_nzd = 0.0
    total_system_penalties_nzd = 0.0
    # Aggregate by unit - EXCLUDE system penalties (they're stored once per hour, not per unit)
    # Note: total_cost_nzd in hourly data includes system penalties, so we'll recompute it
    # as sum of unit-level costs only
    agg_dict = {
        'heat_MW': 'sum',
        'fuel_MWh': 'sum',
        'co2_tonnes': 'sum',
        'fuel_cost_nzd': 'sum',
        'carbon_cost_nzd': 'sum',
    }
    
    # Add optimal mode unit-level columns if present
    if 'var_om_cost_nzd' in dispatch_long.columns:
        agg_dict['var_om_cost_nzd'] = 'sum'
    if 'fixed_on_cost_nzd' in dispatch_long.columns:
        agg_dict['fixed_on_cost_nzd'] = 'sum'
    if 'startup_cost_nzd' in dispatch_long.columns:
        agg_dict['startup_cost_nzd'] = 'sum'
    # NOTE: unserved_MW, unserved_cost_nzd, reserve_shortfall_MW, reserve_penalty_cost_nzd,
    # online_cost_nzd are system-level and stored once per hour - do NOT include in unit aggregation
    
    annual_by_unit = dispatch_long.groupby('unit_id').agg(agg_dict)
    
    # Recompute unit-level total_cost_nzd (excludes system penalties)
    annual_by_unit['total_cost_nzd'] = (
        annual_by_unit['fuel_cost_nzd'] +
        annual_by_unit['carbon_cost_nzd']
    )
    if 'var_om_cost_nzd' in annual_by_unit.columns:
        annual_by_unit['total_cost_nzd'] += annual_by_unit['var_om_cost_nzd']
    if 'fixed_on_cost_nzd' in annual_by_unit.columns:
        annual_by_unit['total_cost_nzd'] += annual_by_unit['fixed_on_cost_nzd']
    if 'startup_cost_nzd' in annual_by_unit.columns:
        annual_by_unit['total_cost_nzd'] += annual_by_unit['startup_cost_nzd']
    
    # Convert MW to GWh
    annual_by_unit['annual_heat_GWh'] = annual_by_unit['heat_MW'] / 1000.0
    annual_by_unit['annual_fuel_MWh'] = annual_by_unit['fuel_MWh']
    annual_by_unit['annual_co2_tonnes'] = annual_by_unit['co2_tonnes']
    annual_by_unit['annual_fuel_cost_nzd'] = annual_by_unit['fuel_cost_nzd']
    annual_by_unit['annual_carbon_cost_nzd'] = annual_by_unit['carbon_cost_nzd']
    annual_by_unit['annual_total_cost_nzd'] = annual_by_unit['total_cost_nzd']
    
    # Compute average cost per MWh_heat (safe when annual heat is zero)
    # avg_cost = total_cost / (heat_GWh * 1000) to get cost per MWh
    annual_by_unit['avg_cost_nzd_per_MWh_heat'] = (
        annual_by_unit['annual_total_cost_nzd'] / 
        (annual_by_unit['annual_heat_GWh'] * 1000.0).replace(0, np.nan)
    ).fillna(0.0)
    
    # Add annual diagnostics for optimal mode (before creating TOTAL row)
    total_dict_base = {}
    if 'reserve_shortfall_MW' in dispatch_long.columns:
        # Compute diagnostics across all hours (not per unit)
        # Parse timestamp_utc if needed for grouping
        dispatch_long_for_diag = dispatch_long.copy()
        if not pd.api.types.is_datetime64_any_dtype(dispatch_long_for_diag['timestamp_utc']):
            dispatch_long_for_diag['timestamp_utc'] = parse_any_timestamp(dispatch_long_for_diag['timestamp_utc'])
        hourly_reserve = dispatch_long_for_diag.groupby('timestamp_utc')['reserve_shortfall_MW'].first()
        hours_reserve_shortfall_gt_0 = (hourly_reserve > 0).sum()
        p95_reserve_shortfall_MW = hourly_reserve.quantile(0.95) if len(hourly_reserve) > 0 else 0.0
        
        # Average online units
        if 'unit_online' in dispatch_long_for_diag.columns:
            hourly_online = dispatch_long_for_diag.groupby('timestamp_utc')['unit_online'].sum()
            avg_online_units = hourly_online.mean()
            
            # Seasonal averages (simple month split: May-Aug low, Oct-Mar peak)
            dispatch_long_for_diag['month'] = dispatch_long_for_diag['timestamp_utc'].dt.month
            low_season_months = [5, 6, 7, 8]  # May-Aug
            peak_season_months = [10, 11, 12, 1, 2, 3]  # Oct-Mar
            
            low_season_mask = dispatch_long_for_diag['month'].isin(low_season_months)
            peak_season_mask = dispatch_long_for_diag['month'].isin(peak_season_months)
            
            low_season_online = dispatch_long_for_diag.loc[low_season_mask].groupby('timestamp_utc')['unit_online'].sum().mean() if low_season_mask.sum() > 0 else 0.0
            peak_season_online = dispatch_long_for_diag.loc[peak_season_mask].groupby('timestamp_utc')['unit_online'].sum().mean() if peak_season_mask.sum() > 0 else 0.0
        else:
            avg_online_units = np.nan
            low_season_online = np.nan
            peak_season_online = np.nan
        
        # Compute average online headroom (online capacity - dispatch)
        if 'unit_online' in dispatch_long_for_diag.columns and 'reserve_shortfall_MW' in dispatch_long_for_diag.columns:
            # Reconstruct headroom from reserve shortfall: headroom = online_cap - dispatch
            # reserve_shortfall = max(0, reserve_req - headroom)
            # So if reserve_shortfall > 0: headroom = reserve_req - reserve_shortfall
            # If reserve_shortfall = 0: headroom >= reserve_req (we don't know exact, but can estimate)
            # Use direct aggregation instead of apply() to avoid FutureWarning
            served_by_ts = dispatch_long_for_diag.groupby('timestamp_utc', sort=False)['heat_MW'].sum()
            hourly_reserve_req = reserve_frac * served_by_ts
            hourly_headroom_est = hourly_reserve_req - hourly_reserve
            # For hours with no shortfall, headroom is at least reserve_req, but we'll use the estimate
            avg_online_headroom_MW = hourly_headroom_est.mean()
        else:
            avg_online_headroom_MW = np.nan
        
        # Store for later addition to total_dict
        total_dict_base['hours_reserve_shortfall_gt_0'] = hours_reserve_shortfall_gt_0
        total_dict_base['p95_reserve_shortfall_MW'] = p95_reserve_shortfall_MW
        if 'unit_online' in dispatch_long.columns:
            total_dict_base['avg_online_units'] = avg_online_units
            total_dict_base['avg_online_units_low_season'] = low_season_online
            total_dict_base['avg_online_units_peak_season'] = peak_season_online
            total_dict_base['avg_online_headroom_MW'] = avg_online_headroom_MW
    
    # Set system penalty columns to 0 for all unit rows (they're system-level, not unit-level)
    system_penalty_cols = ['unserved_MW', 'unserved_cost_nzd', 'reserve_shortfall_MW', 
                          'reserve_penalty_cost_nzd', 'online_cost_nzd']
    for col in system_penalty_cols:
        if col in dispatch_long.columns:
            annual_by_unit[col] = 0.0
    
    # Add operational/penalty cost columns to unit rows (unit-level operational costs only)
    annual_by_unit['annual_operational_cost_nzd'] = annual_by_unit['annual_total_cost_nzd']  # Units have no penalties
    annual_by_unit['annual_penalty_cost_nzd'] = 0.0  # Units have no penalties
    annual_by_unit['annual_system_cost_nzd'] = 0.0  # Units have no system costs
    annual_by_unit['avg_operational_cost_nzd_per_MWh_heat'] = annual_by_unit['avg_cost_nzd_per_MWh_heat']
    
    # Aggregate system-level costs separately (stored once per hour, not per unit)
    # Note: parse timestamp_utc if it's a string for grouping
    dispatch_long_for_agg = dispatch_long.copy()
    if not pd.api.types.is_datetime64_any_dtype(dispatch_long_for_agg['timestamp_utc']):
        dispatch_long_for_agg['timestamp_utc'] = parse_any_timestamp(dispatch_long_for_agg['timestamp_utc'])
    system_costs = aggregate_system_costs(dispatch_long_for_agg)
    
    # Extract penalty values from system_costs (always defined, defaults to 0.0)
    unserved_penalty_cost_nzd = system_costs.get('unserved_cost_nzd', 0.0)
    reserve_penalty_cost_nzd = system_costs.get('reserve_penalty_cost_nzd', 0.0)
    unserved_energy_mwh = system_costs.get('unserved_MW', 0.0)
    reserve_shortfall_mwh = system_costs.get('reserve_shortfall_MW', 0.0)
    
    # Compute total system penalties (unserved + reserve, excluding online which is operational)
    total_system_penalties_nzd = unserved_penalty_cost_nzd + reserve_penalty_cost_nzd
    
    # Create SYSTEM row for system penalties
    system_dict = {
        'heat_MW': 0.0,
        'fuel_MWh': 0.0,
        'co2_tonnes': 0.0,
        'fuel_cost_nzd': 0.0,
        'carbon_cost_nzd': 0.0,
        'total_cost_nzd': 0.0,
        'annual_heat_GWh': 0.0,
        'annual_fuel_MWh': 0.0,
        'annual_co2_tonnes': 0.0,
        'annual_fuel_cost_nzd': 0.0,
        'annual_carbon_cost_nzd': 0.0,
        'annual_total_cost_nzd': total_system_penalties_nzd,  # SYSTEM row = penalties only
        'annual_operational_cost_nzd': 0.0,  # SYSTEM has no operational costs
        'annual_penalty_cost_nzd': total_system_penalties_nzd,  # Only infeasibility penalties (unserved + reserve)
        'annual_system_cost_nzd': total_system_penalties_nzd + system_costs.get('online_cost_nzd', 0.0),  # Penalties + online
        'avg_cost_nzd_per_MWh_heat': 0.0,
        'avg_operational_cost_nzd_per_MWh_heat': 0.0,
    }
    # Add optimal mode unit columns (set to 0 for system row)
    if 'var_om_cost_nzd' in annual_by_unit.columns:
        system_dict['var_om_cost_nzd'] = 0.0
    if 'fixed_on_cost_nzd' in annual_by_unit.columns:
        system_dict['fixed_on_cost_nzd'] = 0.0
    if 'startup_cost_nzd' in annual_by_unit.columns:
        system_dict['startup_cost_nzd'] = 0.0
    # Add system penalty columns (unserved and reserve only; online_cost is operational, not in SYSTEM row)
    if 'unserved_MW' in system_costs:
        system_dict['unserved_MW'] = system_costs['unserved_MW']
    if 'unserved_cost_nzd' in system_costs:
        system_dict['unserved_cost_nzd'] = system_costs['unserved_cost_nzd']
    if 'reserve_shortfall_MW' in system_costs:
        system_dict['reserve_shortfall_MW'] = system_costs['reserve_shortfall_MW']
    if 'reserve_penalty_cost_nzd' in system_costs:
        system_dict['reserve_penalty_cost_nzd'] = system_costs['reserve_penalty_cost_nzd']
    # Note: online_cost_nzd is NOT added to SYSTEM row (it's operational, shown in TOTAL)
    # Add diagnostic metrics to system row (NaN for system row)
    if total_dict_base:
        for key in total_dict_base:
            system_dict[key] = np.nan
    
    system_row = pd.Series(system_dict, name='SYSTEM')
    
    # Compute three totals explicitly
    # Note: total_system_penalties_nzd already computed above before SYSTEM row creation
    total_units_cost_nzd = annual_by_unit['annual_total_cost_nzd'].sum()
    # Note: online_cost is operational (no-load/hot-standby), not a penalty
    total_operational_cost_nzd = total_units_cost_nzd + system_costs.get('online_cost_nzd', 0)
    grand_total_cost_nzd = total_operational_cost_nzd + total_system_penalties_nzd
    
    # Cost closure check
    assert abs(grand_total_cost_nzd - (total_operational_cost_nzd + total_system_penalties_nzd)) < 1e-6, \
        f"Cost closure failed: {grand_total_cost_nzd} != {total_operational_cost_nzd} + {total_system_penalties_nzd}"
    print(f"[OK] Cost closure: grand_total = operational + penalties ({grand_total_cost_nzd:,.2f} = {total_operational_cost_nzd:,.2f} + {total_system_penalties_nzd:,.2f})")
    
    # Create TOTAL row (unit costs + system costs)
    total_dict = {
        'heat_MW': annual_by_unit['heat_MW'].sum(),
        'fuel_MWh': annual_by_unit['fuel_MWh'].sum(),
        'co2_tonnes': annual_by_unit['co2_tonnes'].sum(),
        'fuel_cost_nzd': annual_by_unit['fuel_cost_nzd'].sum(),
        'carbon_cost_nzd': annual_by_unit['carbon_cost_nzd'].sum(),
        'total_cost_nzd': annual_by_unit['total_cost_nzd'].sum(),
        'annual_heat_GWh': annual_by_unit['annual_heat_GWh'].sum(),
        'annual_fuel_MWh': annual_by_unit['annual_fuel_MWh'].sum(),
        'annual_co2_tonnes': annual_by_unit['annual_co2_tonnes'].sum(),
        'annual_fuel_cost_nzd': annual_by_unit['annual_fuel_cost_nzd'].sum(),
        'annual_carbon_cost_nzd': annual_by_unit['annual_carbon_cost_nzd'].sum(),
        'annual_total_cost_nzd': grand_total_cost_nzd,  # Operational + penalties
        'annual_operational_cost_nzd': total_operational_cost_nzd,  # Units + online/no-load
        'annual_penalty_cost_nzd': total_system_penalties_nzd,  # Unserved + reserve penalties
        'annual_system_cost_nzd': total_system_penalties_nzd + system_costs.get('online_cost_nzd', 0.0),  # Penalties + online
        'avg_cost_nzd_per_MWh_heat': (
            grand_total_cost_nzd /
            (annual_by_unit['annual_heat_GWh'].sum() * 1000.0) if annual_by_unit['annual_heat_GWh'].sum() > 0 else 0.0
        ),  # Based on served heat (not demand)
        'avg_operational_cost_nzd_per_MWh_heat': (
            total_operational_cost_nzd /
            (annual_by_unit['annual_heat_GWh'].sum() * 1000.0) if annual_by_unit['annual_heat_GWh'].sum() > 0 else 0.0
        ),  # Operational cost per MWh served
    }
    
    # Add optimal mode unit totals
    if 'var_om_cost_nzd' in annual_by_unit.columns:
        total_dict['var_om_cost_nzd'] = annual_by_unit['var_om_cost_nzd'].sum()
    if 'fixed_on_cost_nzd' in annual_by_unit.columns:
        total_dict['fixed_on_cost_nzd'] = annual_by_unit['fixed_on_cost_nzd'].sum()
    if 'startup_cost_nzd' in annual_by_unit.columns:
        total_dict['startup_cost_nzd'] = annual_by_unit['startup_cost_nzd'].sum()
    
    # Add system costs to TOTAL
    total_dict.update(system_costs)
    
    # Add diagnostic metrics to total_dict
    if total_dict_base:
        total_dict.update(total_dict_base)
    
    total_row = pd.Series(total_dict, name='TOTAL')
    
    # Combine unit rows, SYSTEM row, and TOTAL row
    summary = pd.concat([annual_by_unit, system_row.to_frame().T, total_row.to_frame().T])
    
    # Ensure unit_id is a column (not index) for stable CSV schema
    summary = summary.reset_index()
    # Handle both 'index' and 'Unnamed: 0' column names
    if 'index' in summary.columns:
        summary = summary.rename(columns={'index': 'unit_id'})
    elif 'Unnamed: 0' in summary.columns:
        summary = summary.rename(columns={'Unnamed: 0': 'unit_id'})
    elif summary.index.name is None or summary.index.name == 'unit_id':
        # If index has no name or is unit_id, set name and reset
        summary.index.name = 'unit_id'
        summary = summary.reset_index()
    # If unit_id is already a column, ensure it exists
    if 'unit_id' not in summary.columns:
        # Last resort: create from index
        summary = summary.reset_index()
        if 'index' in summary.columns:
            summary = summary.rename(columns={'index': 'unit_id'})
        else:
            summary['unit_id'] = summary.index
    
    # Select and reorder columns for output
    output_cols = [
        'annual_heat_GWh',
        'annual_fuel_MWh',
        'annual_co2_tonnes',
        'annual_fuel_cost_nzd',
        'annual_carbon_cost_nzd',
    ]
    
    # Add optimal mode columns if present
    if 'var_om_cost_nzd' in summary.columns:
        output_cols.append('var_om_cost_nzd')
    if 'fixed_on_cost_nzd' in summary.columns:
        output_cols.append('fixed_on_cost_nzd')
    if 'startup_cost_nzd' in summary.columns:
        output_cols.append('startup_cost_nzd')
    
    output_cols.extend([
        'annual_total_cost_nzd',
        'annual_operational_cost_nzd',
        'annual_penalty_cost_nzd',
        'annual_system_cost_nzd',
        'avg_cost_nzd_per_MWh_heat',
        'avg_operational_cost_nzd_per_MWh_heat',
    ])
    
    if 'unserved_MW' in summary.columns:
        output_cols.append('unserved_MW')
    if 'unserved_cost_nzd' in summary.columns:
        output_cols.append('unserved_cost_nzd')
    if 'reserve_shortfall_MW' in summary.columns:
        output_cols.append('reserve_shortfall_MW')
    if 'reserve_penalty_cost_nzd' in summary.columns:
        output_cols.append('reserve_penalty_cost_nzd')
    if 'online_cost_nzd' in summary.columns:
        output_cols.append('online_cost_nzd')
    
    # Add diagnostic columns if present
    diagnostic_cols = [
        'hours_reserve_shortfall_gt_0',
        'p95_reserve_shortfall_MW',
        'avg_online_units',
        'avg_online_headroom_MW',
        'avg_online_units_peak_season',
        'avg_online_units_low_season'
    ]
    for col in diagnostic_cols:
        if col in summary.columns:
            output_cols.append(col)
    
    # Compute annual_penalty_cost_nzd for all rows (unserved + reserve)
    # Use .get() with defaults to handle missing columns
    for idx in summary.index:
        unserved = summary.loc[idx, 'unserved_cost_nzd'] if 'unserved_cost_nzd' in summary.columns else 0.0
        reserve = summary.loc[idx, 'reserve_penalty_cost_nzd'] if 'reserve_penalty_cost_nzd' in summary.columns else 0.0
        summary.loc[idx, 'annual_penalty_cost_nzd'] = unserved + reserve
    
    # Compute annual_system_cost_nzd for SYSTEM and TOTAL rows only (penalty + online)
    for idx in summary.index:
        unit_id = summary.loc[idx, 'unit_id'] if 'unit_id' in summary.columns else summary.index[idx]
        if unit_id in ['SYSTEM', 'TOTAL']:
            penalty = summary.loc[idx, 'annual_penalty_cost_nzd']
            online = summary.loc[idx, 'online_cost_nzd'] if 'online_cost_nzd' in summary.columns else 0.0
            summary.loc[idx, 'annual_system_cost_nzd'] = penalty + online
        else:
            # Unit rows: system cost = 0
            summary.loc[idx, 'annual_system_cost_nzd'] = 0.0
    
    # Ensure unit_id is first column
    final_cols = ['unit_id'] + [col for col in output_cols if col in summary.columns and col != 'unit_id']
    result = summary[final_cols].copy()
    
    # Assert unit_id column exists before returning
    assert 'unit_id' in result.columns, "unit_id column missing from annual summary"
    
    return result


def plot_dispatch_stack(dispatch_wide_path: str, output_path: str, demand_df: Optional[pd.DataFrame] = None):
    """
    Generate stacked area plot of unit dispatch over time.
    
    Args:
        dispatch_wide_path: Path to wide-form dispatch CSV
        output_path: Path to save the figure
        demand_df: Optional DataFrame with actual demand (timestamp, heat_demand_MW)
    """
    df = pd.read_csv(dispatch_wide_path)
    # Support both timestamp and timestamp_utc for backward compatibility
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = parse_any_timestamp(df['timestamp_utc'])
    elif 'timestamp' in df.columns:
        df['timestamp_utc'] = parse_any_timestamp(df['timestamp'])
    else:
        raise ValueError(f"CSV file {dispatch_wide_path} must have either 'timestamp' or 'timestamp_utc' column")
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Get unit columns (all columns ending in _MW except total_heat_MW)
    unit_cols = [col for col in df.columns if col.endswith('_MW') and col != 'total_heat_MW']
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create stacked area plot
    ax.stackplot(df['timestamp_utc'], 
                 [df[col] for col in unit_cols],
                 labels=[col.replace('_MW', '') for col in unit_cols],
                 alpha=0.7)
    
    # Overlay black line for total demand (visible above stack)
    if demand_df is not None and 'heat_demand_MW' in demand_df.columns:
        demand_df = demand_df.copy()
        # Support both timestamp and timestamp_utc
        if 'timestamp_utc' in demand_df.columns:
            demand_df['timestamp_utc'] = parse_any_timestamp(demand_df['timestamp_utc'])
        elif 'timestamp' in demand_df.columns:
            demand_df['timestamp_utc'] = parse_any_timestamp(demand_df['timestamp'])
        df_merged = df.merge(demand_df[['timestamp_utc', 'heat_demand_MW']], on='timestamp_utc', how='left')
        demand_values = df_merged['heat_demand_MW'].fillna(df['total_heat_MW'])
        ax.plot(df['timestamp_utc'], demand_values,
                color='black', linewidth=2.0, linestyle='-', label='Total demand', zorder=20)
        ymax_data = max(demand_values.max(), df[unit_cols].sum(axis=1).max() if unit_cols else 0)
    elif 'total_heat_MW' in df.columns:
        ax.plot(df['timestamp_utc'], df['total_heat_MW'],
                color='black', linewidth=2.0, linestyle='-', label='Total demand', zorder=20)
        ymax_data = max(df['total_heat_MW'].max(), df[unit_cols].sum(axis=1).max() if unit_cols else 0)
    else:
        ymax_data = df[unit_cols].sum(axis=1).max() if unit_cols else 0
    
    # Force y-axis: 140 MW minimum, or 1.05 * max rounded up
    ymax = max(140.0, np.ceil(1.05 * ymax_data))
    ax.set_ylim(0, ymax)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Heat Demand (MW)', fontsize=12)
    ax.set_title('Unit Dispatch Stack - 2020', fontsize=14, fontweight='bold')
    
    # Move legend outside to avoid overlap (never clipped)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0, ncol=1)
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    
    # Reserve space for legend (tuned to prevent clipping)
    fig.subplots_adjust(right=0.78)
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


def plot_units_online(dispatch_long_path: str, output_path: str):
    """
    Plot number of boilers online over time (daily or weekly mean).
    
    Args:
        dispatch_long_path: Path to long-form dispatch CSV with unit_on column
        output_path: Path to save the figure
    """
    df = pd.read_csv(dispatch_long_path)
    # Support both timestamp and timestamp_utc
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = parse_any_timestamp(df['timestamp_utc'])
    elif 'timestamp' in df.columns:
        df['timestamp_utc'] = parse_any_timestamp(df['timestamp'])
    else:
        raise ValueError(f"CSV file {dispatch_long_path} must have either 'timestamp' or 'timestamp_utc' column")
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Count units online per hour using unit_online (in subset, can be hot standby)
    # Use unit_online if available, otherwise fall back to unit_on for backwards compatibility
    if 'unit_online' in df.columns:
        online_col = 'unit_online'
        plot_label = 'Units Online (hot standby allowed)'
    else:
        online_col = 'unit_on'
        plot_label = 'Units Firing (heat_MW > 0)'
    
    units_online = df.groupby('timestamp_utc')[online_col].sum().reset_index()
    units_online.columns = ['timestamp_utc', 'n_units_online']
    
    # Compute daily mean
    units_online['date'] = units_online['timestamp_utc'].dt.date
    daily_mean = units_online.groupby('date')['n_units_online'].mean().reset_index()
    daily_mean['timestamp_utc'] = pd.to_datetime(daily_mean['date']).dt.tz_localize('UTC')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot daily mean
    ax.plot(daily_mean['timestamp_utc'], daily_mean['n_units_online'],
            color='darkblue', linewidth=1.5, label='Daily mean')
    
    ax.set_xlabel('Date (UTC)', fontsize=12)
    ax.set_ylabel('Number of Units Online', fontsize=12)
    ax.set_title(f'{plot_label} - 2020', fontsize=14, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


def plot_unit_utilisation_duration(dispatch_long_path: str, output_path: str, 
                                   utilities_csv: str = 'Input/site_utilities_2020.csv'):
    """
    Plot per-unit utilisation/load-duration style plot (one figure with all units).
    
    Args:
        dispatch_long_path: Path to long-form dispatch CSV
        output_path: Path to save the figure
        utilities_csv: Path to utilities CSV to get max_heat_MW and availability_factor
    """
    df = pd.read_csv(dispatch_long_path)
    # Support both timestamp and timestamp_utc
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = parse_any_timestamp(df['timestamp_utc'])
    elif 'timestamp' in df.columns:
        df['timestamp_utc'] = parse_any_timestamp(df['timestamp'])
    else:
        raise ValueError(f"CSV file {dispatch_long_path} must have either 'timestamp' or 'timestamp_utc' column")
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Load utilities to get actual max capacity
    util_df = pd.read_csv(utilities_csv)
    # Create capacity map: max_heat_MW * availability_factor
    capacity_map = {}
    for _, util_row in util_df.iterrows():
        unit_id = util_row['unit_id']
        max_cap = util_row['max_heat_MW'] * util_row['availability_factor']
        capacity_map[unit_id] = max_cap
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # For each unit, compute load duration curve (utilisation %)
    unit_ids = sorted(df['unit_id'].unique())
    colors = plt.cm.tab10(range(len(unit_ids)))
    
    for i, unit_id in enumerate(unit_ids):
        unit_data = df[df['unit_id'] == unit_id].copy()
        max_capacity = capacity_map.get(unit_id, 0.0)
        
        if max_capacity > 0:
            # Compute utilisation percentage
            unit_data['utilisation'] = (unit_data['heat_MW'] / max_capacity) * 100.0
            
            # Sort by utilisation descending
            sorted_util = np.sort(unit_data['utilisation'].values)[::-1]
            hours = np.arange(1, len(sorted_util) + 1)
            percent_of_time = (hours / len(sorted_util)) * 100
            
            ax.plot(percent_of_time, sorted_util,
                   linewidth=2, label=unit_id, color=colors[i])
    
    ax.set_xlabel('Percent of Time (%)', fontsize=12)
    ax.set_ylabel('Utilisation (%)', fontsize=12)
    ax.set_title('Unit Utilisation Duration Curves - 2020', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


def main():
    """CLI entrypoint for site dispatch computation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute 2020 baseline site utility dispatch')
    parser.add_argument('--demand-csv', default='Output/hourly_heat_demand_2020.csv',
                       help='Path to hourly demand CSV')
    parser.add_argument('--utilities-csv', default='Input/site_utilities_2020.csv',
                       help='Path to site utilities CSV')
    parser.add_argument('--mode', choices=['proportional', 'optimal_subset'], default='proportional',
                       help='Dispatch mode: proportional (default) or optimal_subset')
    parser.add_argument('--commitment-block-hours', type=int, default=24,
                       help='Hours per commitment block for optimal_subset mode (default: 24 = daily)')
    parser.add_argument('--unserved-penalty-nzd-per-MWh', type=float, default=50000.0,
                       help='Penalty for unserved demand in optimal_subset mode (default: 50000)')
    parser.add_argument('--reserve-frac', type=float, default=0.0,
                       help='Reserve requirement fraction of demand for optimal_subset mode (default: 0.0)')
    parser.add_argument('--reserve-penalty-nzd-per-MWh', type=float, default=0.0,
                       help='Penalty for reserve shortfall in optimal_subset mode (default: 0.0)')
    parser.add_argument('--no-load-cost-nzd-per-h', type=float, default=50.0,
                       help='No-load/hot-standby cost per hour per online unit in optimal_subset mode (default: 50.0)')
    parser.add_argument('--online-cost-applies-when', choices=['online_only', 'firing_only'], default='online_only',
                       help='When to apply online cost: online_only (default) or firing_only')
    parser.add_argument('--out-dispatch-long', default=None,
                       help='Output path for long-form dispatch CSV (auto-set based on mode)')
    parser.add_argument('--out-dispatch-wide', default=None,
                       help='Output path for wide-form dispatch CSV (auto-set based on mode)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate dispatch plots')
    parser.add_argument('--signals-config', default='Input/signals_config.toml',
                       help='Path to signals config TOML file')
    parser.add_argument('--epoch', default='2020',
                       help='Epoch label for signals (default: 2020)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: Output/)')
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('Output')
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'Figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default output paths based on mode
    if args.out_dispatch_long is None:
        if args.mode == 'proportional':
            args.out_dispatch_long = str(output_dir / 'site_dispatch_2020_long.csv')
        else:
            args.out_dispatch_long = str(output_dir / 'site_dispatch_2020_long_costed_opt.csv')
    
    if args.out_dispatch_wide is None:
        if args.mode == 'proportional':
            args.out_dispatch_wide = str(output_dir / 'site_dispatch_2020_wide.csv')
        else:
            args.out_dispatch_wide = str(output_dir / 'site_dispatch_2020_wide_opt.csv')
    
    # Load data
    print(f"Loading hourly demand from {args.demand_csv}...")
    demand_df = load_hourly_demand(args.demand_csv)
    
    print(f"Loading utilities from {args.utilities_csv}...")
    util_df = load_utilities(args.utilities_csv)
    
    print(f"Found {len(util_df)} utilities with total capacity {util_df['max_heat_MW'].sum():.2f} MW")
    
    # Load signals for the epoch
    print(f"Loading signals for epoch '{args.epoch}' from {args.signals_config}...")
    signals_config = load_signals_config(args.signals_config)
    signals = get_signals_for_epoch(signals_config, args.epoch)
    
    # Compute dispatch based on mode
    if args.mode == 'proportional':
        print("Computing proportional dispatch...")
        dispatch_long, dispatch_wide = allocate_baseline_dispatch(demand_df, util_df)
        
        # Add cost columns based on fuel type
        print("Computing costs...")
        dispatch_long = add_costs_to_dispatch(dispatch_long, util_df, signals)
    else:  # optimal_subset
        print(f"Computing optimal subset dispatch (block size: {args.commitment_block_hours}h)...")
        dispatch_long, dispatch_wide = allocate_dispatch_optimal_subset(
            demand_df, util_df, signals,
            commitment_block_hours=args.commitment_block_hours,
            unserved_penalty_nzd_per_MWh=args.unserved_penalty_nzd_per_MWh,
            reserve_frac=args.reserve_frac,
            reserve_penalty_nzd_per_MWh=args.reserve_penalty_nzd_per_MWh,
            no_load_cost_nzd_per_h=args.no_load_cost_nzd_per_h,
            online_cost_applies_when=args.online_cost_applies_when
        )
    
    # Convert timestamp_utc to ISO Z format for CSV output (keep datetime internally)
    dispatch_long_for_csv = dispatch_long.copy()
    dispatch_wide_for_csv = dispatch_wide.copy()
    dispatch_long_for_csv['timestamp_utc'] = to_iso_z(dispatch_long_for_csv['timestamp_utc'])
    dispatch_wide_for_csv['timestamp_utc'] = to_iso_z(dispatch_wide_for_csv['timestamp_utc'])
    
    # Save outputs
    print(f"Saving long-form dispatch to {args.out_dispatch_long}...")
    dispatch_long_for_csv.to_csv(args.out_dispatch_long, index=False)
    
    # For proportional mode, also save costed version
    if args.mode == 'proportional':
        costed_path = str(output_dir / 'site_dispatch_2020_long_costed.csv')
        print(f"Saving costed long-form dispatch to {costed_path}...")
        dispatch_long_for_csv.to_csv(costed_path, index=False)
    
    print(f"Saving wide-form dispatch to {args.out_dispatch_wide}...")
    dispatch_wide_for_csv.to_csv(args.out_dispatch_wide, index=False)
    
    # Compute annual summary
    print("\nComputing annual summary...")
    reserve_frac_val = args.reserve_frac if args.mode == 'optimal_subset' else 0.15
    annual_summary = compute_annual_summary(dispatch_long, reserve_frac=reserve_frac_val)
    
    # Save summary CSV
    if args.mode == 'proportional':
        summary_path = str(output_dir / 'site_dispatch_2020_summary.csv')
    else:
        summary_path = str(output_dir / 'site_dispatch_2020_summary_opt.csv')
    print(f"Saving annual summary to {summary_path}...")
    # Assert unit_id column exists before saving
    assert 'unit_id' in annual_summary.columns, "unit_id column missing before CSV write"
    annual_summary.to_csv(summary_path, index=False)
    
    # Self-check: verify TOTAL row cost closure
    if args.mode == 'optimal_subset' and 'TOTAL' in annual_summary['unit_id'].values:
        total_row = annual_summary[annual_summary['unit_id'] == 'TOTAL'].iloc[0]
        expected_total = (
            total_row.get('annual_fuel_cost_nzd', 0.0) +
            total_row.get('annual_carbon_cost_nzd', 0.0) +
            total_row.get('var_om_cost_nzd', 0.0) +
            total_row.get('fixed_on_cost_nzd', 0.0) +
            total_row.get('startup_cost_nzd', 0.0) +
            total_row.get('online_cost_nzd', 0.0) +
            total_row.get('annual_penalty_cost_nzd', 0.0)
        )
        actual_total = total_row.get('annual_total_cost_nzd', 0.0)
        tolerance = 1e-3  # 0.001 NZD tolerance
        assert abs(actual_total - expected_total) < tolerance, \
            f"TOTAL row cost closure failed: annual_total_cost_nzd={actual_total:.2f} != sum of components={expected_total:.2f} (diff={abs(actual_total - expected_total):.2f})"
        print(f"[OK] TOTAL row cost closure verified: {actual_total:,.2f} NZD")
    
    # Print summary table
    print("\n" + "="*80)
    print("Annual Summary by Unit")
    print("="*80)
    
    # Print unit rows (exclude SYSTEM and TOTAL)
    # After reset_index(), unit_id is a column, so filter by column value
    if 'unit_id' in annual_summary.columns:
        unit_rows = annual_summary[~annual_summary['unit_id'].isin(['SYSTEM', 'TOTAL'])]
        for _, row in unit_rows.iterrows():
            unit_id = row['unit_id']
    else:
        # Fallback if index-based (shouldn't happen after fix)
        unit_rows = annual_summary[~annual_summary.index.isin(['SYSTEM', 'TOTAL'])]
        for unit_id in unit_rows.index:
            row = unit_rows.loc[unit_id]
        print(f"\n{unit_id}:")
        print(f"  Annual heat (GWh):           {row['annual_heat_GWh']:10.2f}")
        print(f"  Annual fuel (MWh):           {row['annual_fuel_MWh']:10.2f}")
        print(f"  Annual CO2 (tCO2):           {row['annual_co2_tonnes']:10.2f}")
        print(f"  Annual fuel cost (NZD):      {row['annual_fuel_cost_nzd']:12,.2f}")
        print(f"  Annual carbon cost (NZD):    {row['annual_carbon_cost_nzd']:12,.2f}")
        print(f"  Annual total cost (NZD):     {row['annual_total_cost_nzd']:12,.2f}")
        print(f"  Avg cost per MWh_heat (NZD): {row['avg_cost_nzd_per_MWh_heat']:10.2f}")
    
    # Print system penalties section (if present)
    if 'unit_id' in annual_summary.columns:
        system_mask = annual_summary['unit_id'] == 'SYSTEM'
        if system_mask.any():
            system_row = annual_summary[system_mask].iloc[0]
    elif 'SYSTEM' in annual_summary.index:
        system_row = annual_summary.loc['SYSTEM']
    else:
        system_row = None
    
    if system_row is not None:
        print("\n" + "="*80)
        print("System-Level Penalties (Infeasibility Only)")
        print("="*80)
        # Show unserved (always show, even if 0)
        unserved_mwh = system_row.get('unserved_MW', 0.0)
        unserved_cost = system_row.get('unserved_cost_nzd', 0.0)
        print(f"  Unserved energy (MWh):       {unserved_mwh:10.2f}")
        print(f"  Unserved penalty cost (NZD): {unserved_cost:12,.2f}")
        
        # Show reserve (always show, even if 0)
        reserve_shortfall_mwh = system_row.get('reserve_shortfall_MW', 0.0)
        reserve_penalty_cost = system_row.get('reserve_penalty_cost_nzd', 0.0)
        print(f"  Reserve shortfall (MWh):     {reserve_shortfall_mwh:10.2f}")
        print(f"  Reserve penalty cost (NZD):  {reserve_penalty_cost:12,.2f}")
        
        # Total penalties (unserved + reserve only)
        total_penalties = unserved_cost + reserve_penalty_cost
        print(f"  Total penalty cost (NZD):    {total_penalties:12,.2f}")
        
        # Note: online/no-load cost is operational, not a penalty (shown in TOTAL section)
    
    # Get system_costs for online_cost display (needed for TOTAL section)
    system_costs_for_display = aggregate_system_costs(dispatch_long) if args.mode == 'optimal_subset' else {}
    
    # Print TOTAL row (use column-based lookup after reset_index)
    if 'unit_id' in annual_summary.columns:
        total_mask = annual_summary['unit_id'] == 'TOTAL'
        if total_mask.any():
            total_row = annual_summary[total_mask].iloc[0]
        else:
            total_row = None
    elif 'TOTAL' in annual_summary.index:
        total_row = annual_summary.loc['TOTAL']
    else:
        total_row = None
    
    if total_row is not None:
        print("\n" + "="*80)
        print("TOTAL (Units + System Penalties)")
        print("="*80)
        print(f"  Annual heat (GWh):           {total_row['annual_heat_GWh']:10.2f}")
        print(f"  Annual fuel (MWh):           {total_row['annual_fuel_MWh']:10.2f}")
        print(f"  Annual CO2 (tCO2):           {total_row['annual_co2_tonnes']:10.2f}")
        print(f"  Annual fuel cost (NZD):      {total_row['annual_fuel_cost_nzd']:12,.2f}")
        print(f"  Annual carbon cost (NZD):    {total_row['annual_carbon_cost_nzd']:12,.2f}")
        if 'var_om_cost_nzd' in total_row:
            print(f"  Variable O&M cost (NZD):     {total_row['var_om_cost_nzd']:12,.2f}")
        if 'fixed_on_cost_nzd' in total_row:
            print(f"  Fixed-on cost (NZD):         {total_row['fixed_on_cost_nzd']:12,.2f}")
        if 'startup_cost_nzd' in total_row:
            print(f"  Startup cost (NZD):          {total_row['startup_cost_nzd']:12,.2f}")
        
        # Operational adders (no-load/hot-standby) - not a penalty
        online_cost = system_costs_for_display.get('online_cost_nzd', 0.0)
        if online_cost > 0:
            print(f"  Online/no-load cost (NZD):  {online_cost:12,.2f}")
        
        print()
        if 'annual_operational_cost_nzd' in total_row:
            print(f"  Annual operational cost (NZD): {total_row['annual_operational_cost_nzd']:12,.2f}")
            print(f"    (includes: fuel + carbon + var_om + fixed_on + startup + no-load)")
        if 'annual_penalty_cost_nzd' in total_row:
            print(f"  Annual penalty cost (NZD):    {total_row['annual_penalty_cost_nzd']:12,.2f}")
            print(f"    (includes: unserved + reserve penalties)")
        print(f"  Annual total cost (NZD):     {total_row['annual_total_cost_nzd']:12,.2f}")
        if 'avg_operational_cost_nzd_per_MWh_heat' in total_row:
            print(f"  Avg operational cost/MWh (NZD): {total_row['avg_operational_cost_nzd_per_MWh_heat']:10.2f}")
        print(f"  Avg total cost per MWh_heat (NZD): {total_row['avg_cost_nzd_per_MWh_heat']:10.2f}")
    
    print("="*80)
    
    # Print annual diagnostics for optimal mode
    if args.mode == 'optimal_subset' and 'TOTAL' in annual_summary.index:
        total_row = annual_summary.loc['TOTAL']
        
        # Energy closure summary
        annual_demand_GWh = demand_df['heat_demand_MW'].sum() / 1000.0
        annual_served_GWh = dispatch_wide['total_heat_MW'].sum() / 1000.0
        # Parse timestamp_utc if needed for grouping
        dispatch_long_for_summary = dispatch_long.copy()
        if not pd.api.types.is_datetime64_any_dtype(dispatch_long_for_summary['timestamp_utc']):
            dispatch_long_for_summary['timestamp_utc'] = parse_any_timestamp(dispatch_long_for_summary['timestamp_utc'])
        annual_unserved_GWh = dispatch_long_for_summary.groupby('timestamp_utc')['unserved_MW'].first().sum() / 1000.0 if 'unserved_MW' in dispatch_long_for_summary.columns else 0.0
        
        print("\n[Annual Energy Summary]")
        print(f"  Annual demand:    {annual_demand_GWh:.6f} GWh")
        print(f"  Annual served:    {annual_served_GWh:.6f} GWh")
        print(f"  Annual unserved:  {annual_unserved_GWh:.6f} GWh")
        
        print("\n[Annual Diagnostics]")
        if 'hours_reserve_shortfall_gt_0' in total_row and not pd.isna(total_row['hours_reserve_shortfall_gt_0']):
            print(f"  Hours with reserve shortfall: {total_row['hours_reserve_shortfall_gt_0']:.0f}")
        if 'avg_online_units' in total_row and not pd.isna(total_row['avg_online_units']):
            print(f"  Mean online units: {total_row['avg_online_units']:.2f}")
        if 'avg_online_headroom_MW' in total_row and not pd.isna(total_row['avg_online_headroom_MW']):
            print(f"  Mean online headroom: {total_row['avg_online_headroom_MW']:.2f} MW")
        if 'p95_reserve_shortfall_MW' in total_row and not pd.isna(total_row['p95_reserve_shortfall_MW']):
            print(f"  P95 reserve shortfall: {total_row['p95_reserve_shortfall_MW']:.2f} MW")
        if 'avg_online_units_peak_season' in total_row and not pd.isna(total_row['avg_online_units_peak_season']):
            print(f"  Mean online units (peak season): {total_row['avg_online_units_peak_season']:.2f}")
        if 'avg_online_units_low_season' in total_row and not pd.isna(total_row['avg_online_units_low_season']):
            print(f"  Mean online units (low season): {total_row['avg_online_units_low_season']:.2f}")
    
    # Generate plots if requested (with error handling)
    if args.plot:
        print("\nGenerating dispatch plots...")
        
        try:
            if args.mode == 'proportional':
                plot_path = figures_dir / 'heat_2020_unit_stack.png'
                plot_dispatch_stack(args.out_dispatch_wide, str(plot_path), demand_df)
            else:  # optimal_subset
                # Stack plot
                plot_path = figures_dir / 'heat_2020_unit_stack_opt.png'
                try:
                    plot_dispatch_stack(args.out_dispatch_wide, str(plot_path), demand_df)
                except Exception as e:
                    print(f"[FAIL] Stack plot: {e}")
                    raise
                
                # Units online plot
                plot_path = figures_dir / 'heat_2020_units_online_opt.png'
                try:
                    plot_units_online(args.out_dispatch_long, str(plot_path))
                except Exception as e:
                    print(f"[FAIL] Units online plot: {e}")
                    raise
                
                # Utilisation duration plot
                plot_path = figures_dir / 'heat_2020_unit_utilisation_duration_opt.png'
                try:
                    plot_unit_utilisation_duration(args.out_dispatch_long, str(plot_path), args.utilities_csv)
                except Exception as e:
                    print(f"[FAIL] Utilisation duration plot: {e}")
                    raise
        except Exception as e:
            print(f"\n[ERROR] Plot generation failed: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()

