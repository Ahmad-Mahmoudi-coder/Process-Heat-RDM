"""
Site Utility Dispatch Module for 2020 Baseline

Allocates hourly heat demand across site utilities (coal boilers) using
proportional capacity dispatch and computes fuel consumption and CO2 emissions.
"""

# Bootstrap: allow `python .\src\script.py` (adds repo root to sys.path)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Union

from src.path_utils import repo_root, resolve_path, resolve_cfg_path, input_root
from src.load_signals import load_signals_config, get_signals_for_epoch, map_eval_epoch_to_signals_epoch
from src.time_utils import parse_any_timestamp, to_iso_z


def load_hourly_demand(path: str) -> Tuple[pd.DataFrame, float]:
    """
    Load hourly demand CSV, parse timestamp, sort by time, and validate.
    
    Supports both 'timestamp' and 'timestamp_utc' columns for backward compatibility.
    Normalizes to timestamp_utc with UTC timezone.
    
    Args:
        path: Path to hourly demand CSV file
        
    Returns:
        Tuple of (DataFrame with timestamp_utc and heat_demand_MW, dt_h in hours)
        DataFrame has timestamp_utc (UTC datetime) and heat_demand_MW columns.
        dt_h is the timestep duration in hours (constant across all steps).
        
    Raises:
        ValueError: If validation checks fail (duplicates, non-constant timestep, etc.)
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
    
    # Validate timestamps: strictly increasing, no duplicates
    if df['timestamp_utc'].duplicated().any():
        duplicates = df[df['timestamp_utc'].duplicated(keep=False)]['timestamp_utc'].unique()
        raise ValueError(f"Duplicate timestamps found: {duplicates[:5]}..." if len(duplicates) > 5 else f"Duplicate timestamps found: {duplicates}")
    
    if not df['timestamp_utc'].is_monotonic_increasing:
        raise ValueError("Timestamps are not strictly increasing after sort")
    
    # Compute timestep duration dt_h (hours per step)
    if len(df) < 2:
        raise ValueError(f"CSV file must have at least 2 rows to compute timestep duration, got {len(df)}")
    
    # Compute time deltas between consecutive rows
    time_deltas = df['timestamp_utc'].diff().dropna()
    time_deltas_hours = time_deltas.dt.total_seconds() / 3600.0
    
    # Find most common timestep (mode)
    dt_h = time_deltas_hours.mode()[0] if len(time_deltas_hours.mode()) > 0 else time_deltas_hours.iloc[0]
    
    # Validate constant timestep (all diffs must equal dt_h within tolerance)
    tolerance = 1e-6  # hours
    if not (time_deltas_hours - dt_h).abs().max() < tolerance:
        max_diff = (time_deltas_hours - dt_h).abs().max()
        raise ValueError(
            f"Timestep is not constant. Expected dt_h={dt_h:.6f} hours, "
            f"but found max deviation of {max_diff:.6f} hours. "
            f"Time deltas: {time_deltas_hours.head(10).tolist()}"
        )
    
    # Optional: validate row count for full-year data (only if dt_h==1.0 hour)
    # For PoC, allow any length as long as timestep is constant
    if dt_h == 1.0 and len(df) in [8760, 8784]:
        # Full year data - this is fine
        pass
    # Otherwise, allow any length (PoC flexibility)
    
    return df[['timestamp_utc', 'heat_demand_MW']], dt_h


def load_utilities(path: str, epoch: int = None) -> pd.DataFrame:
    """
    Load site utilities CSV and validate required fields.
    
    Filters to dispatchable units only (status in: existing, active, online).
    Excludes non-dispatchable units (status in: retired, planned, candidate, disabled, "", none, na).
    
    Required columns: unit_id, site_id, tech_type, fuel, status_<epoch> (or status as alias), max_heat_MW,
                     min_load_frac, efficiency_th, availability_factor, co2_factor_t_per_MWh_fuel
    
    Optional columns (defaults if missing):
    - fixed_on_cost_nzd_per_h (default 0.0)
    - startup_cost_nzd (default 0.0)
    - var_om_nzd_per_MWh_heat (default 0.0)
    - min_up_time_h (default 0)
    - min_down_time_h (default 0)
    
    Args:
        path: Path to site utilities CSV file
        epoch: Optional epoch year for status column selection (prefer status_<epoch>, fallback to status)
        
    Returns:
        DataFrame with dispatchable utility information only (all columns, with defaults filled)
        
    Raises:
        ValueError: If validation checks fail or no dispatchable units found
    """
    df = pd.read_csv(path)
    
    # Define status column name (prefer status_<epoch>, fallback to status)
    if epoch is not None:
        status_col = f"status_{epoch}"
    else:
        # For backward compatibility, try status_2020 first, then status
        if "status_2020" in df.columns:
            status_col = "status_2020"
        else:
            status_col = "status"  # Default fallback to generic status column
    
    # Handle status column selection
    if status_col not in df.columns:
        # Try fallback to generic "status" column
        if "status" in df.columns:
            df[status_col] = df["status"]
            print(f"[WARN] Column '{status_col}' not found, using 'status' as alias in {path}")
        else:
            # Both missing - will be caught in required_cols check below
            pass
    
    # Required columns
    required_cols = ['unit_id', 'site_id', 'tech_type', 'fuel', status_col, 
                     'max_heat_MW', 'min_load_frac', 'efficiency_th', 
                     'availability_factor', 'co2_factor_t_per_MWh_fuel']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {path}: {missing_cols}")
    
    # Check unit_id unique
    if df['unit_id'].duplicated().any():
        duplicates = df[df['unit_id'].duplicated(keep=False)]['unit_id'].unique()
        raise ValueError(f"Duplicate unit_id values in {path}: {list(duplicates)}")
    
    # Normalize status values to lowercase/strip
    df[status_col] = df[status_col].astype(str).str.lower().str.strip()
    
    # Define allowed status sets
    allowed_dispatchable = {"existing", "active", "online"}
    allowed_nondispatchable = {"retired", "planned", "candidate", "disabled", "", "none", "na"}
    allowed_all = allowed_dispatchable | allowed_nondispatchable
    
    # Validate all status values are in allowed sets
    invalid_status = df[~df[status_col].isin(allowed_all)]
    if len(invalid_status) > 0:
        bad_statuses = invalid_status[['unit_id', status_col]].to_dict('records')
        raise ValueError(f"Found units with invalid status values in {path}: {bad_statuses}")
    
    # Filter to dispatchable units only
    df_dispatchable = df[df[status_col].isin(allowed_dispatchable)].copy()
    
    # Print warning for excluded units
    excluded = df[~df[status_col].isin(allowed_dispatchable)]
    if len(excluded) > 0:
        excluded_list = excluded[['unit_id', status_col]].to_dict('records')
        print(f"[WARN] Excluding {len(excluded)} non-dispatchable unit(s) from {path}:")
        for unit_info in excluded_list:
            print(f"  - {unit_info['unit_id']}: status='{unit_info[status_col]}'")
    
    # Check if filtered df is empty
    if len(df_dispatchable) == 0:
        raise ValueError(f"No dispatchable units found in {path} (all units have non-dispatchable status)")
    
    # Continue with df_dispatchable
    df = df_dispatchable
    
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


def allocate_baseline_dispatch(demand_df: pd.DataFrame, util_df: pd.DataFrame, dt_h: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Allocate heat demand across utilities proportionally by capacity (vectorized).
    
    For each timestep, allocates heat to each unit proportional to its capacity:
    q_u_t = Q_t * (max_heat_MW_u / P_total)
    
    Note: heat_MW is average MW over the timestep. Energy per step = heat_MW * dt_h.
    
    Args:
        demand_df: DataFrame with timestamp_utc and heat_demand_MW columns
        util_df: DataFrame with utility information including max_heat_MW, 
                 efficiency_th, co2_factor_t_per_MWh_fuel
        dt_h: Timestep duration in hours
        
    Returns:
        Tuple of (long-form DataFrame, wide-form DataFrame)
        Long-form columns: timestamp_utc, unit_id, heat_MW (avg MW per step), fuel_MWh (energy per step), co2_tonnes (per step)
        Wide-form columns: timestamp_utc, total_heat_MW, CB1_MW, CB2_MW, ...
    """
    # Compute total installed capacity
    P_total = util_df['max_heat_MW'].sum()
    
    # Create allocation factors (capacity share for each unit)
    capacity_share = util_df['max_heat_MW'].values / P_total  # Shape: (n_units,)
    
    # Extract demand as array (shape: n_timesteps,)
    demand_heat = demand_df['heat_demand_MW'].values
    
    # Vectorized allocation: heat_matrix[i, j] = demand[i] * capacity_share[j]
    # Shape: (n_timesteps, n_units)
    heat_matrix = demand_heat[:, None] * capacity_share[None, :]
    
    # Build wide-form DataFrame
    unit_ids = util_df['unit_id'].tolist()
    dispatch_wide = pd.DataFrame({
        'timestamp_utc': demand_df['timestamp_utc'].values
    })
    
    # Add unit columns
    for i, unit_id in enumerate(unit_ids):
        dispatch_wide[f'{unit_id}_MW'] = heat_matrix[:, i]
    
    # Add total_heat_MW (should equal demand_heat, but compute for consistency)
    unit_cols = [f'{uid}_MW' for uid in unit_ids]
    dispatch_wide['total_heat_MW'] = dispatch_wide[unit_cols].sum(axis=1)
    
    # Reorder columns: timestamp_utc, total_heat_MW, then unit columns
    dispatch_wide = dispatch_wide[['timestamp_utc', 'total_heat_MW'] + unit_cols]
    
    # Build long-form by stacking (melt) the wide matrix
    long_results = []
    for i, timestamp in enumerate(demand_df['timestamp_utc']):
        for j, unit_id in enumerate(unit_ids):
            util_row = util_df.iloc[j]
            efficiency = util_row['efficiency_th']
            co2_factor = util_row['co2_factor_t_per_MWh_fuel']
            
            # heat_MW is average MW over the timestep
            heat_MW = heat_matrix[i, j]
            
            # Compute fuel consumption per step: energy = power * time
            # fuel_MWh_step = (heat_MW * dt_h) / efficiency
            fuel_MWh_step = (heat_MW * dt_h) / efficiency if efficiency > 0 else 0.0
            
            # Compute CO2 emissions per step
            co2_tonnes_step = fuel_MWh_step * co2_factor
            
            long_results.append({
                'timestamp_utc': timestamp,
                'unit_id': unit_id,
                'heat_MW': heat_MW,  # Average MW over timestep
                'fuel_MWh': fuel_MWh_step,  # Energy per step (MWh)
                'co2_tonnes': co2_tonnes_step,  # CO2 per step (tonnes)
            })
    
    dispatch_long = pd.DataFrame(long_results)
    
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
    online_cost_applies_when: str = 'online_only',
    dt_h: float = 1.0
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
    
    # Filter to existing units only (use the status column that was set in load_utilities)
    # load_utilities() sets a status_<epoch> column (or status as fallback) and filters to dispatchable units
    # Since load_utilities() already filters to dispatchable units, we just need to find the status column
    # Find the status column with deterministic preference: status_2020 > status
    if 'status_2020' in util_df.columns:
        status_col = 'status_2020'
    elif 'status' in util_df.columns:
        status_col = 'status'
    else:
        raise ValueError("No status column found in utilities DataFrame. Expected status_2020 or status.")
    # Note: load_utilities() already filters to dispatchable units, so all units here should be dispatchable
    existing_units = util_df[util_df[status_col].isin(['existing', 'active', 'online'])].copy()
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
    
    # Define stable first unit_id for deterministic system-level cost storage
    stable_first_unit_id = sorted(unit_ids)[0]
    
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
                            
                            # Energy per step = power * time
                            fuel_MWh_step = (heat_MW * dt_h) / efficiency
                            co2_tonnes_step = fuel_MWh_step * co2_factor
                            
                            # Fuel cost
                            fuel_type = util_row['fuel']
                            if fuel_type in ['lignite', 'coal']:
                                fuel_price = signals['coal_price_nzd_per_MWh_fuel']
                            elif fuel_type == 'biomass':
                                fuel_price = signals['biomass_price_nzd_per_MWh_fuel']
                            else:
                                fuel_price = 0.0
                            
                            hour_fuel_cost += fuel_MWh_step * fuel_price
                            hour_carbon_cost += co2_tonnes_step * signals['ets_price_nzd_per_tCO2']
                            # var_om is NZD/MWh_heat, so multiply by energy (MW * dt_h)
                            hour_var_om_cost += heat_MW * dt_h * var_om
                            # fixed_on is per hour, multiply by dt_h for this timestep
                            hour_fixed_on_cost += fixed_on * dt_h
                    
                    # Unserved penalty: penalty is per MWh, so multiply unserved_MW * dt_h to get energy
                    unserved_cost = unserved_MW * dt_h * unserved_penalty_nzd_per_MWh
                    
                    # Reserve shortfall penalty: penalty is per MWh, so multiply reserve_shortfall_MW * dt_h
                    reserve_penalty_cost = reserve_shortfall_MW * dt_h * reserve_penalty_nzd_per_MWh
                    
                    # Compute no-load/hot-standby cost: per hour, multiply by dt_h for this timestep
                    # Count online units (in subset)
                    n_online_units = len(subset_units)
                    if online_cost_applies_when == 'firing_only':
                        # Only count units that are actually firing (heat_MW > 0)
                        n_online_units = sum(1 for uid in subset_units if unit_outputs.get(uid, 0.0) > 0)
                    online_cost = no_load_cost_nzd_per_h * n_online_units * dt_h
                    
                    # Store results for this hour (for all units)
                    for unit_id in unit_ids:
                        util_row = existing_units[existing_units['unit_id'] == unit_id].iloc[0]
                        heat_MW = unit_outputs.get(unit_id, 0.0)
                        
                        # unit_online: 1 if unit is in selected subset (can be online with heat_MW = 0)
                        unit_online = 1 if unit_id in subset_units else 0
                        
                        # unit_on: 1 if unit is firing (heat_MW > 0), else 0
                        unit_on = 1 if heat_MW > 0 else 0
                        
                        # Compute fuel and emissions per step
                        efficiency = util_row['efficiency_th']
                        co2_factor = util_row['co2_factor_t_per_MWh_fuel']
                        # Energy per step = power * time
                        fuel_MWh = (heat_MW * dt_h) / efficiency if efficiency > 0 and heat_MW > 0 else 0.0
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
                            # var_om is NZD/MWh_heat, so multiply by energy (MW * dt_h)
                            unit_var_om_cost = heat_MW * dt_h * var_om
                            # fixed_on is per hour, multiply by dt_h for this timestep
                            unit_fixed_on_cost = fixed_on * dt_h
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
                        
                        # Store unserved_MW, reserve_shortfall_MW, and online_cost_nzd only once per hour (on stable first unit)
                        unserved_value = unserved_MW if unit_id == stable_first_unit_id else 0.0
                        reserve_shortfall_value = reserve_shortfall_MW if unit_id == stable_first_unit_id else 0.0
                        reserve_penalty_value = reserve_penalty_cost if unit_id == stable_first_unit_id else 0.0
                        unserved_cost_value = unserved_cost if unit_id == stable_first_unit_id else 0.0
                        online_cost_value = online_cost if unit_id == stable_first_unit_id else 0.0
                        
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
                                           online_cost_applies_when=online_cost_applies_when,
                                           dt_h=dt_h)
    
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
                            online_cost_applies_when: str = 'online_only',
                            dt_h: float = 1.0) -> pd.DataFrame:
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
    
    # Normalize timestamp_utc once at the start (for CSV-loaded data with string timestamps)
    # Ensure it's datetime64[ns, UTC] for reliable time operations
    if not pd.api.types.is_datetime64_any_dtype(dispatch_long['timestamp_utc']):
        dispatch_long['timestamp_utc'] = parse_any_timestamp(dispatch_long['timestamp_utc'])
    
    # Ensure timezone is UTC (parse_any_timestamp should handle this, but verify)
    if dispatch_long['timestamp_utc'].dtype.tz is None:
        dispatch_long['timestamp_utc'] = dispatch_long['timestamp_utc'].dt.tz_localize('UTC')
    elif str(dispatch_long['timestamp_utc'].dtype.tz) != 'UTC':
        dispatch_long['timestamp_utc'] = dispatch_long['timestamp_utc'].dt.tz_convert('UTC')
    
    # Derive year_start from timestamps (epoch-agnostic, replaces hard-coded 2020-01-01)
    ts0 = dispatch_long['timestamp_utc'].min()
    year_start = pd.Timestamp(year=ts0.year, month=1, day=1, tz='UTC')
    
    # Initialize cost columns
    dispatch_long['fuel_cost_nzd'] = 0.0
    dispatch_long['carbon_cost_nzd'] = 0.0
    dispatch_long['var_om_cost_nzd'] = 0.0
    dispatch_long['fixed_on_cost_nzd'] = 0.0
    dispatch_long['startup_cost_nzd'] = 0.0
    
    # Explicitly zero penalty-cost columns before recompute (avoid double-counting)
    if 'unserved_cost_nzd' in dispatch_long.columns:
        dispatch_long['unserved_cost_nzd'] = 0.0
    if 'reserve_penalty_cost_nzd' in dispatch_long.columns:
        dispatch_long['reserve_penalty_cost_nzd'] = 0.0
    if 'online_cost_nzd' in dispatch_long.columns:
        dispatch_long['online_cost_nzd'] = 0.0
    
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
        # Energy per step = power * time
        unit_data['fuel_MWh'] = (unit_data['heat_MW'] * dt_h) / efficiency
        unit_data['co2_tonnes'] = unit_data['fuel_MWh'] * co2_factor
        
        dispatch_long.loc[mask, 'fuel_cost_nzd'] = unit_data['fuel_MWh'] * fuel_price
        dispatch_long.loc[mask, 'carbon_cost_nzd'] = unit_data['co2_tonnes'] * ets_price
        # var_om is NZD/MWh_heat, so multiply by energy (MW * dt_h)
        dispatch_long.loc[mask, 'var_om_cost_nzd'] = unit_data['heat_MW'] * dt_h * var_om
        # fixed_on is per hour, multiply by dt_h for this timestep
        dispatch_long.loc[mask, 'fixed_on_cost_nzd'] = unit_data['unit_on'] * fixed_on * dt_h
        
        # Unserved cost (if not already computed)
        if 'unserved_cost_nzd' in dispatch_long.columns:
            # Recompute unserved cost from unserved_MW
            # Penalty is per MWh, so multiply unserved_MW * dt_h to get energy
            unserved_mask = mask & (dispatch_long['unserved_MW'] > 0)
            dispatch_long.loc[unserved_mask, 'unserved_cost_nzd'] = (
                dispatch_long.loc[unserved_mask, 'unserved_MW'] * dt_h * unserved_penalty_nzd_per_MWh
            )
        
        # Reserve penalty cost (if not already computed)
        if 'reserve_penalty_cost_nzd' in dispatch_long.columns:
            # Recompute reserve penalty from reserve_shortfall_MW
            # Penalty is per MWh, so multiply reserve_shortfall_MW * dt_h
            reserve_mask = mask & (dispatch_long['reserve_shortfall_MW'] > 0)
            dispatch_long.loc[reserve_mask, 'reserve_penalty_cost_nzd'] = (
                dispatch_long.loc[reserve_mask, 'reserve_shortfall_MW'] * dt_h * reserve_penalty_nzd_per_MWh
            )
        
        # Startup costs: detect transitions OFF->ON using unit_online (aligns with subset selection)
        # Fallback to unit_on if unit_online is missing (for older CSVs)
        unit_data = unit_data.sort_values('timestamp_utc')
        if 'unit_online' in unit_data.columns:
            # Use unit_online transitions (OFF->ON means unit becomes ONLINE, not just firing)
            unit_data['prev_unit_online'] = unit_data['unit_online'].shift(1, fill_value=0)
            startup_mask = (unit_data['unit_online'] == 1) & (unit_data['prev_unit_online'] == 0)
            use_unit_online = True
        else:
            # Fallback to unit_on for older CSVs
            unit_data['prev_unit_on'] = unit_data['unit_on'].shift(1, fill_value=0)
            startup_mask = (unit_data['unit_on'] == 1) & (unit_data['prev_unit_on'] == 0)
            use_unit_online = False
        
        # Allocate startup cost across block hours where unit is ONLINE
        startup_hours = unit_data[startup_mask]
        if len(startup_hours) > 0:
            for _, startup_row in startup_hours.iterrows():
                # Find block containing this hour
                timestamp = startup_row['timestamp_utc']
                # Calculate block start: round down to nearest block boundary
                # year_start derived from timestamps at function start (epoch-agnostic)
                hours_since_start = (timestamp - year_start).total_seconds() / 3600
                block_idx = int(hours_since_start // commitment_block_hours)
                block_start = year_start + pd.Timedelta(hours=block_idx * commitment_block_hours)
                block_end = block_start + pd.Timedelta(hours=commitment_block_hours)
                
                # Allocate across hours where unit is ONLINE (not just firing)
                if use_unit_online:
                    block_mask = (dispatch_long['unit_id'] == unit_id) & \
                                (dispatch_long['timestamp_utc'] >= block_start) & \
                                (dispatch_long['timestamp_utc'] < block_end) & \
                                (dispatch_long['unit_online'] == 1)
                else:
                    # Fallback: use unit_on
                    block_mask = (dispatch_long['unit_id'] == unit_id) & \
                                (dispatch_long['timestamp_utc'] >= block_start) & \
                                (dispatch_long['timestamp_utc'] < block_end) & \
                                (dispatch_long['unit_on'] == 1)
                n_block_hours = block_mask.sum()
                if n_block_hours > 0:
                    # Additive allocation (handles multiple startups in same block)
                    dispatch_long.loc[block_mask, 'startup_cost_nzd'] += startup_cost / n_block_hours
    
    # Recompute online cost if needed (store once per hour)
    if 'online_cost_nzd' in dispatch_long.columns:
        # Column already zeroed above
        
        # Group by timestamp_utc and compute online cost once per hour
        # (timestamp_utc already normalized at function start)
        for timestamp, hour_group in dispatch_long.groupby('timestamp_utc', sort=False):
            hour_mask = dispatch_long['timestamp_utc'] == timestamp
            
            # Count online units based on online_cost_applies_when with fallback
            if online_cost_applies_when == 'firing_only':
                # Count units with unit_on == 1
                n_online = int(hour_group['unit_on'].sum()) if 'unit_on' in hour_group.columns else 0
            else:  # 'online_only'
                # Count units with unit_online == 1, fallback to unit_on if missing
                if 'unit_online' in hour_group.columns:
                    n_online = int(hour_group['unit_online'].sum())
                else:
                    # Fallback for older CSVs without unit_online
                    n_online = int(hour_group['unit_on'].sum()) if 'unit_on' in hour_group.columns else 0
            
            if n_online > 0:
                # Store on first unit only (deterministic: sorted unit_id, not iloc[0])
                first_unit_id = sorted(hour_group['unit_id'].unique())[0]
                first_unit_mask = hour_mask & (dispatch_long['unit_id'] == first_unit_id)
                # online_cost is per hour, multiply by dt_h for this timestep
                dispatch_long.loc[first_unit_mask, 'online_cost_nzd'] = n_online * no_load_cost_nzd_per_h * dt_h
    
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
                          signals: Dict[str, float], dt_h: float = 1.0) -> pd.DataFrame:
    """
    Add cost columns to dispatch DataFrame based on fuel type and signals.
    
    Note: fuel_MWh in dispatch_long is already energy per step (MW * dt_h / efficiency),
    so fuel_cost = fuel_MWh * price is correct. No additional dt_h multiplication needed here.
    
    Args:
        dispatch_long: Long-form dispatch DataFrame with heat_MW, fuel_MWh (energy per step), co2_tonnes
        util_df: Utilities DataFrame with unit_id and fuel columns
        signals: Signals dict from get_signals_for_epoch
        dt_h: Timestep duration in hours (for backward compatibility, default 1.0)
        
    Returns:
        DataFrame with added columns: fuel_cost_nzd, carbon_cost_nzd, total_cost_nzd
    """
    dispatch_long = dispatch_long.copy()
    
    # Merge fuel type from utilities
    fuel_map = dict(zip(util_df['unit_id'], util_df['fuel']))
    dispatch_long['fuel_type'] = dispatch_long['unit_id'].map(fuel_map)
    
    # Initialize cost columns (ensure they exist and are float dtype)
    if 'fuel_cost_nzd' not in dispatch_long.columns:
        dispatch_long['fuel_cost_nzd'] = 0.0
    else:
        dispatch_long['fuel_cost_nzd'] = dispatch_long['fuel_cost_nzd'].astype(float).fillna(0.0)
    
    if 'carbon_cost_nzd' not in dispatch_long.columns:
        dispatch_long['carbon_cost_nzd'] = 0.0
    else:
        dispatch_long['carbon_cost_nzd'] = dispatch_long['carbon_cost_nzd'].astype(float).fillna(0.0)
    
    # Get ETS price (same for all units) - handle missing with warning
    ets_price = signals.get('ets_price_nzd_per_tCO2', 0.0)
    if ets_price == 0.0 and 'ets_price_nzd_per_tCO2' not in signals:
        print("[WARN] ETS price (ets_price_nzd_per_tCO2) not found in signals, defaulting to 0.0")
    
    # Process by fuel type
    for fuel_type in dispatch_long['fuel_type'].unique():
        mask = dispatch_long['fuel_type'] == fuel_type
        
        if fuel_type == 'lignite' or fuel_type == 'coal':
            # Use coal price and ETS
            # fuel_MWh is already energy per step, so multiply by price directly
            coal_price = signals.get('coal_price_nzd_per_MWh_fuel', 0.0)
            if coal_price == 0.0 and 'coal_price_nzd_per_MWh_fuel' not in signals:
                print("[WARN] Coal price (coal_price_nzd_per_MWh_fuel) not found in signals, defaulting to 0.0")
            dispatch_long.loc[mask, 'fuel_cost_nzd'] = (
                dispatch_long.loc[mask, 'fuel_MWh'] * coal_price
            )
            dispatch_long.loc[mask, 'carbon_cost_nzd'] = (
                dispatch_long.loc[mask, 'co2_tonnes'] * ets_price
            )
        elif fuel_type == 'biomass':
            # Use biomass price and minimal ETS (biomass has low emissions factor)
            biomass_price = signals.get('biomass_price_nzd_per_MWh_fuel', 0.0)
            if biomass_price == 0.0 and 'biomass_price_nzd_per_MWh_fuel' not in signals:
                print("[WARN] Biomass price (biomass_price_nzd_per_MWh_fuel) not found in signals, defaulting to 0.0")
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
    # Ensure all cost columns are float dtype before summing
    cost_components = ['fuel_cost_nzd', 'carbon_cost_nzd']
    for col in ['var_om_cost_nzd', 'fixed_on_cost_nzd', 'startup_cost_nzd', 
                'unserved_cost_nzd', 'reserve_penalty_cost_nzd', 'online_cost_nzd']:
        if col in dispatch_long.columns:
            dispatch_long[col] = dispatch_long[col].astype(float).fillna(0.0)
            cost_components.append(col)
    
    # Compute total_cost_nzd (ensure it's float)
    dispatch_long['total_cost_nzd'] = dispatch_long[cost_components].sum(axis=1).astype(float)
    
    # Ensure required cost columns are float dtype
    dispatch_long['fuel_cost_nzd'] = dispatch_long['fuel_cost_nzd'].astype(float)
    dispatch_long['carbon_cost_nzd'] = dispatch_long['carbon_cost_nzd'].astype(float)
    
    # Drop fuel_type column (it was just for internal use)
    dispatch_long = dispatch_long.drop(columns=['fuel_type'])
    
    return dispatch_long


def aggregate_system_costs(dispatch_long: pd.DataFrame) -> dict:
    """
    Aggregate system-level costs (stored once per hour on first unit row).
    These must NOT be summed by unit to avoid attributing all penalties to CB1.
    
    Returns dict with system cost totals.
    """
    # Normalize timestamp_utc if needed (for CSV-loaded data with string timestamps)
    dispatch_long_for_agg = dispatch_long.copy()
    if not pd.api.types.is_datetime64_any_dtype(dispatch_long_for_agg['timestamp_utc']):
        dispatch_long_for_agg['timestamp_utc'] = parse_any_timestamp(dispatch_long_for_agg['timestamp_utc'])
    
    system_costs = {}
    # System penalties are stored once per hour (on first unit row), so aggregate by timestamp_utc first
    if 'unserved_MW' in dispatch_long_for_agg.columns:
        system_costs['unserved_MW'] = dispatch_long_for_agg.groupby('timestamp_utc')['unserved_MW'].first().sum()
    if 'unserved_cost_nzd' in dispatch_long_for_agg.columns:
        system_costs['unserved_cost_nzd'] = dispatch_long_for_agg.groupby('timestamp_utc')['unserved_cost_nzd'].first().sum()
    if 'reserve_shortfall_MW' in dispatch_long_for_agg.columns:
        system_costs['reserve_shortfall_MW'] = dispatch_long_for_agg.groupby('timestamp_utc')['reserve_shortfall_MW'].first().sum()
    if 'reserve_penalty_cost_nzd' in dispatch_long_for_agg.columns:
        system_costs['reserve_penalty_cost_nzd'] = dispatch_long_for_agg.groupby('timestamp_utc')['reserve_penalty_cost_nzd'].first().sum()
    if 'online_cost_nzd' in dispatch_long_for_agg.columns:
        system_costs['online_cost_nzd'] = dispatch_long_for_agg.groupby('timestamp_utc')['online_cost_nzd'].first().sum()
    return system_costs


def compute_annual_summary(dispatch_long: pd.DataFrame, reserve_frac: float = 0.0, dt_h: float = 1.0) -> pd.DataFrame:
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
        'heat_MW': 'sum',  # Sum of average MW per step
        'fuel_MWh': 'sum',  # Sum of energy per step (already MWh per step)
        'co2_tonnes': 'sum',  # Sum of CO2 per step
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
    
    # Convert to annual totals: energy = sum(MW * dt_h) / 1000 = GWh
    # heat_MW is average MW per step, so annual energy = sum(heat_MW * dt_h) / 1000
    annual_by_unit['annual_heat_GWh'] = (annual_by_unit['heat_MW'] * dt_h) / 1000.0
    annual_by_unit['annual_fuel_MWh'] = annual_by_unit['fuel_MWh']  # Already energy per step, summed
    annual_by_unit['annual_co2_tonnes'] = annual_by_unit['co2_tonnes']  # Already per step, summed
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
    # Note: system_costs contains sums of MW values (stored once per timestamp)
    # Convert to energy: unserved_MW and reserve_shortfall_MW are average MW per step
    # Annual energy = sum(MW * dt_h)
    unserved_penalty_cost_nzd = system_costs.get('unserved_cost_nzd', 0.0)
    reserve_penalty_cost_nzd = system_costs.get('reserve_penalty_cost_nzd', 0.0)
    # system_costs['unserved_MW'] is sum of MW values (stored once per timestamp)
    # Convert to energy: multiply by dt_h to get MWh
    unserved_energy_mwh = system_costs.get('unserved_MW', 0.0) * dt_h
    reserve_shortfall_mwh = system_costs.get('reserve_shortfall_MW', 0.0) * dt_h
    
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
    # Note: system_costs contains sums of MW values (average MW per step, stored once per timestamp)
    # For display, keep as MW (sum of average MW per step) - energy conversion happens in annual totals
    if 'unserved_MW' in system_costs:
        system_dict['unserved_MW'] = system_costs['unserved_MW']  # Sum of average MW per step
    if 'unserved_cost_nzd' in system_costs:
        system_dict['unserved_cost_nzd'] = system_costs['unserved_cost_nzd']  # Already cost (includes dt_h)
    if 'reserve_shortfall_MW' in system_costs:
        system_dict['reserve_shortfall_MW'] = system_costs['reserve_shortfall_MW']  # Sum of average MW per step
    if 'reserve_penalty_cost_nzd' in system_costs:
        system_dict['reserve_penalty_cost_nzd'] = system_costs['reserve_penalty_cost_nzd']  # Already cost (includes dt_h)
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
    # Note: heat_MW sum is sum of average MW per step; convert to GWh using dt_h
    total_heat_GWh = (annual_by_unit['heat_MW'].sum() * dt_h) / 1000.0
    total_dict = {
        'heat_MW': annual_by_unit['heat_MW'].sum(),  # Sum of average MW per step
        'fuel_MWh': annual_by_unit['fuel_MWh'].sum(),  # Sum of energy per step
        'co2_tonnes': annual_by_unit['co2_tonnes'].sum(),  # Sum of CO2 per step
        'fuel_cost_nzd': annual_by_unit['fuel_cost_nzd'].sum(),
        'carbon_cost_nzd': annual_by_unit['carbon_cost_nzd'].sum(),
        'total_cost_nzd': annual_by_unit['total_cost_nzd'].sum(),
        'annual_heat_GWh': total_heat_GWh,  # Energy = sum(MW * dt_h) / 1000
        'annual_fuel_MWh': annual_by_unit['annual_fuel_MWh'].sum(),  # Already energy per step, summed
        'annual_co2_tonnes': annual_by_unit['annual_co2_tonnes'].sum(),  # Already per step, summed
        'annual_fuel_cost_nzd': annual_by_unit['annual_fuel_cost_nzd'].sum(),
        'annual_carbon_cost_nzd': annual_by_unit['annual_carbon_cost_nzd'].sum(),
        'annual_total_cost_nzd': grand_total_cost_nzd,  # Operational + penalties
        'annual_operational_cost_nzd': total_operational_cost_nzd,  # Units + online/no-load
        'annual_penalty_cost_nzd': total_system_penalties_nzd,  # Unserved + reserve penalties
        'annual_system_cost_nzd': total_system_penalties_nzd + system_costs.get('online_cost_nzd', 0.0),  # Penalties + online
        'avg_cost_nzd_per_MWh_heat': (
            grand_total_cost_nzd /
            (total_heat_GWh * 1000.0) if total_heat_GWh > 0 else 0.0
        ),  # Based on served heat (not demand)
        'avg_operational_cost_nzd_per_MWh_heat': (
            total_operational_cost_nzd /
            (total_heat_GWh * 1000.0) if total_heat_GWh > 0 else 0.0
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
    # The concat index contains unit labels (unit_id values plus 'SYSTEM' and 'TOTAL')
    summary.index.name = 'unit_id'
    summary = summary.reset_index()
    
    # Assert that unit_id is now a column (this should always be true after reset_index)
    assert 'unit_id' in summary.columns, "unit_id column missing after reset_index"
    
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


def plot_dispatch_stack(dispatch_wide_path: str, output_path: str, demand_df: Optional[pd.DataFrame] = None, epoch: int = None):
    """
    Generate stacked area plot of unit dispatch over time.
    
    Args:
        dispatch_wide_path: Path to wide-form dispatch CSV
        output_path: Path to save the figure
        demand_df: Optional DataFrame with actual demand (timestamp, heat_demand_MW)
        epoch: Optional epoch year for title
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
    if epoch:
        ax.set_title(f'Unit Dispatch Stack - {epoch}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Unit Dispatch Stack', fontsize=14, fontweight='bold')
    
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


def plot_units_online(dispatch_long_path: str, output_path: str, epoch: int = None):
    """
    Plot number of boilers online over time (daily or weekly mean).
    
    Args:
        dispatch_long_path: Path to long-form dispatch CSV with unit_on column
        output_path: Path to save the figure
        epoch: Optional epoch year for title
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
    if epoch:
        ax.set_title(f'{plot_label} - {epoch}', fontsize=14, fontweight='bold')
    else:
        ax.set_title(plot_label, fontsize=14, fontweight='bold')
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
                                   utilities_csv: str = 'Input/site_utilities_2020.csv',
                                   epoch: int = None):
    """
    Plot per-unit utilisation/load-duration style plot (one figure with all units).
    
    Args:
        dispatch_long_path: Path to long-form dispatch CSV
        output_path: Path to save the figure
        utilities_csv: Path to utilities CSV to get max_heat_MW and availability_factor
        epoch: Optional epoch year for title
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
    util_csv_resolved = resolve_path(utilities_csv)
    if not util_csv_resolved.exists():
        raise FileNotFoundError(f"Utilities CSV file not found: {util_csv_resolved} (resolved from {utilities_csv})")
    util_df = pd.read_csv(util_csv_resolved)
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
    if epoch:
        ax.set_title(f'Unit Utilisation Duration Curves - {epoch}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Unit Utilisation Duration Curves', fontsize=14, fontweight='bold')
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
    parser.add_argument('--utilities-csv', default=None,
                       help='Path to site utilities CSV (default: auto-discover from Input/site/utilities/)')
    parser.add_argument('--demandpack-config', default=None,
                       help='Path to demandpack config (optional, used to check for utilities path)')
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
    parser.add_argument('--signals-config', default=None,
                       help='Path to signals config TOML file (default: Input/signals/signals_config.toml)')
    parser.add_argument('--epoch', type=int, default=2020,
                       help='Epoch year (default: 2020)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (deprecated: use --output-root and --run-id)')
    parser.add_argument('--output-root', type=str, default=None,
                       help='Output root directory (default: repo_root/Output)')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run ID (if provided, writes to run_dir; otherwise uses output-dir or Output/)')
    args = parser.parse_args()
    
    epoch = args.epoch
    
    # Resolve paths
    ROOT = repo_root()
    INPUT_DIR = input_root()
    
    print(f"Repository root: {ROOT}")
    print(f"Epoch: {epoch}")
    
    # Determine output paths
    output_paths = None
    if args.run_id:
        # Use new output path system
        from src.output_paths import resolve_run_paths
        if args.output_root:
            output_root = resolve_path(args.output_root)
        else:
            output_root = ROOT / 'Output'
        
        # Infer config path from demand_csv if possible
        config_path = None
        if args.demandpack_config:
            config_path = resolve_path(args.demandpack_config)
        
        output_paths = resolve_run_paths(
            output_root=output_root,
            epoch=epoch,
            config_path=config_path,
            run_id=args.run_id
        )
        output_dir = output_paths['run_dir']
        figures_dir = output_paths['run_figures_dir']
    else:
        # Legacy mode: use output-dir or default
        if args.output_dir:
            output_dir = resolve_path(args.output_dir)
        else:
            output_dir = ROOT / 'Output'
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / 'Figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default output paths based on mode (epoch-tagged)
    if args.out_dispatch_long is None:
        if args.mode == 'proportional':
            args.out_dispatch_long = str(output_dir / f'site_dispatch_{epoch}_long.csv')
        else:
            args.out_dispatch_long = str(output_dir / f'site_dispatch_{epoch}_long_costed_opt.csv')
    
    if args.out_dispatch_wide is None:
        if args.mode == 'proportional':
            args.out_dispatch_wide = str(output_dir / f'site_dispatch_{epoch}_wide.csv')
        else:
            args.out_dispatch_wide = str(output_dir / f'site_dispatch_{epoch}_wide_opt.csv')
    
    # Resolve input paths
    demand_csv_resolved = resolve_path(args.demand_csv)
    
    # Resolve utilities CSV with auto-discovery
    if args.utilities_csv:
        utilities_csv_resolved = resolve_path(args.utilities_csv)
    else:
        # Auto-discovery logic
        utilities_csv_resolved = None
        attempted_paths = []
        
        # Try a) From demandpack config (if provided and contains utilities path)
        if args.demandpack_config:
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    tomllib = None
            
            if tomllib:
                try:
                    demandpack_config_path = resolve_path(args.demandpack_config)
                    with open(demandpack_config_path, 'rb') as f:
                        demandpack_config = tomllib.load(f)
                    
                    # Check for utilities path in config (could be in [general] or [site] section)
                    utilities_path = None
                    if 'general' in demandpack_config and 'utilities_csv' in demandpack_config['general']:
                        utilities_path = demandpack_config['general']['utilities_csv']
                    elif 'site' in demandpack_config and 'utilities_csv' in demandpack_config['site']:
                        utilities_path = demandpack_config['site']['utilities_csv']
                    
                    if utilities_path:
                        candidate = resolve_cfg_path(demandpack_config_path, utilities_path)
                        attempted_paths.append(f"  a) From demandpack config: {candidate}")
                        if candidate.exists():
                            utilities_csv_resolved = candidate
                except Exception as e:
                    # Silently fail - config might not have utilities path
                    pass
        
        # Try b) <input_dir>/site/utilities/site_utilities_{epoch}.csv (epoch-aware)
        if utilities_csv_resolved is None:
            candidate = INPUT_DIR / 'site' / 'utilities' / f'site_utilities_{epoch}.csv'
            attempted_paths.append(f"  b) Epoch-specific location: {candidate}")
            if candidate.exists():
                utilities_csv_resolved = candidate
        
        # Try c) Search <input_dir>/site/utilities/ for *utilities*{epoch}*.csv (epoch-aware)
        if utilities_csv_resolved is None:
            utilities_dir = INPUT_DIR / 'site' / 'utilities'
            if utilities_dir.exists():
                matches = sorted(utilities_dir.glob(f'*utilities*{epoch}*.csv'))
                attempted_paths.append(f"  c) Searched in: {utilities_dir} (pattern: *utilities*{epoch}*.csv)")
                if len(matches) == 1:
                    utilities_csv_resolved = matches[0]
                    attempted_paths.append(f"     Found: {utilities_csv_resolved}")
                elif len(matches) > 1:
                    # Multiple matches - list them
                    print(f"[ERROR] Multiple utilities CSV files found for epoch {epoch} in {utilities_dir}:")
                    print("Please specify one using --utilities-csv")
                    print()
                    for i, match in enumerate(matches, 1):
                        print(f"  {i}. {match.name} ({match})")
                    print()
                    print(f"Example: --utilities-csv {matches[0].relative_to(ROOT)}")
                    sys.exit(1)
        
        # Try d) Fallback to single "master" utilities file (e.g., site_utilities.csv) if present
        if utilities_csv_resolved is None:
            utilities_dir = INPUT_DIR / 'site' / 'utilities'
            if utilities_dir.exists():
                master_candidates = sorted(utilities_dir.glob('site_utilities.csv'))
                if len(master_candidates) == 1:
                    utilities_csv_resolved = master_candidates[0]
                    attempted_paths.append(f"  d) Fallback master file: {utilities_csv_resolved}")
                    print(f"[WARN] Using master utilities file (not epoch-specific): {utilities_csv_resolved}")
        
        # If still not found, show error with all attempted paths
        if utilities_csv_resolved is None or not utilities_csv_resolved.exists():
            print(f"[ERROR] Utilities CSV file not found. Attempted paths:")
            for path_str in attempted_paths:
                print(path_str)
            print()
            print("Please specify the utilities CSV using --utilities-csv")
            print(f"Example: --utilities-csv {INPUT_DIR / 'site' / 'utilities' / 'site_utilities_2020.csv'}")
            sys.exit(1)
        else:
            # Successfully auto-discovered
            print(f"[OK] Auto-discovered utilities CSV: {utilities_csv_resolved}")
    
    # Resolve signals config (use default if not provided)
    if args.signals_config:
        signals_config_resolved = resolve_path(args.signals_config)
    else:
        signals_config_resolved = INPUT_DIR / 'signals' / 'signals_config.toml'
    
    # Validation
    if not demand_csv_resolved.exists():
        print(f"[ERROR] Demand CSV file not found: {demand_csv_resolved}")
        sys.exit(1)
    if not utilities_csv_resolved.exists():
        print(f"[ERROR] Utilities CSV file not found: {utilities_csv_resolved}")
        print("Please specify the utilities CSV using --utilities-csv")
        sys.exit(1)
    if not signals_config_resolved.exists():
        print(f"[ERROR] Signals config file not found: {signals_config_resolved}")
        sys.exit(1)
    
    # Load data
    print(f"Loading hourly demand from {demand_csv_resolved}...")
    demand_df, dt_h = load_hourly_demand(str(demand_csv_resolved))
    print(f"[OK] Timestep duration: dt_h = {dt_h:.6f} hours")
    
    print(f"Loading utilities from {utilities_csv_resolved}...")
    util_df = load_utilities(str(utilities_csv_resolved), epoch=epoch)
    
    print(f"Found {len(util_df)} utilities with total capacity {util_df['max_heat_MW'].sum():.2f} MW")
    
    # Load signals for the epoch (with epoch mapping)
    print(f"Loading signals for epoch '{epoch}' from {signals_config_resolved}...")
    signals_config = load_signals_config(str(signals_config_resolved))
    
    # Map eval_epoch to signals_epoch (handles cases where exact epoch not available)
    signals_epoch = map_eval_epoch_to_signals_epoch(epoch, signals_config, INPUT_DIR)
    if signals_epoch != epoch:
        print(f"[WARN] Epoch mapping: eval_epoch {epoch} -> signals_epoch {signals_epoch}")
    signals = get_signals_for_epoch(signals_config, str(signals_epoch))
    
    # Compute dispatch based on mode
    if args.mode == 'proportional':
        print("Computing proportional dispatch...")
        dispatch_long, dispatch_wide = allocate_baseline_dispatch(demand_df, util_df, dt_h)
    else:  # optimal_subset
        print(f"Computing optimal subset dispatch (block size: {args.commitment_block_hours}h)...")
        dispatch_long, dispatch_wide = allocate_dispatch_optimal_subset(
            demand_df, util_df, signals,
            commitment_block_hours=args.commitment_block_hours,
            unserved_penalty_nzd_per_MWh=args.unserved_penalty_nzd_per_MWh,
            reserve_frac=args.reserve_frac,
            reserve_penalty_nzd_per_MWh=args.reserve_penalty_nzd_per_MWh,
            no_load_cost_nzd_per_h=args.no_load_cost_nzd_per_h,
            online_cost_applies_when=args.online_cost_applies_when,
            dt_h=dt_h
        )
    
    # Add cost columns for BOTH modes to ensure consistent schema
    # This ensures fuel_cost_nzd, carbon_cost_nzd, total_cost_nzd always exist
    print("Computing costs...")
    dispatch_long = add_costs_to_dispatch(dispatch_long, util_df, signals, dt_h)
    
    # Validate dispatch outputs (energy closure, schema, etc.) - AFTER costs are added
    validate_dispatch_outputs(dispatch_long, dispatch_wide, demand_df, dt_h)
    
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
        costed_path = str(output_dir / f'site_dispatch_{epoch}_long_costed.csv')
        print(f"Saving costed long-form dispatch to {costed_path}...")
        dispatch_long_for_csv.to_csv(costed_path, index=False)
    
    print(f"Saving wide-form dispatch to {args.out_dispatch_wide}...")
    dispatch_wide_for_csv.to_csv(args.out_dispatch_wide, index=False)
    
    # Validate hourly cost storage convention for optimal_subset mode
    if args.mode == 'optimal_subset':
        _debug_validate_hourly_cost_storage(dispatch_long)
        # Regression check: verify online_cost fallback works (only if debug flag is set)
        if os.environ.get("DISPATCH_DEBUG_VALIDATE", "0") == "1":
            _debug_validate_online_cost_fallback(
                dispatch_long, util_df, signals,
                args.commitment_block_hours,
                args.no_load_cost_nzd_per_h
            )
        else:
            print("[INFO] Skipping online_cost fallback validation (set DISPATCH_DEBUG_VALIDATE=1 to enable)")
    
    # Compute annual summary
    print("\nComputing annual summary...")
    reserve_frac_val = args.reserve_frac if args.mode == 'optimal_subset' else 0.0
    annual_summary = compute_annual_summary(dispatch_long, reserve_frac=reserve_frac_val)
    
    # Validate summary schema (self-test)
    _debug_validate_summary_schema(annual_summary)
    
    # Save summary CSV (epoch-tagged)
    if args.mode == 'proportional':
        summary_path = str(output_dir / f'site_dispatch_{epoch}_summary.csv')
    else:
        summary_path = str(output_dir / f'site_dispatch_{epoch}_summary_opt.csv')
    
    # Outputs are written directly to run_dir (no copying to latest)
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
    # After compute_annual_summary(), unit_id is always a column
    unit_rows = annual_summary[~annual_summary['unit_id'].isin(['SYSTEM', 'TOTAL'])]
    for _, row in unit_rows.iterrows():
        unit_id = row['unit_id']
        print(f"\n{unit_id}:")
        print(f"  Annual heat (GWh):           {row['annual_heat_GWh']:10.2f}")
        print(f"  Annual fuel (MWh):           {row['annual_fuel_MWh']:10.2f}")
        print(f"  Annual CO2 (tCO2):           {row['annual_co2_tonnes']:10.2f}")
        print(f"  Annual fuel cost (NZD):      {row['annual_fuel_cost_nzd']:12,.2f}")
        print(f"  Annual carbon cost (NZD):    {row['annual_carbon_cost_nzd']:12,.2f}")
        print(f"  Annual total cost (NZD):     {row['annual_total_cost_nzd']:12,.2f}")
        print(f"  Avg cost per MWh_heat (NZD): {row['avg_cost_nzd_per_MWh_heat']:10.2f}")
    
    # Print system penalties section (if present)
    # After compute_annual_summary(), unit_id is always a column
    system_mask = annual_summary['unit_id'] == 'SYSTEM'
    system_row = annual_summary[system_mask].iloc[0] if system_mask.any() else None
    
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
    # After compute_annual_summary(), unit_id is always a column
    total_mask = annual_summary['unit_id'] == 'TOTAL'
    total_row = annual_summary[total_mask].iloc[0] if total_mask.any() else None
    
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
    # After compute_annual_summary(), unit_id is always a column, not index
    if args.mode == 'optimal_subset' and 'TOTAL' in annual_summary['unit_id'].values:
        total_row = annual_summary[annual_summary['unit_id'] == 'TOTAL'].iloc[0]
        
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
                plot_path = figures_dir / f'heat_{epoch}_unit_stack.png'
                plot_dispatch_stack(args.out_dispatch_wide, str(plot_path), demand_df, epoch=epoch)
            else:  # optimal_subset
                # Stack plot
                plot_path = figures_dir / f'heat_{epoch}_unit_stack_opt.png'
                try:
                    plot_dispatch_stack(args.out_dispatch_wide, str(plot_path), demand_df, epoch=epoch)
                except Exception as e:
                    print(f"[FAIL] Stack plot: {e}")
                    raise
                
                # Units online plot
                plot_path = figures_dir / f'heat_{epoch}_units_online_opt.png'
                try:
                    plot_units_online(args.out_dispatch_long, str(plot_path), epoch=epoch)
                except Exception as e:
                    print(f"[FAIL] Units online plot: {e}")
                    raise
                
                # Utilisation duration plot
                plot_path = figures_dir / f'heat_{epoch}_unit_utilisation_duration_opt.png'
                try:
                    plot_unit_utilisation_duration(args.out_dispatch_long, str(plot_path), str(utilities_csv_resolved), epoch=epoch)
                except Exception as e:
                    print(f"[FAIL] Utilisation duration plot: {e}")
                    raise
            
            # Figures are written directly to run_figures_dir (no copying to latest)
        except Exception as e:
            print(f"\n[ERROR] Plot generation failed: {e}")
            sys.exit(1)


def _canonicalize_timestamps(ts_series: Union[pd.Series, pd.Index]) -> pd.Series:
    """
    Canonicalize timestamps to UTC datetime64[ns, UTC] for comparison.
    
    Accepts Series/Index of strings or datetime objects and converts to
    standardized UTC datetime64[ns, UTC] format.
    
    Args:
        ts_series: Series or Index of timestamp strings or datetime objects
        
    Returns:
        Series with dtype datetime64[ns, UTC]
    """
    if isinstance(ts_series, pd.Index):
        ts_series = pd.Series(ts_series)
    
    # Convert to UTC datetime64[ns, UTC]
    ts_canonical = pd.to_datetime(ts_series, utc=True, errors='raise')
    
    # Ensure timezone-aware UTC
    if ts_canonical.dt.tz is None:
        ts_canonical = ts_canonical.dt.tz_localize('UTC')
    else:
        ts_canonical = ts_canonical.dt.tz_convert('UTC')
    
    return ts_canonical


def validate_dispatch_outputs(dispatch_long: pd.DataFrame, dispatch_wide: pd.DataFrame, 
                               demand_df: pd.DataFrame, dt_h: float) -> None:
    """
    Validate dispatch outputs for correctness (energy closure, schema, etc.).
    
    Args:
        dispatch_long: Long-form dispatch DataFrame
        dispatch_wide: Wide-form dispatch DataFrame
        demand_df: Demand DataFrame with timestamp_utc and heat_demand_MW
        dt_h: Timestep duration in hours
    """
    # 1. Required columns exist
    required_long_cols = ['timestamp_utc', 'unit_id', 'heat_MW', 'fuel_MWh', 'co2_tonnes', 
                          'fuel_cost_nzd', 'carbon_cost_nzd', 'total_cost_nzd']
    missing_long = [col for col in required_long_cols if col not in dispatch_long.columns]
    if missing_long:
        raise ValueError(f"dispatch_long missing required columns: {missing_long}")
    
    required_wide_cols = ['timestamp_utc', 'total_heat_MW']
    missing_wide = [col for col in required_wide_cols if col not in dispatch_wide.columns]
    if missing_wide:
        raise ValueError(f"dispatch_wide missing required columns: {missing_wide}")
    
    # Check for at least one unit column in wide form
    unit_cols = [col for col in dispatch_wide.columns if col.endswith('_MW') and col != 'total_heat_MW']
    if len(unit_cols) == 0:
        raise ValueError("dispatch_wide must have at least one unit column (unit_id_MW)")
    
    # 2. Timestamps match demand timestamps exactly (canonicalize for comparison)
    # Canonicalize all timestamps to UTC datetime64[ns, UTC]
    demand_ts_canonical = _canonicalize_timestamps(demand_df['timestamp_utc']).sort_values().unique()
    dispatch_long_ts_canonical = _canonicalize_timestamps(dispatch_long['timestamp_utc']).sort_values().unique()
    
    if len(dispatch_long_ts_canonical) != len(demand_ts_canonical):
        raise ValueError(
            f"Timestamp count mismatch: dispatch_long has {len(dispatch_long_ts_canonical)} unique timestamps, "
            f"demand_df has {len(demand_ts_canonical)}"
        )
    
    if len(dispatch_wide) != len(demand_df):
        raise ValueError(
            f"Row count mismatch: dispatch_wide has {len(dispatch_wide)} rows, demand_df has {len(demand_df)}"
        )
    
    # Check timestamp sets match (elementwise comparison after canonicalization)
    if not (demand_ts_canonical == dispatch_long_ts_canonical).all():
        # Find mismatches for better error message
        demand_set = set(demand_ts_canonical)
        dispatch_set = set(dispatch_long_ts_canonical)
        missing_in_dispatch = demand_set - dispatch_set
        extra_in_dispatch = dispatch_set - demand_set
        
        # Show first 3 examples from each set for debugging
        missing_sample = list(missing_in_dispatch)[:3] if missing_in_dispatch else []
        extra_sample = list(extra_in_dispatch)[:3] if extra_in_dispatch else []
        
        raise ValueError(
            f"Timestamp sets don't match after canonicalization. "
            f"Missing in dispatch: {len(missing_in_dispatch)}, Extra in dispatch: {len(extra_in_dispatch)}. "
            f"Sample missing: {missing_sample}, Sample extra: {extra_sample}"
        )
    
    # Validate dispatch_wide timestamps align with demand
    if 'timestamp_utc' in dispatch_wide.columns:
        dispatch_wide_ts_canonical = _canonicalize_timestamps(dispatch_wide['timestamp_utc']).sort_values().unique()
    elif isinstance(dispatch_wide.index, pd.DatetimeIndex):
        dispatch_wide_ts_canonical = _canonicalize_timestamps(dispatch_wide.index).sort_values().unique()
    else:
        raise ValueError("dispatch_wide must have timestamp_utc column or DatetimeIndex")
    
    if not (demand_ts_canonical == dispatch_wide_ts_canonical).all():
        demand_set = set(demand_ts_canonical)
        wide_set = set(dispatch_wide_ts_canonical)
        missing_in_wide = demand_set - wide_set
        extra_in_wide = wide_set - demand_set
        missing_sample = list(missing_in_wide)[:3] if missing_in_wide else []
        extra_sample = list(extra_in_wide)[:3] if extra_in_wide else []
        
        raise ValueError(
            f"dispatch_wide timestamps don't match demand. "
            f"Missing in wide: {len(missing_in_wide)}, Extra in wide: {len(extra_in_wide)}. "
            f"Sample missing: {missing_sample}, Sample extra: {extra_sample}"
        )
    
    # 3. Energy closure at timestep level (within tolerance)
    # Build canonical timestamp columns for consistent grouping and alignment
    demand_df_work = demand_df.copy()
    demand_df_work['_ts'] = _canonicalize_timestamps(demand_df['timestamp_utc'])
    
    dispatch_long_work = dispatch_long.copy()
    dispatch_long_work['_ts'] = _canonicalize_timestamps(dispatch_long['timestamp_utc'])
    
    # Group by canonical timestamps (all indexes will be datetime64[ns, UTC])
    demand_by_ts = demand_df_work.groupby('_ts')['heat_demand_MW'].sum()
    served_by_ts = dispatch_long_work.groupby('_ts')['heat_MW'].sum()
    
    # Get unserved_MW if present (sum across units per timestamp, or use first if stored once per timestamp)
    if 'unserved_MW' in dispatch_long_work.columns:
        # Check if unserved_MW is stored once per timestamp (system-level) or per unit
        # If it's system-level, use first(); if per-unit, use sum()
        unserved_sample = dispatch_long_work.groupby('_ts')['unserved_MW'].apply(lambda x: (x != 0).sum())
        if unserved_sample.max() <= 1:
            # Stored once per timestamp (system-level)
            unserved_by_ts = dispatch_long_work.groupby('_ts')['unserved_MW'].first()
        else:
            # Stored per unit (sum across units)
            unserved_by_ts = dispatch_long_work.groupby('_ts')['unserved_MW'].sum()
    else:
        # Create zero Series with same index as served_by_ts
        unserved_by_ts = pd.Series(0.0, index=served_by_ts.index, dtype=float)
    
    # Force alignment explicitly (prevents tz-naive/aware join errors and enforces strict matching)
    # Reindex to demand_by_ts.index to ensure all Series have the same index
    served_by_ts = served_by_ts.reindex(demand_by_ts.index, fill_value=0.0)
    unserved_by_ts = unserved_by_ts.reindex(demand_by_ts.index, fill_value=0.0)
    
    # Defensive assertions: ensure all indexes are timezone-aware UTC
    assert isinstance(demand_by_ts.index, pd.DatetimeIndex) and demand_by_ts.index.tz is not None, \
        f"demand_by_ts index must be timezone-aware DatetimeIndex, got {type(demand_by_ts.index)} with tz={demand_by_ts.index.tz if hasattr(demand_by_ts.index, 'tz') else 'N/A'}"
    assert isinstance(served_by_ts.index, pd.DatetimeIndex) and served_by_ts.index.tz is not None, \
        f"served_by_ts index must be timezone-aware DatetimeIndex, got {type(served_by_ts.index)} with tz={served_by_ts.index.tz if hasattr(served_by_ts.index, 'tz') else 'N/A'}"
    assert isinstance(unserved_by_ts.index, pd.DatetimeIndex) and unserved_by_ts.index.tz is not None, \
        f"unserved_by_ts index must be timezone-aware DatetimeIndex, got {type(unserved_by_ts.index)} with tz={unserved_by_ts.index.tz if hasattr(unserved_by_ts.index, 'tz') else 'N/A'}"
    
    # Verify all indexes are UTC
    if str(demand_by_ts.index.tz) != 'UTC':
        raise ValueError(f"demand_by_ts index timezone must be UTC, got {demand_by_ts.index.tz}")
    if str(served_by_ts.index.tz) != 'UTC':
        raise ValueError(f"served_by_ts index timezone must be UTC, got {served_by_ts.index.tz}")
    if str(unserved_by_ts.index.tz) != 'UTC':
        raise ValueError(f"unserved_by_ts index timezone must be UTC, got {unserved_by_ts.index.tz}")
    
    # Check closure: demand ≈ served + unserved
    tolerance = 1e-6  # MW
    closure_errors = (demand_by_ts - served_by_ts - unserved_by_ts).abs()
    max_error = closure_errors.max()
    if max_error > tolerance:
        max_error_idx = closure_errors.idxmax()
        raise ValueError(
            f"Energy closure violation: max error = {max_error:.6f} MW at {max_error_idx}. "
            f"demand={demand_by_ts[max_error_idx]:.6f}, served={served_by_ts[max_error_idx]:.6f}, "
            f"unserved={unserved_by_ts[max_error_idx]:.6f}"
        )
    
    # 4. Annual energy totals use dt_h
    demand_GWh = (demand_by_ts.sum() * dt_h) / 1000.0
    served_GWh = (served_by_ts.sum() * dt_h) / 1000.0
    unserved_GWh = (unserved_by_ts.sum() * dt_h) / 1000.0
    
    # Check annual closure (tolerance 0.1 GWh as before)
    annual_error = abs(demand_GWh - served_GWh - unserved_GWh)
    if annual_error > 0.1:
        raise ValueError(
            f"Annual energy closure violation: error = {annual_error:.3f} GWh. "
            f"demand={demand_GWh:.3f}, served={served_GWh:.3f}, unserved={unserved_GWh:.3f}"
        )
    
    print(f"[OK] Dispatch output validation passed: demand={demand_GWh:.3f} GWh, served={served_GWh:.3f} GWh, unserved={unserved_GWh:.3f} GWh")


def _debug_validate_summary_schema(summary: pd.DataFrame) -> None:
    """
    Lightweight self-test helper to validate annual summary schema.
    
    Args:
        summary: Annual summary DataFrame from compute_annual_summary()
    """
    assert 'unit_id' in summary.columns, "unit_id column missing from annual summary"
    assert 'TOTAL' in summary['unit_id'].values, "TOTAL row missing from annual summary"
    assert 'SYSTEM' in summary['unit_id'].values, "SYSTEM row missing from annual summary"
    assert len(summary.columns) == len(set(summary.columns)), "Duplicate column names in annual summary"
    # Additional check: unit_id should not have duplicates (except SYSTEM/TOTAL which are unique)
    unit_id_counts = summary['unit_id'].value_counts()
    assert unit_id_counts['TOTAL'] == 1, "Multiple TOTAL rows found"
    assert unit_id_counts['SYSTEM'] == 1, "Multiple SYSTEM rows found"
    # All other unit_ids should appear once
    other_units = summary[~summary['unit_id'].isin(['SYSTEM', 'TOTAL'])]
    if len(other_units) > 0:
        assert not other_units['unit_id'].duplicated().any(), "Duplicate unit_id values in annual summary"


def _debug_validate_online_cost_fallback(dispatch_long: pd.DataFrame, util_df: pd.DataFrame, signals,
                                         commitment_block_hours: int, no_load_cost_nzd_per_h: float) -> None:
    """
    Regression check: verify online_cost fallback works when unit_online is missing.
    
    Tests that recompute_costs_optimal() with online_cost_applies_when='online_only'
    produces non-zero online_cost_nzd when unit_on has any ones, even if unit_online is missing.
    
    Args:
        dispatch_long: Long-form dispatch DataFrame
        util_df: Utilities DataFrame
        signals: Signals dict
        commitment_block_hours: Block size for startup cost allocation
        no_load_cost_nzd_per_h: No-load cost per hour
    """
    # Check if no-load cost is missing or zero - skip assertion if so
    if 'no_load_cost_nzd_per_h' not in util_df.columns:
        print("[INFO] Skipping online_cost check: no_load_cost_nzd_per_h missing in utilities.")
        return
    
    if util_df['no_load_cost_nzd_per_h'].isna().all() or (util_df['no_load_cost_nzd_per_h'] == 0).all():
        print("[INFO] Skipping online_cost check: no_load_cost_nzd_per_h is zero or missing.")
        return
    
    # Also check the passed parameter
    if no_load_cost_nzd_per_h == 0.0:
        print("[INFO] Skipping online_cost check: no_load_cost_nzd_per_h parameter is zero.")
        return
    
    # Create a copy without unit_online (simulate older CSV)
    dispatch_long_test = dispatch_long.copy()
    if 'unit_online' in dispatch_long_test.columns:
        dispatch_long_test = dispatch_long_test.drop(columns=['unit_online'])
    
    # Recompute costs with online_only mode
    # Infer dt_h from timestamps (default 1.0 if not available)
    if 'timestamp_utc' in dispatch_long_test.columns:
        if pd.api.types.is_datetime64_any_dtype(dispatch_long_test['timestamp_utc']):
            time_deltas = dispatch_long_test['timestamp_utc'].diff().dropna()
            if len(time_deltas) > 0:
                dt_h_test = (time_deltas.dt.total_seconds() / 3600.0).mode()[0] if len(time_deltas.mode()) > 0 else 1.0
            else:
                dt_h_test = 1.0
        else:
            dt_h_test = 1.0
    else:
        dt_h_test = 1.0
    
    dispatch_long_recomputed = recompute_costs_optimal(
        dispatch_long_test, util_df, signals, commitment_block_hours,
        unserved_penalty_nzd_per_MWh=50000.0,
        reserve_penalty_nzd_per_MWh=2000.0,
        no_load_cost_nzd_per_h=no_load_cost_nzd_per_h,
        online_cost_applies_when='online_only',
        dt_h=dt_h_test
    )
    
    # Assert online_cost_nzd is not all zeros when unit_on (or unit_online) has any ones
    if 'online_cost_nzd' in dispatch_long_recomputed.columns:
        # Check if any unit is online (unit_online if available, else unit_on)
        if 'unit_online' in dispatch_long_recomputed.columns:
            has_online = (dispatch_long_recomputed['unit_online'] == 1).any()
        elif 'unit_on' in dispatch_long_recomputed.columns:
            has_online = (dispatch_long_recomputed['unit_on'] == 1).any()
        else:
            has_online = False
        
        has_online_cost = (dispatch_long_recomputed['online_cost_nzd'] != 0).any()
        assert has_online_cost or not has_online, (
            "Regression: online_cost_nzd is all zeros despite units being online. "
            "Fallback to unit_on for online_only mode may be broken."
        )


def _debug_validate_hourly_cost_storage(dispatch_long: pd.DataFrame) -> None:
    """
    Lightweight self-test helper to validate hourly cost storage convention.
    
    Validates that system-level costs (online_cost_nzd, unserved_cost_nzd, reserve_penalty_cost_nzd)
    are stored at most once per timestamp (on the first unit row).
    
    Args:
        dispatch_long: Long-form dispatch DataFrame
    """
    # Ensure timestamp_utc is datetime for grouping
    dispatch_long_for_check = dispatch_long.copy()
    if not pd.api.types.is_datetime64_any_dtype(dispatch_long_for_check['timestamp_utc']):
        dispatch_long_for_check['timestamp_utc'] = parse_any_timestamp(dispatch_long_for_check['timestamp_utc'])
    
    # Check each system-level cost column
    cost_columns = ['online_cost_nzd', 'unserved_cost_nzd', 'reserve_penalty_cost_nzd']
    for col in cost_columns:
        if col in dispatch_long_for_check.columns:
            # Count non-zero values per timestamp
            counts = dispatch_long_for_check.groupby('timestamp_utc')[col].apply(
                lambda s: (s != 0).sum()
            )
            max_count = counts.max()
            assert max_count <= 1, (
                f"{col} appears on >1 row for some timestamp (max: {max_count}). "
                f"Violations at: {counts[counts > 1].index.tolist()}"
            )
    
    # Also check MW fields (optional but useful)
    mw_columns = ['unserved_MW', 'reserve_shortfall_MW']
    for col in mw_columns:
        if col in dispatch_long_for_check.columns:
            counts = dispatch_long_for_check.groupby('timestamp_utc')[col].apply(
                lambda s: (s != 0).sum()
            )
            max_count = counts.max()
            assert max_count <= 1, (
                f"{col} appears on >1 row for some timestamp (max: {max_count}). "
                f"Violations at: {counts[counts > 1].index.tolist()}"
            )


def _self_test_lightweight():
    """
    Lightweight self-test helper for dt_h and dispatch validation.
    
    Tests:
    1. Load hourly_heat_demand_2020.csv and compute dt_h (should be 1.0)
    2. Run proportional mode and validate outputs
    3. Run optimal_subset mode for first 48 hours and validate outputs
    """
    print("="*60)
    print("Running lightweight self-test...")
    print("="*60)
    
    ROOT = repo_root()
    INPUT_DIR = input_root()
    
    # Test 1: Load demand and compute dt_h
    demand_csv = INPUT_DIR / 'site' / 'site_annual.csv'  # Try this first, or use a known path
    # Alternative: look for hourly_heat_demand_2020.csv in Output/latest
    output_latest = ROOT / 'Output' / 'latest' / 'demandpack'
    if (output_latest / 'hourly_heat_demand_2020.csv').exists():
        demand_csv = output_latest / 'hourly_heat_demand_2020.csv'
    elif (ROOT / 'Output' / 'hourly_heat_demand_2020.csv').exists():
        demand_csv = ROOT / 'Output' / 'hourly_heat_demand_2020.csv'
    else:
        print("[SKIP] Self-test: hourly_heat_demand_2020.csv not found, skipping")
        return
    
    print(f"Test 1: Loading {demand_csv}...")
    demand_df, dt_h = load_hourly_demand(str(demand_csv))
    assert abs(dt_h - 1.0) < 1e-6, f"Expected dt_h=1.0 for hourly data, got {dt_h}"
    print(f"[OK] dt_h = {dt_h:.6f} hours (expected 1.0)")
    
    # Test 2: Proportional mode (first 48 hours for speed)
    print("\nTest 2: Proportional dispatch (first 48 hours)...")
    demand_slice = demand_df.head(48).copy()
    utilities_csv = INPUT_DIR / 'site' / 'utilities' / 'site_utilities_2020.csv'
    if not utilities_csv.exists():
        print("[SKIP] Self-test: site_utilities_2020.csv not found, skipping proportional test")
        return
    
    util_df = load_utilities(str(utilities_csv), epoch=2020)
    dispatch_long, dispatch_wide = allocate_baseline_dispatch(demand_slice, util_df, dt_h)
    
    # Load signals for cost computation
    signals_config_path = INPUT_DIR / 'signals' / 'signals_config.toml'
    if not signals_config_path.exists():
        print("[SKIP] Self-test: signals_config.toml not found, skipping cost computation and validation")
        return
    
    signals_config = load_signals_config(str(signals_config_path))
    signals = get_signals_for_epoch(signals_config, '2020')
    
    # Add costs before validation (required for schema consistency)
    dispatch_long = add_costs_to_dispatch(dispatch_long, util_df, signals, dt_h)
    validate_dispatch_outputs(dispatch_long, dispatch_wide, demand_slice, dt_h)
    print("[OK] Proportional dispatch validation passed")
    
    # Test 3: Optimal subset mode (first 48 hours)
    print("\nTest 3: Optimal subset dispatch (first 48 hours)...")
    
    dispatch_long_opt, dispatch_wide_opt = allocate_dispatch_optimal_subset(
        demand_slice, util_df, signals,
        commitment_block_hours=24,
        unserved_penalty_nzd_per_MWh=50000.0,
        reserve_frac=0.0,
        reserve_penalty_nzd_per_MWh=2000.0,
        no_load_cost_nzd_per_h=50.0,
        online_cost_applies_when='online_only',
        dt_h=dt_h
    )
    # Ensure costs are added (for schema consistency, even though optimal_subset already has costs)
    dispatch_long_opt = add_costs_to_dispatch(dispatch_long_opt, util_df, signals, dt_h)
    validate_dispatch_outputs(dispatch_long_opt, dispatch_wide_opt, demand_slice, dt_h)
    print("[OK] Optimal subset dispatch validation passed")
    
    print("\n" + "="*60)
    print("All self-tests passed!")
    print("="*60)


if __name__ == '__main__':
    import sys
    if '--self-test' in sys.argv:
        _self_test_lightweight()
        sys.exit(0)
    main()

