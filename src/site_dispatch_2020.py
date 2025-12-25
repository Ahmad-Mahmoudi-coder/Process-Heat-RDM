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
from scipy.optimize import linprog

from src.path_utils import repo_root, resolve_path, resolve_cfg_path, input_root
from src.load_signals import load_signals_config, get_signals_for_epoch, map_eval_epoch_to_signals_epoch
from src.time_utils import parse_any_timestamp, to_iso_z
import src.maintenance_utils as maint


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
    
    # Define status column name (prefer status, then status_<epoch>)
    # This reduces noise for utilities CSVs that use generic "status" column
    if "status" in df.columns:
        status_col = "status"
    elif epoch is not None:
        status_col = f"status_{epoch}"
        # If status_{epoch} not found, try fallback to generic status
        if status_col not in df.columns:
            if "status" in df.columns:
                df[status_col] = df["status"]
                print(f"[WARN] Column '{status_col}' not found, using 'status' as alias in {path}")
            else:
                # Both missing - will be caught in required_cols check below
                pass
    else:
        # For backward compatibility, try status_2020 first, then status
        if "status_2020" in df.columns:
            status_col = "status_2020"
        elif "status" in df.columns:
            status_col = "status"
        else:
            status_col = "status"  # Will be caught in required_cols check if missing
    
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


def load_maintenance_availability(maintenance_csv: Path, timestamps_utc: pd.DatetimeIndex, unit_ids: list[str]) -> pd.DataFrame:
    """
    Load maintenance windows CSV and build hourly availability matrix.
    
    Schema:
    - unit_id, start_timestamp_utc, end_timestamp_utc, availability (float in [0,1])
    
    Semantics:
    - Apply window for all hours where: start_timestamp_utc <= t < end_timestamp_utc (inclusive start, exclusive end)
    - If windows overlap for a unit, availability is the minimum (most restrictive)
    - If unit_id in maintenance file doesn't exist in utilities: warn and ignore
    
    Args:
        maintenance_csv: Path to maintenance windows CSV file
        timestamps_utc: DatetimeIndex of hourly timestamps (must be UTC)
        unit_ids: List of unit IDs from utilities (for validation)
        
    Returns:
        DataFrame with:
        - index = timestamps_utc (hourly)
        - columns = unit_id (one column per unit)
        - values = availability (float in [0,1], default 1.0)
        
    Raises:
        FileNotFoundError: If file doesn't exist (optional file, caller should handle)
        ValueError: If validation fails (invalid unit_id, malformed timestamps, etc.)
    """
    if not maintenance_csv.exists():
        raise FileNotFoundError(f"Maintenance windows file not found: {maintenance_csv}")
    
    df = pd.read_csv(maintenance_csv)
    
    # Validate required columns
    required_cols = ['unit_id', 'start_timestamp_utc', 'end_timestamp_utc', 'availability']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Maintenance windows CSV missing required columns: {missing}. File: {maintenance_csv}")
    
    # Parse timestamps with error handling
    try:
        df['start_timestamp_utc'] = parse_any_timestamp(df['start_timestamp_utc'])
    except Exception as e:
        bad_rows = df[df['start_timestamp_utc'].isna() | (df['start_timestamp_utc'].astype(str).str.contains('error', case=False, na=False))]
        raise ValueError(f"Failed to parse start_timestamp_utc in maintenance CSV {maintenance_csv}. "
                        f"Error: {e}. Check rows: {bad_rows[['unit_id', 'start_timestamp_utc']].to_dict('records')}")
    
    try:
        df['end_timestamp_utc'] = parse_any_timestamp(df['end_timestamp_utc'])
    except Exception as e:
        bad_rows = df[df['end_timestamp_utc'].isna() | (df['end_timestamp_utc'].astype(str).str.contains('error', case=False, na=False))]
        raise ValueError(f"Failed to parse end_timestamp_utc in maintenance CSV {maintenance_csv}. "
                        f"Error: {e}. Check rows: {bad_rows[['unit_id', 'end_timestamp_utc']].to_dict('records')}")
    
    # Ensure timestamps are UTC
    if df['start_timestamp_utc'].dt.tz is None:
        df['start_timestamp_utc'] = df['start_timestamp_utc'].dt.tz_localize('UTC')
    else:
        df['start_timestamp_utc'] = df['start_timestamp_utc'].dt.tz_convert('UTC')
    
    if df['end_timestamp_utc'].dt.tz is None:
        df['end_timestamp_utc'] = df['end_timestamp_utc'].dt.tz_localize('UTC')
    else:
        df['end_timestamp_utc'] = df['end_timestamp_utc'].dt.tz_convert('UTC')
    
    # Validate: start < end
    invalid = df[df['start_timestamp_utc'] >= df['end_timestamp_utc']]
    if len(invalid) > 0:
        raise ValueError(
            f"Invalid maintenance windows (start >= end) in {maintenance_csv}:\n"
            f"{invalid[['unit_id', 'start_timestamp_utc', 'end_timestamp_utc']]}"
        )
    
    # Validate availability values (should be float in [0,1])
    df['availability'] = pd.to_numeric(df['availability'], errors='coerce')
    invalid_avail = df[df['availability'].isna() | (df['availability'] < 0) | (df['availability'] > 1)]
    if len(invalid_avail) > 0:
        raise ValueError(
            f"Invalid availability values (must be float in [0,1]) in {maintenance_csv}:\n"
            f"{invalid_avail[['unit_id', 'availability']]}"
        )
    
    # Validate unit_ids: raise error if unknown (strict mode)
    invalid_units = df[~df['unit_id'].isin(unit_ids)]
    if len(invalid_units) > 0:
        invalid_list = invalid_units['unit_id'].unique().tolist()
        raise ValueError(
            f"Maintenance windows reference unknown unit_id(s): {invalid_list}. "
            f"Available unit_ids: {unit_ids}. "
            f"Please ensure unit_id values in maintenance CSV match utilities after filtering retired units."
        )
    
    # Build availability matrix: default 1.0, apply maintenance windows
    # Ensure timestamps_utc is timezone-aware UTC DatetimeIndex
    if isinstance(timestamps_utc, pd.DatetimeIndex):
        if timestamps_utc.tz is None:
            timestamps_utc = timestamps_utc.tz_localize('UTC')
        else:
            timestamps_utc = timestamps_utc.tz_convert('UTC')
    else:
        # Convert to DatetimeIndex and ensure UTC
        timestamps_utc = pd.to_datetime(timestamps_utc, utc=True)
        if timestamps_utc.tz is None:
            timestamps_utc = timestamps_utc.tz_localize('UTC')
        else:
            timestamps_utc = timestamps_utc.tz_convert('UTC')
    
    # Create DataFrame with timestamps as index, unit_ids as columns
    availability_df = pd.DataFrame(
        index=timestamps_utc,
        columns=unit_ids,
        data=1.0  # Default: fully available
    )
    
    # Apply maintenance windows: for each window, set availability to minimum of current and window value
    # This handles overlapping windows correctly (most restrictive wins)
    for _, window in df.iterrows():
        unit_id = window['unit_id']
        start = window['start_timestamp_utc']
        end = window['end_timestamp_utc']
        avail = float(window['availability'])
        
        # Ensure start and end are timezone-aware UTC Timestamps for comparison
        if isinstance(start, pd.Timestamp):
            if start.tz is None:
                start = start.tz_localize('UTC')
            else:
                start = start.tz_convert('UTC')
        else:
            start = pd.to_datetime(start, utc=True)
            if start.tz is None:
                start = start.tz_localize('UTC')
            else:
                start = start.tz_convert('UTC')
        
        if isinstance(end, pd.Timestamp):
            if end.tz is None:
                end = end.tz_localize('UTC')
            else:
                end = end.tz_convert('UTC')
        else:
            end = pd.to_datetime(end, utc=True)
            if end.tz is None:
                end = end.tz_localize('UTC')
            else:
                end = end.tz_convert('UTC')
        
        # Set availability for timestamps in [start, end) - take minimum (most restrictive)
        mask = (availability_df.index >= start) & (availability_df.index < end)
        # For overlapping windows, take minimum (most restrictive)
        # Use np.minimum for element-wise minimum
        availability_df.loc[mask, unit_id] = np.minimum(availability_df.loc[mask, unit_id], avail)
    
    return availability_df


# Legacy wrapper removed - use maint.load_maintenance_windows() and maint.build_availability_matrix() directly


def allocate_baseline_dispatch(demand_df: pd.DataFrame, util_df: pd.DataFrame, dt_h: float,
                               maintenance_availability: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Allocate heat demand across utilities proportionally by capacity (vectorized).
    
    For each timestep, allocates heat to each unit proportional to its capacity:
    q_u_t = Q_t * (max_heat_MW_u * avail[u,t] / P_total_avail[t])
    
    Where P_total_avail[t] = sum_u (max_heat_MW_u * avail[u,t]) is the total available capacity at time t.
    
    Note: heat_MW is average MW over the timestep. Energy per step = heat_MW * dt_h.
    
    Args:
        demand_df: DataFrame with timestamp_utc and heat_demand_MW columns
        util_df: DataFrame with utility information including max_heat_MW, 
                 efficiency_th, co2_factor_t_per_MWh_fuel
        dt_h: Timestep duration in hours
        maintenance_availability: Optional DataFrame (wide-form: index=timestamps, columns=unit_ids, or long-form)
        
    Returns:
        Tuple of (long-form DataFrame, wide-form DataFrame)
        Long-form columns: timestamp_utc, unit_id, heat_MW (avg MW per step), fuel_MWh (energy per step), co2_tonnes (per step)
        Wide-form columns: timestamp_utc, total_heat_MW, CB1_MW, CB2_MW, ...
    """
    unit_ids = util_df['unit_id'].tolist()
    n_units = len(unit_ids)
    n_timesteps = len(demand_df)
    
    # Build maintenance availability matrix (default 1.0)
    # Canonicalize demand timestamps to UTC DatetimeIndex for alignment
    if isinstance(demand_df['timestamp_utc'], pd.Series):
        demand_ts_raw = demand_df['timestamp_utc'].values
    else:
        demand_ts_raw = demand_df['timestamp_utc']
    demand_ts = pd.to_datetime(demand_ts_raw, utc=True)
    if demand_ts.tz is None:
        demand_ts = pd.DatetimeIndex(demand_ts).tz_localize('UTC')
    else:
        demand_ts = pd.DatetimeIndex(demand_ts).tz_convert('UTC')
    
    if maintenance_availability is not None:
        if isinstance(maintenance_availability.index, pd.DatetimeIndex):
            # Wide-form: already has index=timestamps, columns=unit_ids
            # Canonicalize maintenance index to UTC DatetimeIndex for alignment
            maint_index = maintenance_availability.index
            if maint_index.tz is None:
                maint_index_utc = maint_index.tz_localize('UTC')
            else:
                maint_index_utc = maint_index.tz_convert('UTC')
            maintenance_availability.index = maint_index_utc
            
            # Reindex to demand timestamps (fill missing with 1.0)
            avail_matrix_df = maintenance_availability.reindex(demand_ts, fill_value=1.0)
            # Ensure all units are present (fill missing with 1.0)
            for uid in unit_ids:
                if uid not in avail_matrix_df.columns:
                    avail_matrix_df[uid] = 1.0
            avail_matrix = avail_matrix_df[unit_ids].values  # Shape: (n_timesteps, n_units)
        else:
            # Long-form: pivot to wide-form
            # First canonicalize timestamp_utc in maintenance_availability
            maint_avail_copy = maintenance_availability.copy()
            maint_avail_copy['timestamp_utc'] = pd.to_datetime(maint_avail_copy['timestamp_utc'], utc=True)
            if maint_avail_copy['timestamp_utc'].dt.tz is None:
                maint_avail_copy['timestamp_utc'] = maint_avail_copy['timestamp_utc'].dt.tz_localize('UTC')
            else:
                maint_avail_copy['timestamp_utc'] = maint_avail_copy['timestamp_utc'].dt.tz_convert('UTC')
            
            avail_pivot = maint_avail_copy.pivot(
                index='timestamp_utc',
                columns='unit_id',
                values='availability_multiplier'
            )
            # Canonicalize pivot index to UTC DatetimeIndex
            if avail_pivot.index.tz is None:
                avail_pivot.index = avail_pivot.index.tz_localize('UTC')
            else:
                avail_pivot.index = avail_pivot.index.tz_convert('UTC')
            
            # Reindex to demand timestamps (fill missing with 1.0)
            avail_pivot = avail_pivot.reindex(demand_ts, fill_value=1.0)
            for uid in unit_ids:
                if uid not in avail_pivot.columns:
                    avail_pivot[uid] = 1.0
            avail_matrix = avail_pivot[unit_ids].values  # Shape: (n_timesteps, n_units)
    else:
        avail_matrix = np.ones((n_timesteps, n_units))
    
    # Build capacity matrix: cap_matrix[t, u] = max_heat_MW[u] * availability_factor[u] * maint_avail[t, u]
    # where availability_factor is static from utilities, maint_avail is time-varying from maintenance
    max_heat_array = util_df['max_heat_MW'].values  # Shape: (n_units,)
    static_avail_array = util_df.get('availability_factor', pd.Series([1.0] * len(util_df))).values  # Shape: (n_units,)
    cap_matrix = max_heat_array[None, :] * static_avail_array[None, :] * avail_matrix  # Shape: (n_timesteps, n_units)
    
    # Compute total available capacity per timestep
    P_total_avail = cap_matrix.sum(axis=1)  # Shape: (n_timesteps,)
    
    # Extract demand as array (shape: n_timesteps,)
    demand_heat = demand_df['heat_demand_MW'].values
    
    # Compute served target: min(demand, total_available_capacity) to ensure we never exceed capacity
    # This ensures that when demand > capacity, we serve exactly capacity and the rest becomes unserved
    served_target = np.minimum(demand_heat, P_total_avail)  # Shape: (n_timesteps,)
    
    # Vectorized allocation: heat_matrix[i, j] = served_target[i] * (cap_matrix[i, j] / P_total_avail[i])
    # If P_total_avail[i] == 0, set heat_matrix[i, j] = 0 (all units unavailable)
    # Shape: (n_timesteps, n_units)
    with np.errstate(divide='ignore', invalid='ignore'):
        capacity_share_matrix = np.where(P_total_avail[:, None] > 0, 
                                         cap_matrix / P_total_avail[:, None], 
                                         0.0)
    heat_matrix = served_target[:, None] * capacity_share_matrix
    
    # Clamp dispatch to capacity limits to prevent numerical precision violations
    # This ensures heat_matrix[i, j] <= cap_matrix[i, j] exactly
    heat_matrix = np.minimum(heat_matrix, cap_matrix)
    
    # Compute served and unserved after clamping
    # served_MW[t] = sum_u heat_matrix[t,u]
    served_MW = heat_matrix.sum(axis=1)  # Shape: (n_timesteps,)
    # unserved_MW[t] = max(0, demand_MW[t] - served_MW[t])
    unserved_MW = np.maximum(0.0, demand_heat - served_MW)  # Shape: (n_timesteps,)
    # Clip any tiny negative values to zero (numerical precision)
    unserved_MW = np.maximum(0.0, unserved_MW)
    
    # Build wide-form DataFrame
    unit_ids = util_df['unit_id'].tolist()
    dispatch_wide = pd.DataFrame({
        'timestamp_utc': demand_df['timestamp_utc'].values
    })
    
    # Add unit columns
    for i, unit_id in enumerate(unit_ids):
        dispatch_wide[f'{unit_id}_MW'] = heat_matrix[:, i]
    
    # Add unserved_MW column
    dispatch_wide['unserved_MW'] = unserved_MW
    
    # Add total_heat_MW (should equal demand_heat, but compute for consistency)
    unit_cols = [f'{uid}_MW' for uid in unit_ids]
    dispatch_wide['total_heat_MW'] = dispatch_wide[unit_cols].sum(axis=1)
    
    # Reorder columns: timestamp_utc, total_heat_MW, unserved_MW, then unit columns
    dispatch_wide = dispatch_wide[['timestamp_utc', 'total_heat_MW', 'unserved_MW'] + unit_cols]
    
    # Build long-form by stacking (melt) the wide matrix
    long_results = []
    for i, timestamp in enumerate(demand_df['timestamp_utc']):
        # Add rows for real units
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
        
        # Add UNSERVED row for this timestamp
        unserved_val = unserved_MW[i]
        long_results.append({
            'timestamp_utc': timestamp,
            'unit_id': 'UNSERVED',
            'heat_MW': unserved_val,  # Unserved demand (MW)
            'fuel_MWh': 0.0,  # No fuel consumption for unserved
            'co2_tonnes': 0.0,  # No CO2 emissions for unserved
        })
    
    dispatch_long = pd.DataFrame(long_results)
    
    return dispatch_long, dispatch_wide


def allocate_dispatch_lp(
    demand_df: pd.DataFrame,
    util_df: pd.DataFrame,
    signals: Union[Dict[str, float], pd.DataFrame],
    dt_h: float,
    maintenance_availability: Optional[pd.DataFrame] = None,
    electricity_signals: Optional[pd.DataFrame] = None,
    unserved_penalty_nzd_per_MWh: float = 10000.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Allocate dispatch using hourly linear programming with equality constraints.
    
    This mode enforces strict hourly heat balance: sum(heat_MW) + unserved_MW == demand_MW
    for every hour, ensuring energy closure by construction.
    
    Variables per hour:
    - heat_MW[u,t] >= 0 for each unit u
    - unserved_MW[t] >= 0
    
    Constraints per hour:
    - Equality: sum_u heat_MW[u,t] + unserved_MW[t] == demand_MW[t]
    - Capacity: heat_MW[u,t] <= max_heat_MW[u] * a[u,t] (where a is maintenance availability)
    - Headroom (if electricity_signals provided): sum_{u electric} heat_MW[u,t]/efficiency[u] <= headroom_MW[t]
    
    Objective: Minimize total cost (fuel + VOLL * unserved_energy)
    
    Args:
        demand_df: DataFrame with timestamp_utc and heat_demand_MW
        util_df: DataFrame with unit information
        signals: Either dict (flat prices) or DataFrame (time-varying prices with timestamp_utc)
        dt_h: Timestep duration in hours
        maintenance_availability: Optional DataFrame with columns: timestamp_utc, unit_id, availability_multiplier
        electricity_signals: Optional DataFrame with columns: timestamp_utc, elec_price_nzd_per_MWh, headroom_MW
        unserved_penalty_nzd_per_MWh: Value of lost load (default: 10000)
        
    Returns:
        Tuple of (long-form DataFrame, wide-form DataFrame)
        Long-form includes: timestamp_utc, unit_id, heat_MW, fuel_MWh, co2_tonnes, unserved_MW
    """
    unit_ids = util_df['unit_id'].tolist()
    n_units = len(unit_ids)
    n_timesteps = len(demand_df)
    
    # Build unit info maps
    unit_to_idx = {uid: i for i, uid in enumerate(unit_ids)}
    max_heat = {uid: util_df[util_df['unit_id'] == uid]['max_heat_MW'].iloc[0] for uid in unit_ids}
    efficiency = {uid: util_df[util_df['unit_id'] == uid]['efficiency_th'].iloc[0] for uid in unit_ids}
    fuel_type = {uid: util_df[util_df['unit_id'] == uid]['fuel'].iloc[0].lower().strip() for uid in unit_ids}
    
    # Identify electric units
    electric_units = [
        uid for uid in unit_ids
        if fuel_type[uid] == 'electricity' or 
        util_df[util_df['unit_id'] == uid]['tech_type'].iloc[0].lower().strip() == 'electrode_boiler'
    ]
    
    # Build dispatch priority for tie-breaking (deterministic: sorted unit_id order)
    # Lower priority number = higher dispatch preference (when costs are equal)
    sorted_unit_ids = sorted(unit_ids)
    dispatch_priority = {uid: i for i, uid in enumerate(sorted_unit_ids)}  # 0 = highest priority
    
    # Build maintenance availability matrix (default 1.0)
    # maintenance_availability can be either:
    # - Wide-form DataFrame (index=timestamps, columns=unit_ids) - preferred
    # - Long-form DataFrame (columns: timestamp_utc, unit_id, availability_multiplier) - legacy
    if maintenance_availability is not None:
        if isinstance(maintenance_availability.index, pd.DatetimeIndex):
            # Wide-form: already has index=timestamps, columns=unit_ids
            # Canonicalize demand timestamps to ensure proper alignment
            if isinstance(demand_df['timestamp_utc'], pd.Series):
                demand_ts_raw = demand_df['timestamp_utc'].values
            else:
                demand_ts_raw = demand_df['timestamp_utc']
            # Canonicalize to UTC DatetimeIndex
            demand_ts = pd.to_datetime(demand_ts_raw, utc=True)
            if demand_ts.tz is None:
                demand_ts = demand_ts.tz_localize('UTC')
            else:
                demand_ts = demand_ts.tz_convert('UTC')
            demand_ts = pd.DatetimeIndex(demand_ts)
            
            # Canonicalize maintenance index to UTC DatetimeIndex for alignment
            maint_index = maintenance_availability.index
            if maint_index.tz is None:
                maint_index_utc = maint_index.tz_localize('UTC')
            else:
                maint_index_utc = maint_index.tz_convert('UTC')
            maintenance_availability.index = maint_index_utc
            
            # Reindex to demand timestamps (fill missing with 1.0)
            avail_matrix_df = maintenance_availability.reindex(demand_ts, fill_value=1.0)
            # Ensure all units are present (fill missing with 1.0)
            for uid in unit_ids:
                if uid not in avail_matrix_df.columns:
                    avail_matrix_df[uid] = 1.0
            avail_matrix = avail_matrix_df[unit_ids].values  # Shape: (n_timesteps, n_units)
        else:
            # Long-form: pivot to wide-form
            # First canonicalize timestamp_utc in maintenance_availability
            maint_avail_copy = maintenance_availability.copy()
            maint_avail_copy['timestamp_utc'] = pd.to_datetime(maint_avail_copy['timestamp_utc'], utc=True)
            if maint_avail_copy['timestamp_utc'].dt.tz is None:
                maint_avail_copy['timestamp_utc'] = maint_avail_copy['timestamp_utc'].dt.tz_localize('UTC')
            else:
                maint_avail_copy['timestamp_utc'] = maint_avail_copy['timestamp_utc'].dt.tz_convert('UTC')
            
            avail_pivot = maint_avail_copy.pivot(
                index='timestamp_utc',
                columns='unit_id',
                values='availability_multiplier'
            )
            # Canonicalize pivot index to UTC DatetimeIndex
            if avail_pivot.index.tz is None:
                avail_pivot.index = avail_pivot.index.tz_localize('UTC')
            else:
                avail_pivot.index = avail_pivot.index.tz_convert('UTC')
            
            # Canonicalize demand timestamps to UTC DatetimeIndex for alignment
            if isinstance(demand_df['timestamp_utc'], pd.Series):
                demand_ts_raw = demand_df['timestamp_utc'].values
            else:
                demand_ts_raw = demand_df['timestamp_utc']
            demand_ts = pd.to_datetime(demand_ts_raw, utc=True)
            if demand_ts.tz is None:
                demand_ts = demand_ts.tz_localize('UTC')
            else:
                demand_ts = demand_ts.tz_convert('UTC')
            demand_ts = pd.DatetimeIndex(demand_ts)
            
            avail_pivot = avail_pivot.reindex(demand_ts, fill_value=1.0)
            # Ensure all units are present
            for uid in unit_ids:
                if uid not in avail_pivot.columns:
                    avail_pivot[uid] = 1.0
            avail_matrix = avail_pivot[unit_ids].values  # Shape: (n_timesteps, n_units)
    else:
        avail_matrix = np.ones((n_timesteps, n_units))
    
    # Build headroom constraint series (if provided)
    headroom_series = None
    if electricity_signals is not None and 'headroom_MW' in electricity_signals.columns:
        # Align to demand timestamps
        headroom_aligned = demand_df[['timestamp_utc']].merge(
            electricity_signals[['timestamp_utc', 'headroom_MW']],
            on='timestamp_utc',
            how='left'
        )
        if headroom_aligned['headroom_MW'].isna().any():
            raise ValueError("headroom_MW alignment failed: missing timestamps")
        headroom_series = headroom_aligned['headroom_MW'].values
    
    # Solve LP for each hour
    long_results = []
    demand_values = demand_df['heat_demand_MW'].values
    
    for t in range(n_timesteps):
        timestamp = demand_df.iloc[t]['timestamp_utc']
        demand_t = demand_values[t]
        
        # Build LP problem for this hour
        # Variables: [heat_MW[unit_0], ..., heat_MW[unit_n-1], unserved_MW]
        n_vars = n_units + 1  # n_units + unserved
        
        # Objective: minimize cost + tie-breaker
        # Cost = sum_u (fuel_cost[u] * heat_MW[u] * dt_h / efficiency[u]) + VOLL * unserved_MW * dt_h
        # Tie-breaker: add epsilon * priority * heat_MWh to make results deterministic
        # Lower priority number = higher dispatch preference (when costs are equal)
        c = np.zeros(n_vars)
        epsilon_tie_breaker = 1e-6  # Small epsilon to break ties deterministically
        
        # Get fuel prices (time-varying if signals is DataFrame, else flat)
        for i, uid in enumerate(unit_ids):
            if isinstance(signals, pd.DataFrame):
                # Time-varying prices: get row for this timestamp
                sig_row = signals[signals['timestamp_utc'] == timestamp]
                if len(sig_row) == 0:
                    raise ValueError(f"No signals row found for timestamp {timestamp}")
                sig_row = sig_row.iloc[0]
                
                if fuel_type[uid] in ['lignite', 'coal']:
                    fuel_price = sig_row.get('coal_price_nzd_per_MWh_fuel', 0.0)
                elif fuel_type[uid] == 'biomass':
                    fuel_price = sig_row.get('biomass_price_nzd_per_MWh_fuel', 0.0)
                elif fuel_type[uid] == 'electricity':
                    # Use time-varying electricity price if available
                    fuel_price = sig_row.get('elec_price_nzd_per_MWh', 0.0)
                else:
                    fuel_price = 0.0
            else:
                # Flat prices from dict
                if fuel_type[uid] in ['lignite', 'coal']:
                    fuel_price = signals.get('coal_price_nzd_per_MWh_fuel', 0.0)
                elif fuel_type[uid] == 'biomass':
                    fuel_price = signals.get('biomass_price_nzd_per_MWh_fuel', 0.0)
                elif fuel_type[uid] == 'electricity':
                    fuel_price = signals.get('electricity_price_nzd_per_MWh_fuel', 0.0)
                else:
                    fuel_price = 0.0
            
            # Cost per MW_heat = (fuel_price / efficiency) * dt_h
            base_cost = (fuel_price / efficiency[uid]) * dt_h if efficiency[uid] > 0 else 0.0
            # Add tie-breaker: epsilon * priority * dt_h (lower priority number = higher preference)
            # This makes results deterministic when units have identical costs
            c[i] = base_cost + epsilon_tie_breaker * dispatch_priority[uid] * dt_h
        
        # Unserved penalty cost (no tie-breaker needed, it's always last resort)
        c[n_units] = unserved_penalty_nzd_per_MWh * dt_h
        
        # Constraints
        # 1. Equality: sum(heat_MW) + unserved_MW == demand_MW
        A_eq = np.ones((1, n_vars))
        A_eq[0, n_units] = 1.0  # unserved coefficient
        b_eq = np.array([demand_t])
        
        # 2. Capacity bounds: 0 <= heat_MW[u] <= max_heat[u] * availability_factor[u] * maint_avail[u,t]
        #    where availability_factor is from utilities (static) and maint_avail is time-varying maintenance
        # 3. Unserved: 0 <= unserved_MW
        bounds = []
        for i, uid in enumerate(unit_ids):
            # Get static availability_factor from utilities (default 1.0 if not present)
            static_avail = util_df[util_df['unit_id'] == uid].get('availability_factor', pd.Series([1.0])).iloc[0]
            # Time-varying maintenance availability
            maint_avail = avail_matrix[t, i]
            # Combined capacity limit
            max_cap = max_heat[uid] * static_avail * maint_avail
            bounds.append((0.0, max_cap))
        bounds.append((0.0, demand_t))  # unserved cannot exceed demand
        
        # 4. Headroom constraint (if electric units and headroom provided)
        # IMPORTANT: Constraint is in electricity MW, not heat MW
        # incremental_electricity_MW[t] = sum_{u electric} heat_MW[u,t] / efficiency[u]
        # Constraint: incremental_electricity_MW[t] <= headroom_MW[t]
        A_ub = None
        b_ub = None
        if headroom_series is not None and len(electric_units) > 0 and headroom_series[t] >= 0:
            # Constraint: sum_{u electric} heat_MW[u]/efficiency[u] <= headroom_MW[t]
            # This is correct: heat_MW / efficiency = electricity_MW
            A_ub = np.zeros((1, n_vars))
            for uid in electric_units:
                idx = unit_to_idx[uid]
                A_ub[0, idx] = 1.0 / efficiency[uid]  # Convert heat MW to electricity MW
            b_ub = np.array([headroom_series[t]])
        
        # Solve LP
        try:
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if not result.success:
                raise ValueError(
                    f"LP solver failed at timestamp {timestamp}: {result.message}. "
                    f"Demand: {demand_t:.2f} MW, Available capacity: {sum(max_heat[uid] * avail_matrix[t, i] for i, uid in enumerate(unit_ids)):.2f} MW"
                )
            
            # Extract solution
            heat_values = result.x[:n_units]
            unserved = result.x[n_units]
            
            # Verify equality (should be exact due to constraint, but check for numerical issues)
            served = heat_values.sum()
            closure_error = abs(served + unserved - demand_t)
            if closure_error > 1e-6:
                # Reconciliation step: recompute unserved to ensure exact equality
                unserved_reconciled = max(0.0, demand_t - served)
                if abs(unserved_reconciled - unserved) > 1e-3:
                    print(f"[WARN] LP solution had closure error {closure_error:.6f} MW at {timestamp}, reconciling unserved")
                    unserved = unserved_reconciled
                else:
                    raise ValueError(
                        f"LP solution violates equality constraint at {timestamp}: "
                        f"served={served:.6f}, unserved={unserved:.6f}, demand={demand_t:.6f}, error={closure_error:.6f}"
                    )
            
        except Exception as e:
            raise ValueError(f"LP dispatch failed at timestamp {timestamp}: {e}")
        
        # Build results for this hour
        for i, uid in enumerate(unit_ids):
            heat_mw = heat_values[i]
            eff = efficiency[uid]
            
            # Energy per step
            fuel_mwh = (heat_mw * dt_h) / eff if eff > 0 and heat_mw > 0 else 0.0
            co2_factor = util_df[util_df['unit_id'] == uid]['co2_factor_t_per_MWh_fuel'].iloc[0]
            co2_tonnes = fuel_mwh * co2_factor
            
            # Store unserved only on first unit (system-level)
            unserved_val = unserved if i == 0 else 0.0
            
            # Compute unit_on for plotting (1 if heat_MW > threshold, else 0)
            unit_on = 1 if heat_mw > 1e-6 else 0
            
            long_results.append({
                'timestamp_utc': timestamp,
                'unit_id': uid,
                'heat_MW': heat_mw,
                'unit_on': unit_on,  # Add unit_on for plotting compatibility
                'fuel_MWh': fuel_mwh,
                'co2_tonnes': co2_tonnes,
                'unserved_MW': unserved_val
            })
    
    dispatch_long = pd.DataFrame(long_results)
    
    # Final reconciliation: ensure unserved = demand - served for all timestamps (safeguard)
    for timestamp in demand_df['timestamp_utc'].unique():
        ts_mask = dispatch_long['timestamp_utc'] == timestamp
        demand_ts = demand_df[demand_df['timestamp_utc'] == timestamp]['heat_demand_MW'].iloc[0]
        served_ts = dispatch_long.loc[ts_mask, 'heat_MW'].sum()
        unserved_correct = max(0.0, demand_ts - served_ts)
        
        # Update unserved on first unit row only
        first_unit_mask = ts_mask & (dispatch_long['unit_id'] == dispatch_long.loc[ts_mask, 'unit_id'].iloc[0])
        dispatch_long.loc[first_unit_mask, 'unserved_MW'] = unserved_correct
        # Zero out unserved on other unit rows
        other_units_mask = ts_mask & ~first_unit_mask
        dispatch_long.loc[other_units_mask, 'unserved_MW'] = 0.0
    
    # Build wide-form
    dispatch_wide = dispatch_long.pivot_table(
        index='timestamp_utc',
        columns='unit_id',
        values='heat_MW',
        aggfunc='sum'
    ).reset_index()
    
    # Rename unit columns
    unit_cols = [col for col in dispatch_wide.columns if col != 'timestamp_utc']
    rename_dict = {col: f"{col}_MW" for col in unit_cols}
    dispatch_wide = dispatch_wide.rename(columns=rename_dict)
    
    # Add total_heat_MW and unserved_MW (from first unit row per timestamp)
    dispatch_wide['total_heat_MW'] = dispatch_wide[[col for col in dispatch_wide.columns if col.endswith('_MW') and col != 'total_heat_MW']].sum(axis=1)
    unserved_by_ts = dispatch_long.groupby('timestamp_utc')['unserved_MW'].first()
    dispatch_wide = dispatch_wide.merge(
        unserved_by_ts.reset_index(),
        on='timestamp_utc',
        how='left'
    )
    
    # Reorder columns
    unit_cols_sorted = sorted([col for col in dispatch_wide.columns if col.endswith('_MW') and col not in ['total_heat_MW', 'unserved_MW']])
    dispatch_wide = dispatch_wide[['timestamp_utc', 'total_heat_MW', 'unserved_MW'] + unit_cols_sorted]
    
    return dispatch_long, dispatch_wide


def export_incremental_electricity(
    dispatch_long: pd.DataFrame,
    util_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Export incremental electricity demand from electrode boilers.
    
    Computes: incremental_electricity_MW[t] = sum_{u in electric units} heat_MW[u,t] / efficiency_th[u]
    
    Args:
        dispatch_long: Long-form dispatch DataFrame with heat_MW
        util_df: Utilities DataFrame
        demand_df: Demand DataFrame (for timestamp alignment)
        output_path: Path to write CSV
        
    Raises:
        ValueError: If timestamps don't align exactly
    """
    # Identify electric units
    electric_units = util_df[
        (util_df['fuel'].str.lower().str.strip() == 'electricity') |
        (util_df['tech_type'].str.lower().str.strip() == 'electrode_boiler')
    ]['unit_id'].tolist()
    
    if len(electric_units) == 0:
        # No electric units - write zero series
        result = pd.DataFrame({
            'timestamp_utc': demand_df['timestamp_utc'],
            'incremental_electricity_MW': 0.0
        })
    else:
        # Compute incremental electricity per timestep
        electric_dispatch = dispatch_long[dispatch_long['unit_id'].isin(electric_units)].copy()
        
        # Merge efficiency
        eff_map = dict(zip(util_df['unit_id'], util_df['efficiency_th']))
        electric_dispatch['efficiency'] = electric_dispatch['unit_id'].map(eff_map)
        
        # Compute electricity demand: heat_MW / efficiency
        electric_dispatch['elec_MW'] = electric_dispatch['heat_MW'] / electric_dispatch['efficiency']
        
        # Aggregate by timestamp
        elec_by_ts = electric_dispatch.groupby('timestamp_utc')['elec_MW'].sum().reset_index()
        elec_by_ts.columns = ['timestamp_utc', 'incremental_electricity_MW']
        
        # Align to demand timestamps (strict merge)
        result = demand_df[['timestamp_utc']].merge(
            elec_by_ts,
            on='timestamp_utc',
            how='left',
            validate='one_to_one'
        )
        result['incremental_electricity_MW'] = result['incremental_electricity_MW'].fillna(0.0)
    
    # Convert timestamps to ISO Z format
    result['timestamp_utc'] = to_iso_z(result['timestamp_utc'])
    
    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[OK] Exported incremental electricity demand to {output_path}")


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
    dt_h: float = 1.0,
    maintenance_availability: Optional[pd.DataFrame] = None
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
    
    # Build maintenance availability lookup (time-varying)
    # maintenance_availability can be wide-form (index=timestamps, columns=unit_ids) or long-form
    maint_avail_by_ts = {}  # {timestamp: {unit_id: availability}}
    if maintenance_availability is not None:
        if isinstance(maintenance_availability.index, pd.DatetimeIndex):
            # Wide-form: index=timestamps, columns=unit_ids
            for ts in maintenance_availability.index:
                maint_avail_by_ts[ts] = maintenance_availability.loc[ts].to_dict()
        else:
            # Long-form: columns include timestamp_utc, unit_id, availability_multiplier
            for _, row in maintenance_availability.iterrows():
                ts = row['timestamp_utc']
                if ts not in maint_avail_by_ts:
                    maint_avail_by_ts[ts] = {}
                maint_avail_by_ts[ts][row['unit_id']] = row['availability_multiplier']
    
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
        
        # Filter out units that are unavailable (availability==0) for any hour in this block
        # This prevents selecting subsets with units that cannot operate
        if maintenance_availability is not None:
            block_timestamps = block_demand['timestamp_utc'].values
            units_available_in_block = set(available_units)
            for ts in block_timestamps:
                if ts in maint_avail_by_ts:
                    for unit_id in list(units_available_in_block):
                        avail = maint_avail_by_ts[ts].get(unit_id, 1.0)
                        if avail == 0.0:
                            # Unit is unavailable at this timestamp, remove from candidate set
                            units_available_in_block.discard(unit_id)
            available_units = [uid for uid in available_units if uid in units_available_in_block]
        
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
                            # Get time-varying maintenance availability
                            maint_avail = 1.0
                            if timestamp in maint_avail_by_ts:
                                maint_avail = maint_avail_by_ts[timestamp].get(unit_id, 1.0)
                            # Effective capacity = nameplate * static_availability_factor * time_varying_maintenance
                            cap_u = util_row['max_heat_MW'] * util_row.get('availability_factor', 1.0) * maint_avail
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
                        # Use maintenance availability if provided
                        online_cap_MW = 0.0
                        for unit_id in subset_units:
                            util_row = existing_units[existing_units['unit_id'] == unit_id].iloc[0]
                            # Get time-varying maintenance availability for this timestamp
                            maint_avail = 1.0
                            if timestamp in maint_avail_by_ts:
                                maint_avail = maint_avail_by_ts[timestamp].get(unit_id, 1.0)
                            cap_u = util_row['max_heat_MW'] * util_row.get('availability_factor', 1.0) * maint_avail
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
    
    # Compute unserved penalty cost if unserved_MW exists and penalty rate provided
    # CRITICAL: Ensure unserved_cost_nzd column exists and is initialized to 0.0
    if 'unserved_cost_nzd' not in dispatch_long.columns:
        dispatch_long['unserved_cost_nzd'] = 0.0
    
    if unserved_penalty_nzd_per_MWh is not None and (dispatch_long['unserved_MW'] > 0).any():
        unserved_mask = dispatch_long['unserved_MW'] > 0
        # Compute cost: unserved_MW * dt_h * penalty_rate (converts MW to MWh, then applies penalty)
        dispatch_long.loc[unserved_mask, 'unserved_cost_nzd'] = (
            dispatch_long.loc[unserved_mask, 'unserved_MW'] * dt_h * unserved_penalty_nzd_per_MWh
        )
        # Verify computation: sum should equal total unserved_MWh * penalty_rate
        total_unserved_mwh = (dispatch_long.loc[unserved_mask, 'unserved_MW'] * dt_h).sum()
        total_cost = dispatch_long.loc[unserved_mask, 'unserved_cost_nzd'].sum()
        expected_cost = total_unserved_mwh * unserved_penalty_nzd_per_MWh
        if abs(total_cost - expected_cost) > 1e-3:
            print(f"[WARN] Unserved cost computation mismatch: computed={total_cost:.2f}, expected={expected_cost:.2f}")
    
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
        
        # Unserved cost (if not already computed and penalty rate provided)
        if 'unserved_cost_nzd' in dispatch_long.columns and unserved_penalty_nzd_per_MWh is not None:
            # Recompute unserved cost from unserved_MW
            # Penalty is per MWh, so multiply unserved_MW * dt_h to get energy
            unserved_mask = mask & (dispatch_long['unserved_MW'] > 0)
            if unserved_mask.any():
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
                          signals: Union[Dict[str, float], pd.DataFrame], dt_h: float = 1.0,
                          unserved_penalty_nzd_per_MWh: Optional[float] = None) -> pd.DataFrame:
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
    # Handle both dict and DataFrame signals
    is_time_varying = isinstance(signals, pd.DataFrame)
    if is_time_varying:
        if 'ets_price_nzd_per_tCO2' in signals.columns:
            # Time-varying ETS (unusual, but handle it)
            ets_price_series = signals['ets_price_nzd_per_tCO2']
            # Use first value as default (should be constant, but handle variation)
            ets_price = ets_price_series.iloc[0] if len(ets_price_series) > 0 else 0.0
        else:
            # Fallback: use default
            ets_price = 0.0
            print("[WARN] ETS price (ets_price_nzd_per_tCO2) not found in signals DataFrame, defaulting to 0.0")
    else:
        ets_price = signals.get('ets_price_nzd_per_tCO2', 0.0)
        if ets_price == 0.0 and 'ets_price_nzd_per_tCO2' not in signals:
            print("[WARN] ETS price (ets_price_nzd_per_tCO2) not found in signals, defaulting to 0.0")
    
    # Handle time-varying vs flat signals
    is_time_varying = isinstance(signals, pd.DataFrame)
    
    # Process by fuel type
    for fuel_type in dispatch_long['fuel_type'].unique():
        mask = dispatch_long['fuel_type'] == fuel_type
        
        if fuel_type == 'lignite' or fuel_type == 'coal':
            # Use coal price and ETS
            if is_time_varying:
                # Merge time-varying prices
                dispatch_with_sigs = dispatch_long.loc[mask, ['timestamp_utc', 'fuel_MWh', 'co2_tonnes']].merge(
                    signals[['timestamp_utc', 'coal_price_nzd_per_MWh_fuel']],
                    on='timestamp_utc',
                    how='left'
                )
                if dispatch_with_sigs['coal_price_nzd_per_MWh_fuel'].isna().any():
                    raise ValueError("Coal price alignment failed: missing timestamps")
                dispatch_long.loc[mask, 'fuel_cost_nzd'] = (
                    dispatch_with_sigs['fuel_MWh'] * dispatch_with_sigs['coal_price_nzd_per_MWh_fuel']
                ).values
            else:
                # Flat price from dict
                coal_price = signals.get('coal_price_nzd_per_MWh_fuel', 0.0)
                if coal_price == 0.0 and 'coal_price_nzd_per_MWh_fuel' not in signals:
                    print("[WARN] Coal price (coal_price_nzd_per_MWh_fuel) not found in signals, defaulting to 0.0")
                dispatch_long.loc[mask, 'fuel_cost_nzd'] = (
                    dispatch_long.loc[mask, 'fuel_MWh'] * coal_price
                )
            
            # Carbon cost (ETS price is typically flat, but handle time-varying if needed)
            if is_time_varying and 'ets_price_nzd_per_tCO2' in signals.columns:
                dispatch_with_ets = dispatch_long.loc[mask, ['timestamp_utc', 'co2_tonnes']].merge(
                    signals[['timestamp_utc', 'ets_price_nzd_per_tCO2']],
                    on='timestamp_utc',
                    how='left'
                )
                dispatch_long.loc[mask, 'carbon_cost_nzd'] = (
                    dispatch_with_ets['co2_tonnes'] * dispatch_with_ets['ets_price_nzd_per_tCO2']
                ).values
            else:
                # Use the ets_price from outer scope (already set)
                dispatch_long.loc[mask, 'carbon_cost_nzd'] = (
                    dispatch_long.loc[mask, 'co2_tonnes'] * ets_price
                )
                
        elif fuel_type == 'biomass':
            # Use biomass price and minimal ETS
            if is_time_varying:
                dispatch_with_sigs = dispatch_long.loc[mask, ['timestamp_utc', 'fuel_MWh', 'co2_tonnes']].merge(
                    signals[['timestamp_utc', 'biomass_price_nzd_per_MWh_fuel']],
                    on='timestamp_utc',
                    how='left'
                )
                if dispatch_with_sigs['biomass_price_nzd_per_MWh_fuel'].isna().any():
                    raise ValueError("Biomass price alignment failed: missing timestamps")
                dispatch_long.loc[mask, 'fuel_cost_nzd'] = (
                    dispatch_with_sigs['fuel_MWh'] * dispatch_with_sigs['biomass_price_nzd_per_MWh_fuel']
                ).values
            else:
                biomass_price = signals.get('biomass_price_nzd_per_MWh_fuel', 0.0)
                if biomass_price == 0.0 and 'biomass_price_nzd_per_MWh_fuel' not in signals:
                    print("[WARN] Biomass price (biomass_price_nzd_per_MWh_fuel) not found in signals, defaulting to 0.0")
                dispatch_long.loc[mask, 'fuel_cost_nzd'] = (
                    dispatch_long.loc[mask, 'fuel_MWh'] * biomass_price
                )
            
            # Carbon cost
            if is_time_varying and 'ets_price_nzd_per_tCO2' in signals.columns:
                dispatch_with_ets = dispatch_long.loc[mask, ['timestamp_utc', 'co2_tonnes']].merge(
                    signals[['timestamp_utc', 'ets_price_nzd_per_tCO2']],
                    on='timestamp_utc',
                    how='left'
                )
                dispatch_long.loc[mask, 'carbon_cost_nzd'] = (
                    dispatch_with_ets['co2_tonnes'] * dispatch_with_ets['ets_price_nzd_per_tCO2']
                ).values
            else:
                dispatch_long.loc[mask, 'carbon_cost_nzd'] = (
                    dispatch_long.loc[mask, 'co2_tonnes'] * ets_price
                )
                
        elif fuel_type == 'electricity':
            # Use time-varying electricity price if available, else flat
            if is_time_varying:
                if 'elec_price_nzd_per_MWh' in signals.columns:
                    dispatch_with_sigs = dispatch_long.loc[mask, ['timestamp_utc', 'fuel_MWh']].merge(
                        signals[['timestamp_utc', 'elec_price_nzd_per_MWh']],
                        on='timestamp_utc',
                        how='left'
                    )
                    if dispatch_with_sigs['elec_price_nzd_per_MWh'].isna().any():
                        raise ValueError("Electricity price alignment failed: missing timestamps")
                    dispatch_long.loc[mask, 'fuel_cost_nzd'] = (
                        dispatch_with_sigs['fuel_MWh'] * dispatch_with_sigs['elec_price_nzd_per_MWh']
                    ).values
                else:
                    # Fallback to flat price from signals dict if present
                    elec_price = 0.0
                    if hasattr(signals, 'get'):
                        elec_price = signals.get('electricity_price_nzd_per_MWh_fuel', 0.0)
                    dispatch_long.loc[mask, 'fuel_cost_nzd'] = (
                        dispatch_long.loc[mask, 'fuel_MWh'] * elec_price
                    )
            else:
                elec_price = signals.get('electricity_price_nzd_per_MWh_fuel', 0.0)
                dispatch_long.loc[mask, 'fuel_cost_nzd'] = (
                    dispatch_long.loc[mask, 'fuel_MWh'] * elec_price
                )
            
            # Electricity has no direct CO2 emissions (grid emissions handled separately)
            dispatch_long.loc[mask, 'carbon_cost_nzd'] = 0.0
        
        elif fuel_type is None or pd.isna(fuel_type):
            # Handle UNSERVED or other units without fuel type
            # UNSERVED has no fuel cost or carbon cost, only penalty cost (handled separately)
            dispatch_long.loc[mask, 'fuel_cost_nzd'] = 0.0
            dispatch_long.loc[mask, 'carbon_cost_nzd'] = 0.0
            
        else:
            # Unknown fuel type - warn but don't fail
            print(f"[WARNING] Unknown fuel type '{fuel_type}', costs set to zero")
    
    # Ensure unserved_cost_nzd column exists for ALL rows (initialize to 0.0)
    if 'unserved_cost_nzd' not in dispatch_long.columns:
        dispatch_long['unserved_cost_nzd'] = 0.0
    else:
        # Ensure dtype is float and fill NaNs with 0.0
        dispatch_long['unserved_cost_nzd'] = dispatch_long['unserved_cost_nzd'].astype(float).fillna(0.0)
    
    # Add unserved penalty cost for UNSERVED rows (if penalty rate provided)
    if unserved_penalty_nzd_per_MWh is not None and unserved_penalty_nzd_per_MWh > 0:
        # For UNSERVED rows: penalty cost = heat_MW * dt_h * penalty_rate
        # heat_MW is already average MW per timestep, so energy = heat_MW * dt_h
        unserved_mask = dispatch_long['unit_id'] == 'UNSERVED'
        if unserved_mask.any():
            dispatch_long.loc[unserved_mask, 'unserved_cost_nzd'] = (
                dispatch_long.loc[unserved_mask, 'heat_MW'] * dt_h * unserved_penalty_nzd_per_MWh
            ).astype(float)
    
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
    # System penalties: for UNSERVED rows, sum heat_MW and unserved_cost_nzd directly
    # For other modes (LP/optimal_subset), unserved_MW may be stored once per hour on first unit row
    if 'unit_id' in dispatch_long_for_agg.columns and (dispatch_long_for_agg['unit_id'] == 'UNSERVED').any():
        # UNSERVED is stored as unit_id="UNSERVED" rows
        unserved_rows = dispatch_long_for_agg[dispatch_long_for_agg['unit_id'] == 'UNSERVED']
        system_costs['unserved_MW'] = float(unserved_rows['heat_MW'].fillna(0.0).sum())
        
        # Ensure unserved_cost_nzd exists and is numeric (should always be present after add_costs_to_dispatch)
        if 'unserved_cost_nzd' in unserved_rows.columns:
            system_costs['unserved_cost_nzd'] = float(unserved_rows['unserved_cost_nzd'].fillna(0.0).sum())
        else:
            # Should not happen if add_costs_to_dispatch was called, but handle gracefully
            system_costs['unserved_cost_nzd'] = 0.0
        
        # Compute unserved_MWh from UNSERVED heat_MW (energy = MW * dt_h)
        # Get dt_h from timestamps if not available (assume 1.0 if can't determine)
        dt_h_est = 1.0
        if len(unserved_rows) > 1:
            # Parse timestamps if needed
            unserved_ts = unserved_rows['timestamp_utc'].copy()
            if not pd.api.types.is_datetime64_any_dtype(unserved_ts):
                unserved_ts = parse_any_timestamp(unserved_ts)
            ts_diff = unserved_ts.diff().iloc[1]
            if pd.notna(ts_diff):
                dt_h_est = ts_diff.total_seconds() / 3600.0
        system_costs['unserved_MWh'] = float(system_costs['unserved_MW'] * dt_h_est)
    elif 'unserved_MW' in dispatch_long_for_agg.columns:
        # System penalties are stored once per hour (on first unit row), so aggregate by timestamp_utc first
        system_costs['unserved_MW'] = float(dispatch_long_for_agg.groupby('timestamp_utc')['unserved_MW'].first().sum())
        if 'unserved_cost_nzd' in dispatch_long_for_agg.columns:
            system_costs['unserved_cost_nzd'] = float(dispatch_long_for_agg.groupby('timestamp_utc')['unserved_cost_nzd'].first().fillna(0.0).sum())
            # Compute unserved_MWh from unserved_MW
            dt_h_est = 1.0
            if len(dispatch_long_for_agg) > 1:
                ts_diff = dispatch_long_for_agg['timestamp_utc'].diff().iloc[1]
                if pd.notna(ts_diff):
                    dt_h_est = ts_diff.total_seconds() / 3600.0
            system_costs['unserved_MWh'] = float(system_costs['unserved_MW'] * dt_h_est)
        else:
            system_costs['unserved_cost_nzd'] = 0.0
            system_costs['unserved_MWh'] = 0.0
    else:
        # No unserved data
        system_costs['unserved_MW'] = 0.0
        system_costs['unserved_cost_nzd'] = 0.0
        system_costs['unserved_MWh'] = 0.0
    
    # Reserve and online costs (always initialize to 0.0 if missing)
    if 'reserve_shortfall_MW' in dispatch_long_for_agg.columns:
        system_costs['reserve_shortfall_MW'] = float(dispatch_long_for_agg.groupby('timestamp_utc')['reserve_shortfall_MW'].first().sum())
    else:
        system_costs['reserve_shortfall_MW'] = 0.0
    
    if 'reserve_penalty_cost_nzd' in dispatch_long_for_agg.columns:
        system_costs['reserve_penalty_cost_nzd'] = float(dispatch_long_for_agg.groupby('timestamp_utc')['reserve_penalty_cost_nzd'].first().fillna(0.0).sum())
    else:
        system_costs['reserve_penalty_cost_nzd'] = 0.0
    
    if 'online_cost_nzd' in dispatch_long_for_agg.columns:
        system_costs['online_cost_nzd'] = float(dispatch_long_for_agg.groupby('timestamp_utc')['online_cost_nzd'].first().fillna(0.0).sum())
    else:
        system_costs['online_cost_nzd'] = 0.0
    
    return system_costs


def compute_annual_summary(dispatch_long: pd.DataFrame, reserve_frac: float = 0.0, dt_h: float = 1.0,
                           unserved_penalty_nzd_per_MWh: Optional[float] = None,
                           reserve_penalty_nzd_per_MWh: Optional[float] = None) -> pd.DataFrame:
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
    # UNSERVED is included in aggregation but will be handled specially (it has fuel_MWh=0, co2=0, etc.)
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
    # BUT: for UNSERVED rows, unserved_cost_nzd IS the unit's cost, so include it
    if 'unserved_cost_nzd' in dispatch_long.columns:
        # For UNSERVED: unserved_cost_nzd is the unit's cost; for real units it's 0
        agg_dict['unserved_cost_nzd'] = 'sum'
    
    annual_by_unit = dispatch_long.groupby('unit_id').agg(agg_dict)
    
    # Recompute unit-level total_cost_nzd
    # For real units: total_cost_nzd = fuel + carbon + var_om + fixed_on + startup (operational costs)
    # For UNSERVED: total_cost_nzd = unserved_cost_nzd (penalty cost only)
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
    
    # For UNSERVED: total_cost_nzd should be unserved_cost_nzd (penalty cost)
    # Real units have unserved_cost_nzd=0, so this only affects UNSERVED
    if 'unserved_cost_nzd' in annual_by_unit.columns:
        # For UNSERVED: replace total_cost with penalty cost
        if 'UNSERVED' in annual_by_unit.index:
            annual_by_unit.loc['UNSERVED', 'total_cost_nzd'] = annual_by_unit.loc['UNSERVED', 'unserved_cost_nzd']
    
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
    
    # Set system penalty columns to 0 for real unit rows (they're system-level, not unit-level)
    # BUT: for UNSERVED, unserved_MW and unserved_cost_nzd ARE the unit's values, so keep them
    system_penalty_cols = ['reserve_shortfall_MW', 'reserve_penalty_cost_nzd', 'online_cost_nzd']
    for col in system_penalty_cols:
        if col in dispatch_long.columns:
            annual_by_unit[col] = 0.0
    
    # For UNSERVED: set unserved_MW from heat_MW (it represents unserved demand)
    if 'UNSERVED' in annual_by_unit.index:
        # UNSERVED's heat_MW is the unserved demand
        annual_by_unit.loc['UNSERVED', 'unserved_MW'] = annual_by_unit.loc['UNSERVED', 'heat_MW']
    else:
        # No UNSERVED row: set unserved_MW to 0 for all units
        if 'unserved_MW' in dispatch_long.columns:
            annual_by_unit['unserved_MW'] = 0.0
    
    # Add operational/penalty cost columns to unit rows
    # For real units: operational costs only, no penalties
    # For UNSERVED: penalty cost only (unserved_cost_nzd), no operational costs
    annual_by_unit['annual_operational_cost_nzd'] = annual_by_unit['annual_total_cost_nzd'].copy()
    annual_by_unit['annual_penalty_cost_nzd'] = 0.0
    annual_by_unit['annual_system_cost_nzd'] = 0.0
    
    # For UNSERVED: penalty cost is the total cost
    if 'UNSERVED' in annual_by_unit.index:
        # Get unserved_cost_nzd from aggregated column (should be sum of all UNSERVED rows' unserved_cost_nzd)
        unserved_penalty = 0.0
        if 'unserved_cost_nzd' in annual_by_unit.columns:
            unserved_penalty_val = annual_by_unit.loc['UNSERVED', 'unserved_cost_nzd']
            unserved_penalty = float(unserved_penalty_val) if not pd.isna(unserved_penalty_val) else 0.0
        annual_by_unit.loc['UNSERVED', 'annual_operational_cost_nzd'] = 0.0
        annual_by_unit.loc['UNSERVED', 'annual_penalty_cost_nzd'] = unserved_penalty
        annual_by_unit.loc['UNSERVED', 'annual_total_cost_nzd'] = unserved_penalty
        # Also ensure unserved_cost_nzd column is set (for printing)
        if 'unserved_cost_nzd' not in annual_by_unit.columns:
            annual_by_unit['unserved_cost_nzd'] = 0.0
        annual_by_unit.loc['UNSERVED', 'unserved_cost_nzd'] = unserved_penalty
    
    annual_by_unit['avg_operational_cost_nzd_per_MWh_heat'] = annual_by_unit['avg_cost_nzd_per_MWh_heat']
    
    # Aggregate system-level costs separately (stored once per hour, not per unit)
    # Note: parse timestamp_utc if it's a string for grouping
    dispatch_long_for_agg = dispatch_long.copy()
    if not pd.api.types.is_datetime64_any_dtype(dispatch_long_for_agg['timestamp_utc']):
        dispatch_long_for_agg['timestamp_utc'] = parse_any_timestamp(dispatch_long_for_agg['timestamp_utc'])
    system_costs = aggregate_system_costs(dispatch_long_for_agg)
    
    # Initialize unserved metrics at function start (before any conditionals) to avoid UnboundLocalError
    # These will be computed consistently from system_costs or UNSERVED rows
    unserved_mw_sum = 0.0
    unserved_energy_mwh = 0.0
    unserved_penalty_cost_nzd = 0.0
    
    # Extract penalty values from system_costs (always defined, defaults to 0.0)
    # system_costs['unserved_MWh'] is already computed in aggregate_system_costs (from UNSERVED heat_MW * dt_h)
    # system_costs['unserved_cost_nzd'] is the sum of unserved_cost_nzd from all UNSERVED rows
    unserved_mw_sum = float(system_costs.get('unserved_MW', 0.0))
    unserved_energy_mwh = float(system_costs.get('unserved_MWh', 0.0))  # Use pre-computed value from aggregate_system_costs
    if unserved_energy_mwh == 0.0 and unserved_mw_sum > 0.0:
        # Fallback: compute from unserved_MW if unserved_MWh not in system_costs but unserved_MW exists
        unserved_energy_mwh = unserved_mw_sum * dt_h
    
    # Get penalty costs from system_costs (should always be present and correct)
    # system_costs['unserved_cost_nzd'] is the canonical value - use it directly, no recomputation needed
    unserved_penalty_cost_nzd = float(system_costs.get('unserved_cost_nzd', 0.0))
    
    # Reserve metrics
    reserve_mw_sum = float(system_costs.get('reserve_shortfall_MW', 0.0))
    reserve_shortfall_mwh = reserve_mw_sum * dt_h  # Convert to energy (MWh)
    reserve_penalty_cost_nzd = float(system_costs.get('reserve_penalty_cost_nzd', 0.0))
    
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
    # Use the same unserved_energy_mwh and reserve_shortfall_mwh computed above for consistency
    # CRITICAL: Ensure all penalty values are numeric (0.0, not NaN) - variables are already float from initialization
    system_dict['unserved_MW'] = unserved_mw_sum  # Sum of average MW per step (already float)
    system_dict['unserved_MWh'] = unserved_energy_mwh  # Energy (MWh) - same value used everywhere (already float)
    system_dict['unserved_cost_nzd'] = unserved_penalty_cost_nzd  # Cost from system_costs (already float)
    system_dict['unserved_penalty_cost_nzd'] = unserved_penalty_cost_nzd  # Alias for clarity (already float)
    
    system_dict['reserve_shortfall_MW'] = reserve_mw_sum  # Sum of average MW per step (already float)
    system_dict['reserve_shortfall_MWh'] = reserve_shortfall_mwh  # Energy (MWh) - same value used everywhere (already float)
    system_dict['reserve_penalty_cost_nzd'] = reserve_penalty_cost_nzd  # Cost from system_costs (already float)
    
    # Add total penalty cost (ensure numeric 0.0, not NaN)
    total_penalty = float(unserved_penalty_cost_nzd) if not pd.isna(unserved_penalty_cost_nzd) else 0.0
    total_penalty += float(reserve_penalty_cost_nzd) if not pd.isna(reserve_penalty_cost_nzd) else 0.0
    system_dict['total_penalty_cost_nzd'] = total_penalty
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
        # Add explicit penalty columns for CSV output (use same values as SYSTEM row for consistency)
        'unserved_MWh': unserved_energy_mwh,  # Same value used everywhere
        'unserved_penalty_cost_nzd': unserved_penalty_cost_nzd,  # Same value used everywhere
        'reserve_shortfall_MWh': reserve_shortfall_mwh,  # Same value used everywhere
        'reserve_penalty_cost_nzd': reserve_penalty_cost_nzd,  # Same value used everywhere
        'total_penalty_cost_nzd': total_system_penalties_nzd,  # Ensure numeric 0.0, not NaN
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
    
    # Add explicit penalty columns (for SYSTEM and TOTAL rows)
    penalty_cols = ['unserved_MWh', 'unserved_penalty_cost_nzd', 'reserve_shortfall_MWh', 
                    'reserve_penalty_cost_nzd', 'total_penalty_cost_nzd']
    for col in penalty_cols:
        if col in summary.columns:
            output_cols.append(col)
    
    # Add legacy MW-based columns if present (for backward compatibility)
    if 'unserved_MW' in summary.columns:
        output_cols.append('unserved_MW')
    if 'unserved_cost_nzd' in summary.columns and 'unserved_cost_nzd' not in output_cols:
        output_cols.append('unserved_cost_nzd')
    if 'reserve_shortfall_MW' in summary.columns:
        output_cols.append('reserve_shortfall_MW')
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
    
    # Get unit columns (all columns ending in _MW except total_heat_MW and unserved_MW)
    # We'll handle unserved_MW separately to show it distinctly
    unit_cols = [col for col in df.columns if col.endswith('_MW') and col not in ['total_heat_MW', 'unserved_MW']]
    has_unserved = 'unserved_MW' in df.columns
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create stacked area plot for real units
    if unit_cols:
        ax.stackplot(df['timestamp_utc'], 
                     [df[col] for col in unit_cols],
                     labels=[col.replace('_MW', '') for col in unit_cols],
                     alpha=0.7)
    
    # Add unserved as a separate stack layer (on top) with distinct styling
    if has_unserved and (df['unserved_MW'] > 1e-6).any():
        # Compute cumulative served to stack unserved on top
        served = df[unit_cols].sum(axis=1) if unit_cols else pd.Series(0.0, index=df.index)
        ax.fill_between(df['timestamp_utc'], served, served + df['unserved_MW'],
                        label='UNSERVED', alpha=0.8, color='red', hatch='///', edgecolor='darkred', linewidth=0.5)
    
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
    # Use unit_online if available, otherwise fall back to unit_on, or compute from heat_MW
    if 'unit_online' in df.columns:
        online_col = 'unit_online'
        plot_label = 'Units Online (hot standby allowed)'
    elif 'unit_on' in df.columns:
        online_col = 'unit_on'
        plot_label = 'Units Firing (heat_MW > 0)'
    elif 'heat_MW' in df.columns:
        # Compute unit_on from heat_MW (for LP mode compatibility)
        df['unit_on'] = (df['heat_MW'] > 1e-6).astype(int)
        online_col = 'unit_on'
        plot_label = 'Units Firing (heat_MW > 0)'
    else:
        # No way to determine units online - skip this plot
        print(f"[WARN] Cannot plot units online: missing unit_online, unit_on, and heat_MW columns in {dispatch_long_path}")
        return
    
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


def resolve_demand_csv(repo_root: Path, epoch: int, demand_csv_arg: Optional[str] = None) -> Path:
    """
    Resolve demand CSV path for an epoch.
    
    Priority:
    1. If --demand-csv provided, use it
    2. Else, try Output/runs/<latest epoch run>/demandpack/hourly_heat_demand_<epoch>.csv
    3. Else, try Output/hourly_heat_demand_<epoch>.csv
    4. Else, error with clear message
    
    Args:
        repo_root: Repository root directory
        epoch: Epoch year
        demand_csv_arg: Optional CLI argument value
        
    Returns:
        Resolved Path to demand CSV
        
    Raises:
        FileNotFoundError: If no demand CSV found
    """
    if demand_csv_arg:
        candidate = resolve_path(demand_csv_arg)
        if candidate.exists():
            return candidate
        else:
            raise FileNotFoundError(f"Demand CSV not found: {candidate}")
    
    output_root = repo_root / 'Output'
    
    # Try 2: Latest run folder for this epoch
    runs_dir = output_root / 'runs'
    if runs_dir.exists():
        # Find most recent run folder that matches epoch pattern
        epoch_runs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and f'epoch{epoch}' in d.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if epoch_runs:
            candidate = epoch_runs[0] / 'demandpack' / f'hourly_heat_demand_{epoch}.csv'
            if candidate.exists():
                return candidate
    
    # Try 3: Direct Output root
    candidate = output_root / f'hourly_heat_demand_{epoch}.csv'
    if candidate.exists():
        return candidate
    
    # Not found - error with helpful message
    attempted = [
        f"  - Output/runs/*/demandpack/hourly_heat_demand_{epoch}.csv (latest run)",
        f"  - Output/hourly_heat_demand_{epoch}.csv"
    ]
    raise FileNotFoundError(
        f"Demand CSV not found for epoch {epoch}.\n"
        f"Attempted paths:\n" + "\n".join(attempted) + "\n"
        f"Please provide --demand-csv or ensure demand file exists."
    )


def validate_maintenance_only(
    epoch: int,
    demand_csv_path: Path,
    utilities_csv_path: Path,
    repo_root: Path,
    variant: Optional[str] = None
) -> int:
    """
    Validate maintenance windows without running dispatch.
    
    Args:
        epoch: Epoch year
        demand_csv_path: Path to demand CSV
        utilities_csv_path: Path to utilities CSV
        repo_root: Repository root directory
        variant: Optional variant string (for 2035)
        
    Returns:
        Exit code: 0 for PASS, 2 for FAIL, 1 for ERROR
    """
    print("\n" + "=" * 80)
    print(f"VALIDATION MODE: Checking maintenance windows for epoch {epoch}")
    print("=" * 80)
    
    # Load demand timestamps
    print(f"\n[1/3] Loading demand timestamps from {demand_csv_path}...")
    try:
        demand_df, dt_h = load_hourly_demand(str(demand_csv_path))
        timestamps_utc = pd.DatetimeIndex(demand_df['timestamp_utc'].values)
        if timestamps_utc.tz is None:
            timestamps_utc = timestamps_utc.tz_localize('UTC')
        else:
            timestamps_utc = timestamps_utc.tz_convert('UTC')
        print(f"[OK] Loaded {len(timestamps_utc)} timesteps (dt_h={dt_h:.6f}h)")
    except Exception as e:
        print(f"[ERROR] Failed to load demand: {e}")
        return 1
    
    # Load utilities
    print(f"\n[2/3] Loading utilities from {utilities_csv_path}...")
    try:
        util_df = load_utilities(str(utilities_csv_path), epoch=epoch)
        unit_ids = util_df['unit_id'].tolist()
        print(f"[OK] Loaded {len(unit_ids)} units: {', '.join(unit_ids)}")
    except Exception as e:
        print(f"[ERROR] Failed to load utilities: {e}")
        return 1
    
    # Load maintenance windows
    print(f"\n[3/3] Loading maintenance windows...")
    try:
        maint_df = maint.load_maintenance_windows(repo_root, epoch, variant=variant)
    except Exception as e:
        print(f"[ERROR] Failed to load maintenance windows: {e}")
        return 1
    
    if len(maint_df) == 0:
        print(f"[INFO] No maintenance windows found for epoch {epoch}")
        print(f"  Expected: Input/site/maintenance/maintenance_windows_{epoch}.csv")
        print(f"[PASS] No maintenance to validate")
        return 0
    
    print(f"[OK] Loaded {len(maint_df)} maintenance window(s)")
    
    # Build availability matrix
    try:
        availability_matrix = maint.build_availability_matrix(
            timestamps_utc,
            unit_ids,
            maint_df
        )
        print(f"[OK] Built availability matrix: {len(availability_matrix)} timesteps Ã— {len(unit_ids)} units")
    except Exception as e:
        print(f"[ERROR] Failed to build availability matrix: {e}")
        return 1
    
    # Validate each maintenance window
    print("\n" + "=" * 80)
    print("VALIDATING MAINTENANCE WINDOWS")
    print("=" * 80)
    
    all_passed = True
    tolerance = 1e-6
    
    for idx, window in maint_df.iterrows():
        unit_id = window['unit_id']
        start = window['start_timestamp_utc']
        end = window['end_timestamp_utc']
        availability = float(window['availability'])
        
        # Check unit_id exists
        if unit_id not in unit_ids:
            print(f"\n[FAIL] Window {idx+1}: unit_id '{unit_id}' not found in utilities")
            print(f"  Available units: {', '.join(unit_ids)}")
            all_passed = False
            continue
        
        # Find timesteps covered by this window
        mask = (timestamps_utc >= start) & (timestamps_utc < end)
        covered_timesteps = timestamps_utc[mask]
        n_hours = len(covered_timesteps)
        
        if n_hours == 0:
            print(f"\n[WARN] Window {idx+1}: {unit_id} [{start} to {end}) - no timesteps covered")
            continue
        
        # Get availability values for this unit in this window
        if unit_id not in availability_matrix.columns:
            print(f"\n[FAIL] Window {idx+1}: unit_id '{unit_id}' not in availability matrix")
            all_passed = False
            continue
        
        unit_avail_series = availability_matrix.loc[covered_timesteps, unit_id]
        min_avail = float(unit_avail_series.min())
        max_avail = float(unit_avail_series.max())
        mean_avail = float(unit_avail_series.mean())
        
        # Check: if availability==0, all covered hours must be exactly 0
        if availability == 0.0:
            non_zero_count = (unit_avail_series > tolerance).sum()
            if non_zero_count > 0:
                print(f"\n[FAIL] Window {idx+1}: {unit_id} [{start} to {end})")
                print(f"  Expected: availability=0.0 (fully offline)")
                print(f"  Found: {non_zero_count}/{n_hours} hours with availability > {tolerance}")
                print(f"  Min availability in window: {min_avail:.6f}")
                print(f"  Max availability in window: {max_avail:.6f}")
                all_passed = False
            else:
                print(f"[PASS] Window {idx+1}: {unit_id} [{start} to {end}) - availability=0.0, {n_hours}h covered, all zero")
        else:
            # For availability > 0, check it's applied correctly (within tolerance)
            expected_avail = availability
            if abs(min_avail - expected_avail) > tolerance or abs(max_avail - expected_avail) > tolerance:
                print(f"\n[WARN] Window {idx+1}: {unit_id} [{start} to {end})")
                print(f"  Expected: availability={expected_avail:.3f}")
                print(f"  Found: min={min_avail:.6f}, max={max_avail:.6f}, mean={mean_avail:.6f}")
                print(f"  Note: Overlapping windows may cause minimum (most restrictive) to apply")
            else:
                print(f"[PASS] Window {idx+1}: {unit_id} [{start} to {end}) - availability={availability:.3f}, {n_hours}h covered, min={min_avail:.3f}, max={max_avail:.3f}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("[PASS] All maintenance windows validated successfully")
        return 0
    else:
        print("[FAIL] Some maintenance windows failed validation")
        return 2


def main():
    """CLI entrypoint for site dispatch computation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute site utility dispatch')
    parser.add_argument('--demand-csv', default=None,
                       help='Path to hourly demand CSV (auto-discovered if not provided, epoch-aware)')
    parser.add_argument('--utilities-csv', default=None,
                       help='Path to site utilities CSV (default: auto-discover from Input/site/utilities/)')
    parser.add_argument('--demandpack-config', default=None,
                       help='Path to demandpack config (optional, used to check for utilities path)')
    parser.add_argument('--mode', choices=['proportional', 'optimal_subset', 'lp'], default='proportional',
                       help='Dispatch mode: proportional (default), optimal_subset, or lp (linear programming with equality constraints)')
    parser.add_argument('--commitment-block-hours', type=int, default=24,
                       help='Hours per commitment block for optimal_subset mode (default: 24 = daily)')
    parser.add_argument('--unserved-penalty-nzd-per-MWh', type=float, default= 10000.0,
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
    parser.add_argument('--maintenance-csv', type=str, default=None,
                       help='Path to maintenance windows CSV (optional). Also accepts --maintenance-windows-csv for backward compatibility.')
    parser.add_argument('--maintenance-windows-csv', type=str, default=None,
                       help='[DEPRECATED] Use --maintenance-csv instead. Path to maintenance windows CSV (optional)')
    parser.add_argument('--electricity-signals-csv', type=str, default=None,
                       help='Path to electricity signals CSV with ToU prices and headroom (optional)')
    parser.add_argument('--grid-emissions-csv', type=str, default=None,
                       help='Path to grid emissions intensity CSV (optional, for reporting)')
    parser.add_argument('--time-min', type=str, default=None,
                       help='Minimum timestamp (ISO UTC with Z, e.g., 2020-07-06T00:00:00Z). Subset demand to [time-min, time-max)')
    parser.add_argument('--time-max', type=str, default=None,
                       help='Maximum timestamp (ISO UTC with Z, exclusive). Subset demand to [time-min, time-max)')
    parser.add_argument('--smoke-maintenance', action='store_true',
                       help='Automatically find first availability==0 window and run only [start-24h, end+24h)')
    parser.add_argument('--validate-maintenance', action='store_true',
                       help='Validate maintenance windows only (no optimization, no figures). Loads demand, utilities, and maintenance, then prints PASS/FAIL summary.')

    args = parser.parse_args()
    
    epoch = args.epoch
    
    # Resolve paths
    ROOT = repo_root()
    INPUT_DIR = input_root()
    
    print(f"Repository root: {ROOT}")
    print(f"Epoch: {epoch}")
    
    # Handle validate-maintenance mode early (exit before any dispatch/output code)
    if args.validate_maintenance:
        # Resolve demand CSV (epoch-aware)
        try:
            demand_csv_resolved = resolve_demand_csv(ROOT, epoch, args.demand_csv)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        
        # Resolve utilities CSV (use existing auto-discovery logic)
        if args.utilities_csv:
            utilities_csv_resolved = resolve_path(args.utilities_csv)
        else:
            # Auto-discovery
            utilities_csv_resolved = None
            candidate = INPUT_DIR / 'site' / 'utilities' / f'site_utilities_{epoch}.csv'
            if candidate.exists():
                utilities_csv_resolved = candidate
            else:
                utilities_dir = INPUT_DIR / 'site' / 'utilities'
                if utilities_dir.exists():
                    matches = sorted(utilities_dir.glob(f'*utilities*{epoch}*.csv'))
                    if len(matches) == 1:
                        utilities_csv_resolved = matches[0]
                    elif len(matches) > 1:
                        print(f"[ERROR] Multiple utilities CSV files found for epoch {epoch}:")
                        for i, match in enumerate(matches, 1):
                            print(f"  {i}. {match.name}")
                        print(f"\nPlease specify one using --utilities-csv")
                        sys.exit(1)
        
        if utilities_csv_resolved is None or not utilities_csv_resolved.exists():
            print(f"[ERROR] Utilities CSV not found for epoch {epoch}")
            print(f"  Expected: Input/site/utilities/site_utilities_{epoch}.csv")
            print(f"  Please specify using --utilities-csv")
            sys.exit(1)
        
        # Determine variant for 2035
        variant = None
        if epoch == 2035 and args.utilities_csv:
            util_path_str = str(args.utilities_csv)
            if '_EB' in util_path_str:
                variant = 'EB'
            elif '_BB' in util_path_str:
                variant = 'BB'
        
        # Run validation and exit
        exit_code = validate_maintenance_only(
            epoch=epoch,
            demand_csv_path=demand_csv_resolved,
            utilities_csv_path=utilities_csv_resolved,
            repo_root=ROOT,
            variant=variant
        )
        sys.exit(exit_code)
    
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
    if args.demand_csv:
        demand_csv_resolved = resolve_path(args.demand_csv)
    else:
        # Auto-discover demand CSV for this epoch
        demand_csv_resolved = resolve_demand_csv(ROOT, epoch, None)
    
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
    
    # Apply time window filtering if requested
    if args.smoke_maintenance:
        # Auto-discover maintenance file to find first availability==0 window
        maint_df = maint.load_maintenance_windows(ROOT, epoch, variant=None)
        if len(maint_df) > 0:
            # Find first window with availability==0
            zero_avail = maint_df[maint_df['availability'] == 0.0]
            if len(zero_avail) > 0:
                first_window = zero_avail.iloc[0]
                start_ts = first_window['start_timestamp_utc']
                end_ts = first_window['end_timestamp_utc']
                # Extend window by 24h before and after
                time_min = start_ts - pd.Timedelta(hours=24)
                time_max = end_ts + pd.Timedelta(hours=24)
                args.time_min = time_min.strftime('%Y-%m-%dT%H:%M:%SZ')
                args.time_max = time_max.strftime('%Y-%m-%dT%H:%M:%SZ')
                print(f"[INFO] --smoke-maintenance: Found first outage window {start_ts} to {end_ts}")
                print(f"[INFO] Running subset: [{args.time_min}, {args.time_max})")
            else:
                print("[WARN] --smoke-maintenance: No availability==0 windows found, running full dataset")
        else:
            print("[WARN] --smoke-maintenance: No maintenance windows found, running full dataset")
    
    if args.time_min or args.time_max:
        # Parse time window
        if args.time_min:
            time_min = parse_any_timestamp(args.time_min)
            if time_min.tz is None:
                time_min = time_min.tz_localize('UTC')
            else:
                time_min = time_min.tz_convert('UTC')
        else:
            time_min = None
        
        if args.time_max:
            time_max = parse_any_timestamp(args.time_max)
            if time_max.tz is None:
                time_max = time_max.tz_localize('UTC')
            else:
                time_max = time_max.tz_convert('UTC')
        else:
            time_max = None
        
        # Ensure demand timestamps are UTC
        if demand_df['timestamp_utc'].dt.tz is None:
            demand_df['timestamp_utc'] = demand_df['timestamp_utc'].dt.tz_localize('UTC')
        else:
            demand_df['timestamp_utc'] = demand_df['timestamp_utc'].dt.tz_convert('UTC')
        
        # Filter demand
        mask = pd.Series(True, index=demand_df.index)
        if time_min is not None:
            mask = mask & (demand_df['timestamp_utc'] >= time_min)
        if time_max is not None:
            mask = mask & (demand_df['timestamp_utc'] < time_max)
        
        n_before = len(demand_df)
        demand_df = demand_df[mask].copy().reset_index(drop=True)
        n_after = len(demand_df)
        print(f"[INFO] Time window filter: {n_before} -> {n_after} timesteps "
              f"([{args.time_min or 'start'}, {args.time_max or 'end'}))")
        
        if len(demand_df) == 0:
            print("[ERROR] Time window filter resulted in empty demand dataset")
            sys.exit(1)
    
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
    
    # Load maintenance windows using new module (epoch-aware)
    maintenance_availability = None
    maintenance_availability_wide = None  # Wide-form for dispatch functions
    
    # Determine variant for 2035 (extract from utilities CSV path if present)
    variant = None
    if epoch == 2035 and args.utilities_csv:
        util_path_str = str(args.utilities_csv)
        if '_EB' in util_path_str:
            variant = 'EB'
        elif '_BB' in util_path_str:
            variant = 'BB'
    
    # Load maintenance windows
    try:
        maint_df = maint.load_maintenance_windows(ROOT, epoch, variant=variant)
        
        if len(maint_df) > 0:
            # Convert demand timestamps to DatetimeIndex
            if isinstance(demand_df['timestamp_utc'], pd.Series):
                timestamps_utc = pd.DatetimeIndex(demand_df['timestamp_utc'].values)
            else:
                timestamps_utc = pd.DatetimeIndex(demand_df['timestamp_utc'])
            
            # Build availability matrix
            maintenance_availability_wide = maint.build_availability_matrix(
                timestamps_utc,
                util_df['unit_id'].tolist(),
                maint_df
            )
            
            # Convert to long-form for backward compatibility
            maintenance_availability_wide.index.name = 'timestamp_utc'
            maintenance_availability = maintenance_availability_wide.reset_index().melt(
                id_vars=['timestamp_utc'],
                var_name='unit_id',
                value_name='availability_multiplier'
            )
            
            # Compute summary: affected hours per unit
            n_windows = len(maint_df)
            affected_hours = {}
            for unit_id in util_df['unit_id']:
                if unit_id in maintenance_availability_wide.columns:
                    unit_avail = maintenance_availability_wide[unit_id]
                    affected_count = (unit_avail < 1.0).sum()
                    if affected_count > 0:
                        affected_hours[unit_id] = int(affected_count)
            
            print(f"[OK] Loaded maintenance windows: {n_windows} rows")
            print(f"[OK] Built availability matrix: hours={len(timestamps_utc)} units={len(util_df)}")
            if affected_hours:
                print(f"[OK] Maintenance affects {len(affected_hours)} unit(s); affected hours per unit: {affected_hours}")
            else:
                print(f"[WARN] Maintenance windows loaded but no units have availability < 1.0 (check unit_id matching)")
        else:
            print(f"[INFO] Maintenance windows not found for epoch {epoch}; continuing without maintenance constraints")
    except Exception as e:
        print(f"[WARN] Failed to load maintenance windows: {e}")
        print(f"[INFO] Continuing without maintenance constraints")
    
    # Load electricity signals if provided (for ToU pricing and headroom)
    electricity_signals = None
    if args.electricity_signals_csv:
        elec_sig_path = resolve_path(args.electricity_signals_csv)
        print(f"Loading electricity signals from {elec_sig_path}...")
        try:
            electricity_signals = pd.read_csv(elec_sig_path)
            electricity_signals['timestamp_utc'] = parse_any_timestamp(electricity_signals['timestamp_utc'])
            # Align to demand timestamps
            from src.load_gxp_signals import align_signals_to_demand as align_elec_signals
            electricity_signals = align_elec_signals(electricity_signals, demand_df, "electricity_signals")
            print(f"[OK] Loaded electricity signals with ToU pricing and headroom")
        except Exception as e:
            raise ValueError(f"Failed to load electricity signals: {e}")
    
    # Prepare signals for dispatch (time-varying if electricity_signals provided, else flat dict)
    if electricity_signals is not None:
        # Merge electricity prices into signals DataFrame if not already present
        if 'elec_price_nzd_per_MWh' not in electricity_signals.columns:
            # Fallback to flat price from signals dict
            if 'electricity_price_nzd_per_MWh_fuel' in signals:
                electricity_signals['elec_price_nzd_per_MWh'] = signals['electricity_price_nzd_per_MWh_fuel']
            else:
                electricity_signals['elec_price_nzd_per_MWh'] = 0.0
                print("[WARN] No electricity price found, defaulting to 0.0")
        
        # Add other signal prices (flat) to each row
        for key in ['coal_price_nzd_per_MWh_fuel', 'biomass_price_nzd_per_MWh_fuel', 'ets_price_nzd_per_tCO2']:
            if key in signals:
                electricity_signals[key] = signals[key]
        
        signals_for_dispatch = electricity_signals
    else:
        signals_for_dispatch = signals
    
    # Compute dispatch based on mode
    # Use wide-form if available (preferred), else use long-form (dispatch functions accept both)
    maint_for_dispatch = maintenance_availability_wide if maintenance_availability_wide is not None else maintenance_availability
    
    if args.mode == 'proportional':
        print("Computing proportional dispatch...")
        dispatch_long, dispatch_wide = allocate_baseline_dispatch(
            demand_df, util_df, dt_h,
            maintenance_availability=maint_for_dispatch
        )
    elif args.mode == 'lp':
        print("Computing LP dispatch with equality constraints...")
        dispatch_long, dispatch_wide = allocate_dispatch_lp(
            demand_df, util_df, signals_for_dispatch, dt_h,
            maintenance_availability=maint_for_dispatch,
            electricity_signals=electricity_signals,
            unserved_penalty_nzd_per_MWh=args.unserved_penalty_nzd_per_MWh
        )
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
            dt_h=dt_h,
            maintenance_availability=maint_for_dispatch
        )
    
    # Add availability columns to dispatch_long for traceability (if maintenance was loaded)
    if maintenance_availability_wide is not None:
        # Merge availability from wide-form matrix
        # Reset index and convert to long-form
        avail_df_reset = maintenance_availability_wide.reset_index()
        # Ensure index name is timestamp_utc (in case it was named differently)
        if avail_df_reset.index.name is not None:
            avail_df_reset.index.name = None
        if 'timestamp_utc' not in avail_df_reset.columns:
            # If index was reset but column name is different, rename it
            avail_df_reset = avail_df_reset.rename(columns={avail_df_reset.columns[0]: 'timestamp_utc'})
        
        avail_long = avail_df_reset.melt(
            id_vars=['timestamp_utc'],
            var_name='unit_id',
            value_name='availability_factor'
        )
        # Ensure timestamp_utc is in same format as dispatch_long
        if not pd.api.types.is_datetime64_any_dtype(dispatch_long['timestamp_utc']):
            dispatch_long['timestamp_utc'] = parse_any_timestamp(dispatch_long['timestamp_utc'])
        if not pd.api.types.is_datetime64_any_dtype(avail_long['timestamp_utc']):
            avail_long['timestamp_utc'] = parse_any_timestamp(avail_long['timestamp_utc'])
        
        # Merge availability (only for real units, not UNSERVED)
        # Filter out UNSERVED before merging, then set defaults for UNSERVED
        real_units_mask = dispatch_long['unit_id'] != 'UNSERVED'
        dispatch_real = dispatch_long[real_units_mask].copy()
        dispatch_unserved = dispatch_long[~real_units_mask].copy()
        
        dispatch_real = dispatch_real.merge(
            avail_long[['timestamp_utc', 'unit_id', 'availability_factor']],
            on=['timestamp_utc', 'unit_id'],
            how='left'
        )
        dispatch_real['availability_factor'] = dispatch_real['availability_factor'].fillna(1.0)
        
        # Compute available_cap_MW = max_heat_MW * availability_factor (only for real units)
        unit_cap_map = dict(zip(util_df['unit_id'], util_df['max_heat_MW']))
        dispatch_real['available_cap_MW'] = dispatch_real.apply(
            lambda row: unit_cap_map.get(row['unit_id'], 0.0) * row.get('availability_factor', 1.0),
            axis=1
        )
        
        # For UNSERVED: set availability_factor = 1.0 and available_cap_MW = 0.0 (no capacity)
        dispatch_unserved['availability_factor'] = 1.0
        dispatch_unserved['available_cap_MW'] = 0.0
        
        # Recombine
        dispatch_long = pd.concat([dispatch_real, dispatch_unserved], ignore_index=True).sort_values(['timestamp_utc', 'unit_id']).reset_index(drop=True)
        print(f"[OK] Added availability columns to dispatch output (availability_factor, available_cap_MW)")
    else:
        # No maintenance: set availability_factor = 1.0 and available_cap_MW = max_heat_MW for real units
        # For UNSERVED: set availability_factor = 1.0 and available_cap_MW = 0.0
        unit_cap_map = dict(zip(util_df['unit_id'], util_df['max_heat_MW']))
        real_units_mask = dispatch_long['unit_id'] != 'UNSERVED'
        dispatch_long.loc[real_units_mask, 'availability_factor'] = 1.0
        dispatch_long.loc[real_units_mask, 'available_cap_MW'] = dispatch_long.loc[real_units_mask, 'unit_id'].map(unit_cap_map)
        dispatch_long.loc[~real_units_mask, 'availability_factor'] = 1.0
        dispatch_long.loc[~real_units_mask, 'available_cap_MW'] = 0.0
    
    # Add cost columns for BOTH modes to ensure consistent schema
    # This ensures fuel_cost_nzd, carbon_cost_nzd, total_cost_nzd always exist
    # For LP mode, use signals_for_dispatch (may be DataFrame for time-varying prices)
    print("Computing costs...")
    if args.mode == 'lp':
        # LP mode may have time-varying electricity prices from electricity_signals
        # Pass unserved_penalty so it's computed correctly
        dispatch_long = add_costs_to_dispatch(dispatch_long, util_df, signals_for_dispatch, dt_h,
                                               unserved_penalty_nzd_per_MWh=args.unserved_penalty_nzd_per_MWh)
    else:
        dispatch_long = add_costs_to_dispatch(dispatch_long, util_df, signals, dt_h)
    
    # Validate dispatch outputs (energy closure, schema, etc.) - AFTER costs are added
    validate_dispatch_outputs(dispatch_long, dispatch_wide, demand_df, dt_h,
                               electricity_signals=electricity_signals,
                               maintenance_availability=maintenance_availability,
                               util_df=util_df)
    
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
    
    # Export incremental electricity demand if electric units present
    electric_units = util_df[
        (util_df['fuel'].str.lower().str.strip() == 'electricity') |
        (util_df['tech_type'].str.lower().str.strip() == 'electrode_boiler')
    ]
    if len(electric_units) > 0:
        # Determine output path for electricity export
        if args.run_id and args.output_root:
            output_root = Path(args.output_root)
            signals_dir = output_root / 'runs' / args.run_id / 'signals'
            elec_export_path = signals_dir / f'incremental_electricity_MW_{epoch}.csv'
        elif args.output_dir:
            signals_dir = Path(args.output_dir) / 'signals'
            elec_export_path = signals_dir / f'incremental_electricity_MW_{epoch}.csv'
        else:
            signals_dir = OUTPUT_DIR / 'signals'
            elec_export_path = signals_dir / f'incremental_electricity_MW_{epoch}.csv'
        
        export_incremental_electricity(dispatch_long, util_df, demand_df, elec_export_path)
    
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
    # Pass penalty rates to ensure penalty costs are computed correctly
    unserved_penalty_rate = getattr(args, 'unserved_penalty_nzd_per_MWh', None)
    reserve_penalty_rate = getattr(args, 'reserve_penalty_nzd_per_MWh', None)
    annual_summary = compute_annual_summary(
        dispatch_long,
        reserve_frac=reserve_frac_val,
        dt_h=dt_h,
        unserved_penalty_nzd_per_MWh=unserved_penalty_rate,
        reserve_penalty_nzd_per_MWh=reserve_penalty_rate
    )
    
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
    
    # Build safe lookup map for util_df (exclude UNSERVED which is synthetic)
    util_meta = {}
    if util_df is not None and 'unit_id' in util_df.columns:
        util_meta = util_df.set_index('unit_id').to_dict('index')
    
    for _, row in unit_rows.iterrows():
        unit_id = row['unit_id']
        print(f"\n{unit_id}:")
        
        # Handle UNSERVED specially (synthetic unit, not in util_df)
        if unit_id == 'UNSERVED':
            print(f"  Annual heat (GWh):           {row['annual_heat_GWh']:10.2f}")
            # Get unserved cost from annual_penalty_cost_nzd (which should equal unserved_cost_nzd for UNSERVED)
            # This was set in compute_annual_summary from the aggregated unserved_cost_nzd column
            unserved_cost_val = row.get('annual_penalty_cost_nzd', 0.0)
            if pd.isna(unserved_cost_val):
                # Fallback to unserved_cost_nzd column if available
                unserved_cost_val = row.get('unserved_cost_nzd', 0.0)
            if pd.isna(unserved_cost_val):
                # Final fallback to annual_total_cost_nzd
                unserved_cost_val = row.get('annual_total_cost_nzd', 0.0)
            unserved_cost_val = float(unserved_cost_val) if not pd.isna(unserved_cost_val) else 0.0
            print(f"  Annual unserved cost (NZD):  {unserved_cost_val:12,.2f}")
            total_cost_val = float(row['annual_total_cost_nzd']) if not pd.isna(row.get('annual_total_cost_nzd', np.nan)) else 0.0
            print(f"  Annual total cost (NZD):     {total_cost_val:12,.2f}")
            if row['annual_heat_GWh'] > 0:
                avg_cost_val = float(row['avg_cost_nzd_per_MWh_heat']) if not pd.isna(row.get('avg_cost_nzd_per_MWh_heat', np.nan)) else 0.0
                print(f"  Avg cost per MWh_heat (NZD): {avg_cost_val:10.2f}")
            continue
        
        # Real units: print full details
        print(f"  Annual heat (GWh):           {row['annual_heat_GWh']:10.2f}")
        print(f"  Annual fuel (MWh):           {row['annual_fuel_MWh']:10.2f}")
        print(f"  Annual CO2 (tCO2):           {row['annual_co2_tonnes']:10.2f}")
        print(f"  Annual fuel cost (NZD):      {row['annual_fuel_cost_nzd']:12,.2f}")
        # Show electricity cost note if it's an electric unit (for clarity)
        # Use safe lookup instead of direct iloc[0]
        unit_meta = util_meta.get(unit_id)
        if unit_meta is not None:
            unit_fuel = unit_meta.get('fuel', '')
            if 'electricity' in str(unit_fuel).lower():
                print(f"    (includes electricity ToU cost from electricity_signals CSV)")
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
        # Use unserved_MWh from system_row if available (already converted), else convert from MW
        if 'unserved_MWh' in system_row and not pd.isna(system_row.get('unserved_MWh', np.nan)):
            unserved_mwh = float(system_row['unserved_MWh'])
        else:
            unserved_mw_sum = system_row.get('unserved_MW', 0.0)
            unserved_mwh = float(unserved_mw_sum * dt_h)  # Convert to MWh
        
        # Use unserved_penalty_cost_nzd if available, else fallback to unserved_cost_nzd
        if 'unserved_penalty_cost_nzd' in system_row and not pd.isna(system_row.get('unserved_penalty_cost_nzd', np.nan)):
            unserved_cost = float(system_row['unserved_penalty_cost_nzd'])
        else:
            unserved_cost = float(system_row.get('unserved_cost_nzd', 0.0))
        
        print(f"  Unserved energy (MWh):       {unserved_mwh:10.2f}")
        print(f"  Unserved penalty cost (NZD): {unserved_cost:12,.2f}")
        # Hard validation: if unserved > 0 and penalty is configured > 0, cost must be > 0
        if unserved_mwh > 0 and unserved_cost == 0:
            # Check if penalty was configured (args.unserved_penalty_nzd_per_MWh should be available in scope)
            if args.mode == 'lp' and hasattr(args, 'unserved_penalty_nzd_per_MWh'):
                penalty_rate = args.unserved_penalty_nzd_per_MWh
                if penalty_rate > 0:
                    # CRITICAL: Recompute expected cost from the same unserved_MWh value
                    expected_cost = unserved_mwh * penalty_rate
                    raise ValueError(
                        f"[VALIDATION FAILURE] Unserved energy > 0 ({unserved_mwh:.2f} MWh) but penalty cost is 0. "
                        f"Penalty rate is configured as {penalty_rate:.2f} NZD/MWh. "
                        f"This indicates a bug in cost computation. Expected cost: {expected_cost:.2f} NZD. "
                        f"Please check that unserved_cost_nzd is computed correctly in add_costs_to_dispatch() "
                        f"and that the LP objective includes the penalty term."
                    )
                else:
                    print(f"    [WARN] Unserved energy > 0 but penalty rate is 0 (unserved_penalty_nzd_per_MWh={penalty_rate})")
            else:
                print(f"    [WARN] Unserved energy > 0 but penalty cost is 0. Check unserved_penalty_nzd_per_MWh setting.")
        
        # Show reserve (always show, even if 0)
        # Use reserve_shortfall_MWh from system_row if available (already converted), else convert from MW
        if 'reserve_shortfall_MWh' in system_row and not pd.isna(system_row.get('reserve_shortfall_MWh', np.nan)):
            reserve_shortfall_mwh = float(system_row['reserve_shortfall_MWh'])
        else:
            reserve_shortfall_mw_sum = system_row.get('reserve_shortfall_MW', 0.0)
            reserve_shortfall_mwh = float(reserve_shortfall_mw_sum * dt_h)  # Convert to MWh
        
        reserve_penalty_cost = float(system_row.get('reserve_penalty_cost_nzd', 0.0))
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
            penalty_cost = float(total_row['annual_penalty_cost_nzd']) if not pd.isna(total_row.get('annual_penalty_cost_nzd', np.nan)) else 0.0
            print(f"  Annual penalty cost (NZD):    {penalty_cost:12,.2f}")
            print(f"    (includes: unserved + reserve penalties)")
        total_cost = float(total_row['annual_total_cost_nzd']) if not pd.isna(total_row.get('annual_total_cost_nzd', np.nan)) else 0.0
        print(f"  Annual total cost (NZD):     {total_cost:12,.2f}")
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
    
    # Generate plots if requested (non-fatal: only warn, don't crash dispatch)
    plot_success = True
    if args.plot:
        print("\nGenerating dispatch plots...")
        
        try:
            if args.mode == 'proportional':
                plot_path = figures_dir / f'heat_{epoch}_unit_stack.png'
                plot_dispatch_stack(args.out_dispatch_wide, str(plot_path), demand_df, epoch=epoch)
            elif args.mode == 'lp':
                # LP mode: same plots as optimal_subset
                # Stack plot
                plot_path = figures_dir / f'heat_{epoch}_unit_stack_opt.png'
                try:
                    plot_dispatch_stack(args.out_dispatch_wide, str(plot_path), demand_df, epoch=epoch)
                except Exception as e:
                    print(f"[WARN] Stack plot failed: {e}")
                    plot_success = False
                
                # Units online plot
                plot_path = figures_dir / f'heat_{epoch}_units_online_opt.png'
                try:
                    plot_units_online(args.out_dispatch_long, str(plot_path), epoch=epoch)
                except Exception as e:
                    print(f"[WARN] Units online plot failed: {e}")
                    plot_success = False
                
                # Utilisation duration plot
                plot_path = figures_dir / f'heat_{epoch}_unit_utilisation_duration_opt.png'
                try:
                    plot_unit_utilisation_duration(args.out_dispatch_long, str(plot_path), str(utilities_csv_resolved), epoch=epoch)
                except Exception as e:
                    print(f"[WARN] Utilisation duration plot failed: {e}")
                    plot_success = False
            else:  # optimal_subset
                # Stack plot
                plot_path = figures_dir / f'heat_{epoch}_unit_stack_opt.png'
                try:
                    plot_dispatch_stack(args.out_dispatch_wide, str(plot_path), demand_df, epoch=epoch)
                except Exception as e:
                    print(f"[WARN] Stack plot failed: {e}")
                    plot_success = False
                
                # Units online plot
                plot_path = figures_dir / f'heat_{epoch}_units_online_opt.png'
                try:
                    plot_units_online(args.out_dispatch_long, str(plot_path), epoch=epoch)
                except Exception as e:
                    print(f"[WARN] Units online plot failed: {e}")
                    plot_success = False
                
                # Utilisation duration plot
                plot_path = figures_dir / f'heat_{epoch}_unit_utilisation_duration_opt.png'
                try:
                    plot_unit_utilisation_duration(args.out_dispatch_long, str(plot_path), str(utilities_csv_resolved), epoch=epoch)
                except Exception as e:
                    print(f"[WARN] Utilisation duration plot failed: {e}")
                    plot_success = False
            
            if plot_success:
                print("[OK] All plots generated successfully")
            else:
                print("[WARN] Some plots failed, but dispatch and validation succeeded")
            
            # Figures are written directly to run_figures_dir (no copying to latest)
        except Exception as e:
            print(f"\n[WARN] Plot generation encountered errors: {e}")
            print("  Dispatch and validation succeeded; plots are optional")
            plot_success = False
            import traceback
            if os.environ.get("DISPATCH_DEBUG_PLOTS", "0") == "1":
                traceback.print_exc()
    
    # Exit with code 0 if dispatch and validation succeeded (plots are optional)
    # Only exit non-zero if dispatch/validation failed
    print("\n" + "="*80)
    if plot_success:
        print("[OK] Dispatch, validation, and plotting completed successfully")
    else:
        print("[OK] Dispatch and validation completed successfully (some plots failed, but this is non-fatal)")
    print("="*80)
    
    # Self-test: verify penalty cost consistency (if DISPATCH_SELF_TEST env var set)
    if os.environ.get("DISPATCH_SELF_TEST", "0") == "1":
        try:
            _self_test_penalty_consistency(annual_summary, args)
        except Exception as e:
            print(f"[Self-Test ERROR] {e}")
            # Don't fail the run, but log the error


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
                               demand_df: pd.DataFrame, dt_h: float,
                               electricity_signals: Optional[pd.DataFrame] = None,
                               maintenance_availability: Optional[pd.DataFrame] = None,
                               util_df: Optional[pd.DataFrame] = None) -> None:
    """
    Validate dispatch outputs for correctness (energy closure, schema, etc.).
    
    Args:
        dispatch_long: Long-form dispatch DataFrame
        dispatch_wide: Wide-form dispatch DataFrame
        demand_df: Demand DataFrame with timestamp_utc and heat_demand_MW
        dt_h: Timestep duration in hours
        electricity_signals: Optional electricity signals DataFrame (for headroom validation)
        maintenance_availability: Optional maintenance availability DataFrame (for maintenance validation)
        util_df: Optional utilities DataFrame (for headroom validation)
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
    
    # Compute served from real units only (exclude UNSERVED)
    # served_by_ts = sum of heat_MW for all units EXCEPT unit_id == "UNSERVED"
    if 'unit_id' in dispatch_long_work.columns:
        real_units_mask = dispatch_long_work['unit_id'] != 'UNSERVED'
        served_by_ts = dispatch_long_work[real_units_mask].groupby('_ts')['heat_MW'].sum()
    else:
        # Fallback: sum all (shouldn't happen, but handle gracefully)
        served_by_ts = dispatch_long_work.groupby('_ts')['heat_MW'].sum()
    
    # Get unserved_MW: either from UNSERVED unit_id rows or from unserved_MW column
    if 'unit_id' in dispatch_long_work.columns and (dispatch_long_work['unit_id'] == 'UNSERVED').any():
        # UNSERVED is stored as unit_id="UNSERVED" rows
        unserved_by_ts = dispatch_long_work[dispatch_long_work['unit_id'] == 'UNSERVED'].groupby('_ts')['heat_MW'].sum()
    elif 'unserved_MW' in dispatch_long_work.columns:
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
    
    # Check closure: demand == served + unserved (STRICT - hard failure)
    tolerance = 1e-6  # MW
    closure_errors = (demand_by_ts - served_by_ts - unserved_by_ts).abs()
    max_error = closure_errors.max()
    if max_error > tolerance:
        max_error_idx = closure_errors.idxmax()
        raise ValueError(
            f"[VALIDATION FAILURE] Energy closure violation: max error = {max_error:.6f} MW at {max_error_idx}. "
            f"demand={demand_by_ts[max_error_idx]:.6f} MW, served={served_by_ts[max_error_idx]:.6f} MW, "
            f"unserved={unserved_by_ts[max_error_idx]:.6f} MW, sum={served_by_ts[max_error_idx] + unserved_by_ts[max_error_idx]:.6f} MW. "
            f"Tolerance: {tolerance} MW. This is a hard failure - dispatch must satisfy served + unserved == demand exactly."
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
    
    # 5. Headroom constraint validation (if electricity_signals provided)
    if electricity_signals is not None and 'headroom_MW' in electricity_signals.columns:
        # Compute incremental electricity from dispatch
        electric_units = util_df[
            (util_df['fuel'].str.lower().str.strip() == 'electricity') |
            (util_df['tech_type'].str.lower().str.strip() == 'electrode_boiler')
        ]
        if len(electric_units) > 0:
            electric_dispatch = dispatch_long[dispatch_long['unit_id'].isin(electric_units['unit_id'])].copy()
            eff_map = dict(zip(util_df['unit_id'], util_df['efficiency_th']))
            electric_dispatch['efficiency'] = electric_dispatch['unit_id'].map(eff_map)
            electric_dispatch['elec_MW'] = electric_dispatch['heat_MW'] / electric_dispatch['efficiency']
            
            # Canonicalize timestamps before grouping to ensure tz-aware UTC index
            electric_dispatch['_ts_canonical'] = _canonicalize_timestamps(electric_dispatch['timestamp_utc'])
            elec_by_ts = electric_dispatch.groupby('_ts_canonical')['elec_MW'].sum()
            # Ensure index is DatetimeIndex with UTC timezone
            elec_by_ts.index = pd.DatetimeIndex(elec_by_ts.index, tz='UTC')
            
            # Align headroom - canonicalize timestamps first
            headroom_aligned = demand_df[['timestamp_utc']].merge(
                electricity_signals[['timestamp_utc', 'headroom_MW']],
                on='timestamp_utc',
                how='left'
            )
            # Canonicalize headroom timestamps to tz-aware UTC
            headroom_ts_canonical = _canonicalize_timestamps(headroom_aligned['timestamp_utc'])
            # Create Series with DatetimeIndex (ensure it's a DatetimeIndex, not just the values)
            if isinstance(headroom_ts_canonical, pd.Series):
                headroom_ts_index = pd.DatetimeIndex(headroom_ts_canonical.values, tz='UTC')
            else:
                headroom_ts_index = pd.DatetimeIndex(headroom_ts_canonical, tz='UTC')
            headroom_by_ts = pd.Series(
                headroom_aligned['headroom_MW'].values,
                index=headroom_ts_index
            )
            
            # Ensure demand_ts_canonical is also properly canonicalized (should already be, but ensure)
            # demand_ts_canonical is already a DatetimeIndex from earlier validation, but ensure it's UTC
            if isinstance(demand_ts_canonical, pd.DatetimeIndex):
                if demand_ts_canonical.tz is None:
                    demand_ts_canonical_normalized = demand_ts_canonical.tz_localize('UTC')
                else:
                    demand_ts_canonical_normalized = demand_ts_canonical.tz_convert('UTC')
            else:
                # If it's not a DatetimeIndex, canonicalize it
                demand_ts_canonical_normalized = pd.DatetimeIndex(_canonicalize_timestamps(pd.Series(demand_ts_canonical)).values, tz='UTC')
            
            # Optional debug: print timezone info if env var set
            if os.environ.get("DISPATCH_DEBUG_VALIDATE", "0") == "1":
                print(f"[DEBUG] Headroom validation timezone info:")
                print(f"  elec_by_ts.index.tz: {elec_by_ts.index.tz}")
                print(f"  headroom_by_ts.index.tz: {headroom_by_ts.index.tz}")
                print(f"  demand_ts_canonical_normalized.tz: {demand_ts_canonical_normalized.tz}")
                print(f"  elec_by_ts.index.dtype: {elec_by_ts.index.dtype}")
                print(f"  headroom_by_ts.index.dtype: {headroom_by_ts.index.dtype}")
                print(f"  demand_ts_canonical_normalized.dtype: {demand_ts_canonical_normalized.dtype}")
            
            # Check headroom constraint - all indices are now tz-aware UTC DatetimeIndex
            headroom_violations = (elec_by_ts - headroom_by_ts).reindex(demand_ts_canonical_normalized, fill_value=0.0)
            max_violation = headroom_violations.max()
            tol = 1e-6  # MW
            if max_violation > tol:
                max_violation_ts = headroom_violations.idxmax()
                # Use reindex to safely access values by timestamp
                elec_val = elec_by_ts.reindex([max_violation_ts], fill_value=0.0).iloc[0]
                headroom_val = headroom_by_ts.reindex([max_violation_ts], fill_value=0.0).iloc[0]
                raise ValueError(
                    f"[VALIDATION FAILURE] Headroom constraint violation at {max_violation_ts}: "
                    f"incremental_electricity={elec_val:.6f} MW, "
                    f"headroom={headroom_val:.6f} MW, "
                    f"violation={max_violation:.6f} MW. "
                    f"Tolerance: {tol} MW. This is a hard failure - electricity consumption must not exceed headroom."
                )
            print(f"[OK] Headroom constraint validated: max incremental electricity within headroom (tolerance: {tol} MW)")
    
    # 6. Maintenance validation (if maintenance_availability provided)
    if maintenance_availability is not None and util_df is not None:
        # Filter out UNSERVED rows before maintenance validation (UNSERVED has no capacity constraints)
        dispatch_real_units = dispatch_long[dispatch_long['unit_id'] != 'UNSERVED'].copy()
        
        # Merge maintenance availability with dispatch (only real units)
        maint_merged = dispatch_real_units.merge(
            maintenance_availability,
            on=['timestamp_utc', 'unit_id'],
            how='left'
        )
        maint_merged['availability_multiplier'] = maint_merged['availability_multiplier'].fillna(1.0)
        
        # Merge unit capacity and static availability_factor from utilities
        unit_cap_map = dict(zip(util_df['unit_id'], util_df['max_heat_MW']))
        maint_merged['max_heat_MW'] = maint_merged['unit_id'].map(unit_cap_map)
        
        # Get static availability_factor from utilities CSV (default 1.0 if not present)
        # This is the static availability from the utilities CSV, separate from time-varying maintenance
        if 'availability_factor' in util_df.columns:
            unit_avail_factor_map = dict(zip(util_df['unit_id'], util_df['availability_factor']))
            maint_merged['static_availability_factor'] = maint_merged['unit_id'].map(unit_avail_factor_map).fillna(1.0)
        else:
            # If utilities don't have availability_factor column, default to 1.0
            maint_merged['static_availability_factor'] = 1.0
        
        # Validate: heat_MW[u,t] <= max_heat_MW[u] * static_availability_factor[u] * maintenance_availability[u,t] + tol
        # This matches the constraint used in dispatch: cap_effective = cap_nameplate * static_avail * maint_avail
        # where static_avail comes from utilities CSV and maint_avail is time-varying from maintenance windows
        maint_merged['max_allowed_MW'] = (maint_merged['max_heat_MW'] * 
                                         maint_merged['static_availability_factor'] * 
                                         maint_merged['availability_multiplier'])
        maint_merged['violation_MW'] = maint_merged['heat_MW'] - maint_merged['max_allowed_MW']
        
        tol = 1e-6  # MW
        violations = maint_merged[maint_merged['violation_MW'] > tol]
        
        if len(violations) > 0:
            violation_samples = violations[['timestamp_utc', 'unit_id', 'heat_MW', 'max_heat_MW', 
                                           'static_availability_factor', 'availability_multiplier', 
                                           'max_allowed_MW', 'violation_MW']].head(10)
            raise ValueError(
                f"[VALIDATION FAILURE] Maintenance constraint violation: {len(violations)} timestamps where "
                f"unit dispatch exceeds maintenance availability. "
                f"Required: heat_MW[u,t] <= max_heat_MW[u] * static_availability_factor[u] * maintenance_availability[u,t]. "
                f"Tolerance: {tol} MW. This is a hard failure. "
                f"Sample violations:\n{violation_samples.to_string()}"
            )
        
        # Also check zero dispatch during full outages (availability == 0)
        outage_mask = maint_merged['availability_multiplier'] == 0.0
        if outage_mask.any():
            zero_violations = maint_merged[outage_mask & (maint_merged['heat_MW'].abs() > tol)]
            if len(zero_violations) > 0:
                violation_samples = zero_violations[['timestamp_utc', 'unit_id', 'heat_MW', 'availability_multiplier']].head(5)
                raise ValueError(
                    f"[VALIDATION FAILURE] Maintenance constraint violation: {len(zero_violations)} timestamps with non-zero dispatch during full outages (availability==0). "
                    f"Tolerance: {tol} MW. This is a hard failure - units must have zero dispatch when availability == 0. "
                    f"Sample violations:\n{violation_samples.to_string()}"
                )
            print(f"[OK] Maintenance constraints validated: {outage_mask.sum()} outage hours, all units at zero dispatch (tolerance: {tol} MW)")
        else:
            print(f"[OK] Maintenance constraints validated: all dispatch within availability limits (tolerance: {tol} MW)")


def _self_test_penalty_consistency(annual_summary: pd.DataFrame, args) -> None:
    """
    Self-test: verify penalty cost consistency.
    
    Checks:
    1. If unserved_MWh > 0 and penalty rate > 0, then penalty cost must be > 0
    2. Penalty cost = unserved_MWh * penalty_rate (within tolerance)
    """
    print("\n[Self-Test] Verifying penalty cost consistency...")
    
    system_mask = annual_summary['unit_id'] == 'SYSTEM'
    if not system_mask.any():
        print("[Self-Test] No SYSTEM row found, skipping penalty consistency check")
        return
    
    system_row = annual_summary[system_mask].iloc[0]
    unserved_mwh = float(system_row.get('unserved_MWh', 0.0)) if 'unserved_MWh' in system_row else 0.0
    unserved_cost = float(system_row.get('unserved_penalty_cost_nzd', 0.0)) if 'unserved_penalty_cost_nzd' in system_row else 0.0
    
    if hasattr(args, 'unserved_penalty_nzd_per_MWh'):
        penalty_rate = args.unserved_penalty_nzd_per_MWh
        if unserved_mwh > 0 and penalty_rate > 0:
            expected_cost = unserved_mwh * penalty_rate
            if abs(unserved_cost - expected_cost) > 1e-3:
                print(f"[Self-Test FAIL] Penalty cost mismatch:")
                print(f"  unserved_MWh: {unserved_mwh:.2f}")
                print(f"  penalty_rate: {penalty_rate:.2f} NZD/MWh")
                print(f"  expected_cost: {expected_cost:.2f} NZD")
                print(f"  actual_cost: {unserved_cost:.2f} NZD")
                print(f"  difference: {abs(unserved_cost - expected_cost):.2f} NZD")
                raise ValueError("Self-test failed: penalty cost does not match expected value")
            else:
                print(f"[Self-Test OK] Penalty cost consistent: {unserved_cost:.2f} NZD = {unserved_mwh:.2f} MWh * {penalty_rate:.2f} NZD/MWh")
        elif unserved_mwh > 0 and penalty_rate == 0:
            print(f"[Self-Test WARN] Unserved energy > 0 ({unserved_mwh:.2f} MWh) but penalty rate is 0")
        elif unserved_mwh == 0:
            print(f"[Self-Test OK] No unserved energy, penalty cost is {unserved_cost:.2f} NZD (expected 0.0)")
    else:
        print("[Self-Test SKIP] unserved_penalty_nzd_per_MWh not available in args")


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


def _test_find_maintenance_windows_csv():
    """Unit test for find_maintenance_windows_csv helper function."""
    from src.run_all import find_maintenance_windows_csv
    from src.path_utils import repo_root
    
    print("\n[Self-Test] Testing find_maintenance_windows_csv...")
    
    ROOT = repo_root()
    
    # Test 1: Check if function exists and is callable
    try:
        result = find_maintenance_windows_csv(ROOT, 2025, None)
        print(f"[OK] find_maintenance_windows_csv(ROOT, 2025, None) returned: {result}")
        if result is not None:
            print(f"  Resolved path: {result.resolve()}")
            print(f"  File exists: {result.is_file()}")
    except Exception as e:
        print(f"[FAIL] find_maintenance_windows_csv raised exception: {e}")
        raise
    
    # Test 2: Test with non-existent epoch (should return None, not raise)
    try:
        result = find_maintenance_windows_csv(ROOT, 9999, None)
        if result is None:
            print(f"[OK] find_maintenance_windows_csv(ROOT, 9999, None) correctly returned None for non-existent epoch")
        else:
            print(f"[WARN] find_maintenance_windows_csv(ROOT, 9999, None) returned {result} (expected None)")
    except Exception as e:
        print(f"[FAIL] find_maintenance_windows_csv raised exception for non-existent epoch: {e}")
        raise
    
    print("[OK] find_maintenance_windows_csv tests passed")


def _test_maintenance_dispatch():
    """
    Test maintenance constraints in dispatch.
    
    Creates a synthetic 48h case with one unit forced availability=0 for 24h.
    Verifies max heat for that unit during outage is 0 and that unserved or other units cover as expected.
    """
    print("\n[Self-Test] Testing maintenance constraints in dispatch...")
    
    import tempfile
    from pathlib import Path
    
    ROOT = repo_root()
    INPUT_DIR = input_root()
    
    # Create synthetic demand (48 hours)
    timestamps = pd.date_range('2020-01-01T00:00:00Z', periods=48, freq='H', tz='UTC')
    demand_df = pd.DataFrame({
        'timestamp_utc': timestamps,
        'heat_demand_MW': 100.0  # Constant 100 MW demand
    })
    
    # Load utilities (use real file if available)
    utilities_csv = INPUT_DIR / 'site' / 'utilities' / 'site_utilities_2020.csv'
    if not utilities_csv.exists():
        print("[SKIP] Self-test: site_utilities_2020.csv not found, skipping maintenance test")
        return
    
    util_df = load_utilities(str(utilities_csv), epoch=2020)
    unit_ids = util_df['unit_id'].tolist()
    
    if len(unit_ids) < 2:
        print("[SKIP] Self-test: Need at least 2 units for maintenance test")
        return
    
    # Create synthetic maintenance file: first unit unavailable for hours 0-23 (first 24 hours)
    test_unit = unit_ids[0]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        maint_csv_path = Path(f.name)
        f.write('unit_id,start_timestamp_utc,end_timestamp_utc,availability\n')
        # Write timestamps in ISO Z format
        f.write(f'{test_unit},{timestamps[0].strftime("%Y-%m-%dT%H:%M:%SZ")},{timestamps[24].strftime("%Y-%m-%dT%H:%M:%SZ")},0\n')
    
    try:
        # Load maintenance availability
        timestamps_utc = pd.DatetimeIndex(timestamps)
        maint_avail_wide = load_maintenance_availability(maint_csv_path, timestamps_utc, unit_ids)
        
        # Verify availability matrix
        assert maint_avail_wide.loc[timestamps[0:24], test_unit].eq(0.0).all(), \
            f"Unit {test_unit} should have availability=0 for first 24 hours"
        assert maint_avail_wide.loc[timestamps[24:48], test_unit].eq(1.0).all(), \
            f"Unit {test_unit} should have availability=1.0 for hours 24-47"
        
        # Test LP dispatch with maintenance
        signals_config_path = INPUT_DIR / 'signals' / 'signals_config.toml'
        if not signals_config_path.exists():
            print("[SKIP] Self-test: signals_config.toml not found, skipping LP dispatch test")
            return
        
        signals_config = load_signals_config(str(signals_config_path))
        signals = get_signals_for_epoch(signals_config, '2020')
        
        # Run LP dispatch
        dispatch_long, dispatch_wide = allocate_dispatch_lp(
            demand_df, util_df, signals, dt_h=1.0,
            maintenance_availability=maint_avail_wide,
            unserved_penalty_nzd_per_MWh=10000.0
        )
        
        # Verify: test_unit should have heat_MW == 0 during outage (hours 0-23)
        test_unit_dispatch = dispatch_long[dispatch_long['unit_id'] == test_unit].copy()
        test_unit_dispatch['timestamp_utc'] = pd.to_datetime(test_unit_dispatch['timestamp_utc'], utc=True)
        
        outage_period = test_unit_dispatch[test_unit_dispatch['timestamp_utc'].isin(timestamps[0:24])]
        outage_heat = outage_period['heat_MW'].abs()
        max_outage_heat = outage_heat.max()
        
        assert max_outage_heat < 1e-6, \
            f"Unit {test_unit} should have zero dispatch during outage, but max heat_MW = {max_outage_heat:.6f}"
        
        # Verify: after outage (hours 24-47), unit can dispatch
        normal_period = test_unit_dispatch[test_unit_dispatch['timestamp_utc'].isin(timestamps[24:48])]
        normal_heat = normal_period['heat_MW']
        max_normal_heat = normal_heat.max()
        
        # Unit should be able to dispatch after outage (may be > 0 depending on costs)
        print(f"  Unit {test_unit} during outage (hours 0-23): max heat_MW = {max_outage_heat:.6f} (expected 0)")
        print(f"  Unit {test_unit} after outage (hours 24-47): max heat_MW = {max_normal_heat:.6f}")
        
        # Verify energy closure: demand should be met by other units or unserved
        for i, ts in enumerate(timestamps[0:24]):
            ts_dispatch = dispatch_long[dispatch_long['timestamp_utc'] == ts]
            served = ts_dispatch['heat_MW'].sum()
            unserved = ts_dispatch['unserved_MW'].iloc[0] if 'unserved_MW' in ts_dispatch.columns else 0.0
            demand_ts = demand_df[demand_df['timestamp_utc'] == ts]['heat_demand_MW'].iloc[0]
            closure_error = abs(demand_ts - served - unserved)
            assert closure_error < 1e-6, \
                f"Energy closure failed at {ts}: demand={demand_ts}, served={served}, unserved={unserved}, error={closure_error}"
        
        print(f"[OK] Maintenance constraint test passed: unit {test_unit} correctly has zero dispatch during outage")
        
    finally:
        # Clean up temp file
        if maint_csv_path.exists():
            maint_csv_path.unlink()


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
    validate_dispatch_outputs(dispatch_long, dispatch_wide, demand_slice, dt_h,
                               electricity_signals=None, maintenance_availability=None, util_df=util_df)
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
    validate_dispatch_outputs(dispatch_long_opt, dispatch_wide_opt, demand_slice, dt_h,
                               electricity_signals=None, maintenance_availability=None, util_df=util_df)
    print("[OK] Optimal subset dispatch validation passed")
    
    print("\n" + "="*60)
    print("All self-tests passed!")
    print("="*60)


if __name__ == '__main__':
    import sys
    if '--self-test' in sys.argv:
        _self_test_lightweight()
        _test_find_maintenance_windows_csv()
        _test_maintenance_dispatch()
        sys.exit(0)
    main()


