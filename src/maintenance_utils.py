"""
Maintenance windows utilities for epoch-aware dispatch.

Provides functions to load and process maintenance windows CSV files
and build availability matrices for dispatch constraints.
"""

# Bootstrap: allow `python .\src\script.py` (adds repo root to sys.path)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from typing import Optional, List
from src.path_utils import repo_root, input_root
from src.time_utils import parse_any_timestamp


def load_maintenance_windows(repo_root: Path, eval_year: int, variant: Optional[str] = None) -> pd.DataFrame:
    """
    Load maintenance windows CSV for an epoch with variant support.
    
    Lookup order:
    1. If variant is not None: try maintenance_windows_{eval_year}_{variant}.csv
    2. Try canonical: maintenance_windows_{eval_year}.csv
    3. If not found: return empty DataFrame (no maintenance)
    
    Args:
        repo_root: Repository root directory
        eval_year: Evaluation year (e.g., 2020, 2025, 2028, 2035)
        variant: Optional variant string (e.g., "EB", "BB" for 2035)
        
    Returns:
        DataFrame with columns: unit_id, start_timestamp_utc, end_timestamp_utc, availability
        Empty DataFrame if file not found (no maintenance)
        
    Raises:
        ValueError: If validation fails (invalid columns, timestamps, availability values)
    """
    input_dir = repo_root / "Input"
    maint_dir = input_dir / "site" / "maintenance"
    
    # Try variant-specific first if variant provided
    maint_path = None
    if variant:
        variant_path = maint_dir / f"maintenance_windows_{eval_year}_{variant}.csv"
        if variant_path.exists():
            maint_path = variant_path
    
    # Try canonical if variant path not found
    if maint_path is None:
        canonical_path = maint_dir / f"maintenance_windows_{eval_year}.csv"
        if canonical_path.exists():
            maint_path = canonical_path
    
    # If still not found, return empty DataFrame
    if maint_path is None:
        return pd.DataFrame(columns=['unit_id', 'start_timestamp_utc', 'end_timestamp_utc', 'availability'])
    
    if not maint_path.exists():
        # Return empty DataFrame (no maintenance)
        return pd.DataFrame(columns=['unit_id', 'start_timestamp_utc', 'end_timestamp_utc', 'availability'])
    
    # Read CSV
    try:
        df = pd.read_csv(maint_path)
    except Exception as e:
        raise ValueError(f"Failed to read maintenance windows CSV {maint_path}: {e}")
    
    # Validate required columns
    required_cols = ['unit_id', 'start_timestamp_utc', 'end_timestamp_utc', 'availability']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Maintenance windows CSV missing required columns: {missing}. "
            f"File: {maint_path}. Expected columns: {required_cols}"
        )
    
    # Strip whitespace from unit_id
    df['unit_id'] = df['unit_id'].astype(str).str.strip()
    
    # Parse timestamps with error handling
    n_before = len(df)
    try:
        df['start_timestamp_utc'] = parse_any_timestamp(df['start_timestamp_utc'])
        df['end_timestamp_utc'] = parse_any_timestamp(df['end_timestamp_utc'])
    except Exception as e:
        # Try to identify bad rows
        bad_rows = df[df['start_timestamp_utc'].isna() | df['end_timestamp_utc'].isna()]
        if len(bad_rows) > 0:
            print(f"[WARN] Dropping {len(bad_rows)} rows with unparseable timestamps")
            df = df.dropna(subset=['start_timestamp_utc', 'end_timestamp_utc'])
        else:
            raise ValueError(f"Failed to parse timestamps in maintenance CSV {maint_path}: {e}")
    
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
        print(f"[WARN] Dropping {len(invalid)} rows with start >= end")
        df = df[df['start_timestamp_utc'] < df['end_timestamp_utc']]
    
    # Coerce availability to numeric and clip to [0,1]
    df['availability'] = pd.to_numeric(df['availability'], errors='coerce')
    invalid_avail = df[df['availability'].isna()]
    if len(invalid_avail) > 0:
        print(f"[WARN] Dropping {len(invalid_avail)} rows with non-numeric availability")
        df = df.dropna(subset=['availability'])
    
    # Clip availability to [0,1] with warning
    out_of_range = df[(df['availability'] < 0) | (df['availability'] > 1)]
    if len(out_of_range) > 0:
        print(f"[WARN] Clipping {len(out_of_range)} availability values to [0,1]")
        df['availability'] = df['availability'].clip(0.0, 1.0)
    
    # Sort by unit_id, start
    df = df.sort_values(['unit_id', 'start_timestamp_utc']).reset_index(drop=True)
    
    n_after = len(df)
    if n_after < n_before:
        print(f"[WARN] Dropped {n_before - n_after} invalid rows from maintenance windows")
    
    return df


def build_availability_matrix(
    timestamps: pd.DatetimeIndex,
    unit_ids: List[str],
    maint_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build hourly availability matrix from maintenance windows.
    
    Args:
        timestamps: DatetimeIndex of hourly timestamps (must be UTC)
        unit_ids: List of unit IDs from utilities (for validation)
        maint_df: DataFrame from load_maintenance_windows() with columns:
                  unit_id, start_timestamp_utc, end_timestamp_utc, availability
                  
    Returns:
        DataFrame with:
        - index = timestamps (hourly)
        - columns = unit_ids (one column per unit)
        - values = availability (float in [0,1], default 1.0)
        
    Notes:
        - If maintenance windows overlap for a unit, availability is the minimum (most restrictive)
        - Ignores maintenance rows for unit_ids not present in unit_ids; warns once
    """
    # Ensure timestamps are UTC DatetimeIndex
    if isinstance(timestamps, pd.DatetimeIndex):
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize('UTC')
        else:
            timestamps = timestamps.tz_convert('UTC')
    else:
        timestamps = pd.to_datetime(timestamps, utc=True)
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize('UTC')
        else:
            timestamps = timestamps.tz_convert('UTC')
        timestamps = pd.DatetimeIndex(timestamps)
    
    # Create DataFrame with timestamps as index, unit_ids as columns
    availability_df = pd.DataFrame(
        index=timestamps,
        columns=unit_ids,
        data=1.0  # Default: fully available
    )
    
    if len(maint_df) == 0:
        # No maintenance windows
        return availability_df
    
    # Check for unmatched unit_ids
    unmatched_units = maint_df[~maint_df['unit_id'].isin(unit_ids)]['unit_id'].unique()
    if len(unmatched_units) > 0:
        print(f"[WARN] Maintenance windows reference unknown unit_id(s): {list(unmatched_units)}. "
              f"Available unit_ids: {unit_ids}. Ignoring these rows.")
        maint_df = maint_df[maint_df['unit_id'].isin(unit_ids)]
    
    # Apply maintenance windows: for each window, set availability to minimum of current and window value
    # This handles overlapping windows correctly (most restrictive wins)
    for _, window in maint_df.iterrows():
        unit_id = window['unit_id']
        start = window['start_timestamp_utc']
        end = window['end_timestamp_utc']
        avail = float(window['availability'])
        
        # Ensure start and end are timezone-aware UTC Timestamps
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
        availability_df.loc[mask, unit_id] = np.minimum(availability_df.loc[mask, unit_id], avail)
    
    return availability_df





