"""
Timestamp utilities for standardizing timestamps across site outputs.

Provides a single timestamp contract:
- Column name: timestamp_utc
- Format: ISO-8601 with Z, e.g. 2020-01-01T00:00:00Z
- Internal type: pandas datetime64[ns, UTC]
"""

import pandas as pd
import numpy as np
from typing import Union
import pytz


def build_hourly_utc_index(year: int) -> pd.DatetimeIndex:
    """
    Build hourly DatetimeIndex for the full year in UTC.
    
    Creates periods = hours in that year, starting at YYYY-01-01 00:00 UTC.
    Handles leap years automatically (8760 or 8784 hours).
    
    Args:
        year: Year to generate index for
        
    Returns:
        DatetimeIndex with timezone UTC, dtype datetime64[ns, UTC]
    """
    start = pd.Timestamp(f'{year}-01-01 00:00:00', tz='UTC')
    
    # Calculate number of hours in the year
    # Check if it's a leap year
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    hours_per_year = 8784 if is_leap else 8760
    
    index = pd.date_range(start=start, periods=hours_per_year, freq='h', tz='UTC')
    
    return index


def to_iso_z(dt_series: Union[pd.Series, pd.DatetimeIndex]) -> pd.Series:
    """
    Convert datetime series to ISO-8601 format with Z suffix.
    
    Format: %Y-%m-%dT%H:%M:%SZ
    Example: 2020-01-01T00:00:00Z
    
    Args:
        dt_series: Series or DatetimeIndex with datetime values
        
    Returns:
        Series[str] with ISO Z formatted strings
    """
    if isinstance(dt_series, pd.DatetimeIndex):
        dt_series = pd.Series(dt_series)
    
    # Ensure timezone-aware (convert to UTC if naive)
    if dt_series.dt.tz is None:
        # Assume naive timestamps are in UTC
        dt_series = dt_series.dt.tz_localize('UTC')
    else:
        # Convert to UTC if in different timezone
        dt_series = dt_series.dt.tz_convert('UTC')
    
    # Format as ISO Z string
    result = dt_series.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    return result


def parse_any_timestamp(series: Union[pd.Series, list]) -> pd.Series:
    """
    Parse timestamp series accepting various formats and convert to UTC.
    
    Accepts:
    - ISO-8601 with Z (e.g. "2020-01-01T00:00:00Z")
    - Excel-like formats (e.g. "1/01/2020 1:00", "01/01/2020 01:00")
    - Other pandas-parseable datetime strings
    
    For Excel-like formats, treats as NZ-local time (UTC+12 or UTC+13 depending on DST)
    or naive (assumes UTC if no timezone info).
    
    Args:
        series: Series or list of timestamp strings or datetime objects
        
    Returns:
        Series with dtype datetime64[ns, UTC]
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    # If already datetime, ensure UTC
    if pd.api.types.is_datetime64_any_dtype(series):
        if series.dt.tz is None:
            # Naive datetime: assume UTC
            result = series.dt.tz_localize('UTC')
        else:
            # Timezone-aware: convert to UTC
            result = series.dt.tz_convert('UTC')
        return result
    
    # Convert to string for parsing
    series_str = series.astype(str)
    
    # Try parsing with pandas (handles ISO formats well)
    try:
        parsed = pd.to_datetime(series_str, utc=True, errors='coerce')
    except Exception:
        parsed = pd.to_datetime(series_str, errors='coerce')
    
    # Check if we have any unparseable values
    if parsed.isna().any():
        # Try Excel-like formats (NZ-local interpretation)
        nz_tz = pytz.timezone('Pacific/Auckland')
        
        # Try parsing as NZ-local
        for idx in series_str[parsed.isna()].index:
            val = series_str.loc[idx]
            try:
                # Try parsing as naive first
                naive_dt = pd.to_datetime(val, errors='coerce')
                if pd.notna(naive_dt):
                    # Localize to NZ timezone
                    if isinstance(naive_dt, pd.Timestamp):
                        # Single value
                        nz_dt = nz_tz.localize(naive_dt.to_pydatetime().replace(tzinfo=None))
                        parsed.loc[idx] = pd.Timestamp(nz_dt).tz_convert('UTC')
                    else:
                        # Series
                        nz_dt = nz_tz.localize(naive_dt.iloc[0].to_pydatetime().replace(tzinfo=None))
                        parsed.loc[idx] = pd.Timestamp(nz_dt).tz_convert('UTC')
            except Exception:
                # If NZ-local parsing fails, try as UTC naive
                try:
                    naive_dt = pd.to_datetime(val, errors='coerce')
                    if pd.notna(naive_dt):
                        parsed.loc[idx] = pd.Timestamp(naive_dt).tz_localize('UTC')
                except Exception:
                    pass  # Leave as NaT
    
    # Ensure all are UTC
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize('UTC')
    else:
        parsed = parsed.dt.tz_convert('UTC')
    
    return parsed


def validate_time_alignment(demand_df: pd.DataFrame, gxp_df: pd.DataFrame) -> None:
    """
    Validate that demand and GXP dataframes have aligned timestamps.
    
    Checks:
    - Same number of rows
    - Same first timestamp
    - Same last timestamp
    - All timestamps match exactly
    
    Args:
        demand_df: DataFrame with timestamp_utc column
        gxp_df: DataFrame with timestamp_utc column
        
    Raises:
        ValueError: If any alignment check fails
    """
    # Check timestamp column exists
    if 'timestamp_utc' not in demand_df.columns:
        raise ValueError("demand_df missing 'timestamp_utc' column")
    if 'timestamp_utc' not in gxp_df.columns:
        raise ValueError("gxp_df missing 'timestamp_utc' column")
    
    # Parse timestamps if needed
    demand_ts = parse_any_timestamp(demand_df['timestamp_utc'])
    gxp_ts = parse_any_timestamp(gxp_df['timestamp_utc'])
    
    # Check row count
    if len(demand_df) != len(gxp_df):
        raise ValueError(
            f"Row count mismatch: demand_df has {len(demand_df)} rows, "
            f"gxp_df has {len(gxp_df)} rows"
        )
    
    # Check first timestamp
    if demand_ts.iloc[0] != gxp_ts.iloc[0]:
        raise ValueError(
            f"First timestamp mismatch: demand_df={demand_ts.iloc[0]}, "
            f"gxp_df={gxp_ts.iloc[0]}"
        )
    
    # Check last timestamp
    if demand_ts.iloc[-1] != gxp_ts.iloc[-1]:
        raise ValueError(
            f"Last timestamp mismatch: demand_df={demand_ts.iloc[-1]}, "
            f"gxp_df={gxp_ts.iloc[-1]}"
        )
    
    # Check all timestamps match
    if not demand_ts.equals(gxp_ts):
        # Find first mismatch
        mismatches = demand_ts != gxp_ts
        if mismatches.any():
            first_mismatch_idx = mismatches.idxmax() if isinstance(mismatches, pd.Series) else next(i for i, m in enumerate(mismatches) if m)
            raise ValueError(
                f"Timestamp mismatch at index {first_mismatch_idx}: "
                f"demand_df={demand_ts.iloc[first_mismatch_idx]}, "
                f"gxp_df={gxp_ts.iloc[first_mismatch_idx]}"
            )




