"""
DemandPack Generator for Edendale Site (11_031)

Generates synthetic hourly heat demand profiles for 2020 anchored to annual energy
targets with realistic seasonal, weekday, and hourly patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import sys

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Need tomllib (Python 3.11+) or tomli package")

from src.time_utils import build_hourly_utc_index, to_iso_z, parse_any_timestamp


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse TOML configuration file."""
    with open(config_path, 'rb') as f:
        return tomllib.load(f)


def load_annual_anchor(config: Dict[str, Any]) -> float:
    """
    Load annual heat target from site file.
    Returns annual_heat_GWh for the target site and year.
    """
    site_file = config['general']['site_file']
    target_site_id = config['general']['target_site_id']
    target_year = config['general']['target_year']
    
    site_id_col = config['columns']['site_id']
    year_col = config['columns']['year']
    annual_heat_col = config['columns']['annual_heat_gwh']
    
    df = pd.read_csv(site_file)
    
    # Filter for target site and year
    mask = (df[site_id_col] == target_site_id) & (df[year_col] == target_year)
    if not mask.any():
        raise ValueError(f"No data found for site_id={target_site_id}, year={target_year}")
    
    annual_heat_GWh = df.loc[mask, annual_heat_col].iloc[0]
    return float(annual_heat_GWh)


# build_hourly_index is now replaced by build_hourly_utc_index from time_utils


def load_seasonal_factors(config: Dict[str, Any], hourly_index: pd.DatetimeIndex) -> pd.Series:
    """
    Load seasonal factors and interpolate to hourly resolution.
    
    Treats each month as an anchor at mid-month (anchor_day), interpolates
    to daily values, applies smoothing, then maps to hourly.
    """
    seasonality = config['seasonality']
    df = pd.read_csv(seasonality['file'])
    
    month_col = seasonality['month_col']
    factor_col = seasonality['factor_col']
    anchor_day = seasonality['anchor_day']
    smooth_days = seasonality['smooth_days']
    wrap_year = seasonality['wrap_year']
    
    year = hourly_index[0].year
    
    # Create anchor points: one per month at mid-month
    anchors = []
    for _, row in df.iterrows():
        month = int(row[month_col])
        factor = float(row[factor_col])
        anchor_date = pd.Timestamp(f'{year}-{month:02d}-{anchor_day:02d}')
        anchors.append({'date': anchor_date, 'factor': factor})
    
    # If wrapping, add Dec→Jan continuity
    if wrap_year:
        # Add a point at end of year (Dec 31) with Dec's factor
        dec_factor = df[df[month_col] == 12][factor_col].iloc[0]
        anchors.append({'date': pd.Timestamp(f'{year}-12-31'), 'factor': float(dec_factor)})
        # Add a point at start of next year (Jan 1) with Jan's factor
        jan_factor = df[df[month_col] == 1][factor_col].iloc[0]
        anchors.append({'date': pd.Timestamp(f'{year+1}-01-01'), 'factor': float(jan_factor)})
    
    anchors_df = pd.DataFrame(anchors).sort_values('date')
    
    # Create daily index for the year
    daily_index = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    
    # Interpolate anchors to daily values using time-based interpolation
    daily_factors = np.interp(
        daily_index.astype(np.int64),
        anchors_df['date'].astype(np.int64),
        anchors_df['factor']
    )
    daily_series = pd.Series(daily_factors, index=daily_index)
    
    # Apply rolling smoothing
    if smooth_days > 0:
        # Use centered rolling window
        daily_series = daily_series.rolling(window=smooth_days, center=True, min_periods=1).mean()
    
    # Map daily factors to hourly (repeat each day's factor for all 24 hours)
    # Convert hourly_index to dates and map to daily factors
    hourly_dates = pd.to_datetime(hourly_index.date)
    hourly_factors = daily_series.reindex(hourly_dates, method='ffill').values
    return pd.Series(hourly_factors, index=hourly_index, name='seasonal_factor')


def load_weekday_factors(config: Dict[str, Any], hourly_index: pd.DatetimeIndex) -> pd.Series:
    """Load weekday factors and map to hourly timestamps."""
    weekday = config['weekday']
    df = pd.read_csv(weekday['file'])
    
    day_col = weekday['day_col']
    factor_col = weekday['factor_col']
    
    # Create mapping from day name to factor
    day_to_factor = dict(zip(df[day_col], df[factor_col]))
    
    # Map each timestamp's day name to its factor
    day_names = hourly_index.day_name()
    factors = [day_to_factor.get(day, 1.0) for day in day_names]
    
    return pd.Series(factors, index=hourly_index, name='weekday_factor')


def load_hourly_factors(config: Dict[str, Any], hourly_index: pd.DatetimeIndex) -> pd.Series:
    """Load hour-of-day factors and map to hourly timestamps."""
    daily = config['daily']
    df = pd.read_csv(daily['file'])
    
    hour_col = daily['hour_col']
    factor_col = daily['factor_col']
    
    # Create mapping from hour (0-23) to factor
    hour_to_factor = dict(zip(df[hour_col], df[factor_col]))
    
    # Map each timestamp's hour to its factor
    hours = hourly_index.hour
    factors = [hour_to_factor.get(hour, 1.0) for hour in hours]
    
    return pd.Series(factors, index=hourly_index, name='hourly_factor')


def generate_weekly_drift(config: Dict[str, Any], hourly_index: pd.DatetimeIndex) -> pd.Series:
    """
    Generate weekly drift factors using AR(1) process in log-space.
    
    Creates one factor per ISO week, then maps to hourly timestamps.
    """
    weekly_drift = config['weekly_drift']
    
    if not weekly_drift['enabled']:
        return pd.Series(1.0, index=hourly_index, name='weekly_factor')
    
    sigma = weekly_drift['sigma']
    rho = weekly_drift['rho']
    min_factor = weekly_drift['min']
    max_factor = weekly_drift['max']
    
    # Get ISO week numbers for all hours
    week_numbers = hourly_index.isocalendar().week
    unique_weeks = sorted(week_numbers.unique())
    n_weeks = len(unique_weeks)
    
    # Generate AR(1) process in log-space
    np.random.seed(config['noise']['seed'])
    log_r = np.zeros(n_weeks)
    log_r[0] = 0.0  # Initial factor = 1.0
    
    for w in range(1, n_weeks):
        eps = np.random.normal(0, sigma)
        log_r[w] = rho * log_r[w-1] + eps
    
    # Convert to factors and clamp
    r = np.exp(log_r)
    r = np.clip(r, min_factor, max_factor)
    
    # Create mapping from week number to factor
    week_to_factor = dict(zip(unique_weeks, r))
    
    # Map each hour to its week's factor
    factors = [week_to_factor.get(week, 1.0) for week in week_numbers]
    
    return pd.Series(factors, index=hourly_index, name='weekly_factor')


def generate_hour_noise(config: Dict[str, Any], hourly_index: pd.DatetimeIndex) -> pd.Series:
    """Generate hour-to-hour noise factors."""
    noise = config['noise']
    
    if not noise['enabled']:
        return pd.Series(1.0, index=hourly_index, name='noise_factor')
    
    hour_sigma = noise['hour_sigma']
    seed = noise['seed']
    
    np.random.seed(seed)
    eps = np.random.normal(0, hour_sigma, len(hourly_index))
    factors = 1.0 + eps
    
    # Clamp to reasonable bounds
    factors = np.clip(factors, 0.95, 1.05)
    
    return pd.Series(factors, index=hourly_index, name='noise_factor')


def load_and_resample_temperature(
    config: Dict[str, Any], 
    hourly_index: pd.DatetimeIndex,
    target_site_id: str
) -> Optional[pd.Series]:
    """
    Load temperature series and resample to hourly UTC index.
    
    Handles various input frequencies (half-hourly, hourly, etc.) and resamples
    to match the DemandPack hourly index. If temperature is already hourly,
    it passes through unchanged after reindexing.
    
    Args:
        config: Configuration dict (may contain temperature settings)
        hourly_index: Target hourly UTC DatetimeIndex
        target_site_id: Site ID to filter temperature data
        
    Returns:
        Series with temperature values aligned to hourly_index, or None if not configured
    """
    # Check if temperature is configured (optional feature)
    if 'temperature' not in config:
        return None
    
    temp_config = config['temperature']
    if not temp_config.get('enabled', False):
        return None
    
    temp_file = temp_config.get('file')
    if not temp_file or not Path(temp_file).exists():
        print(f"[SKIP] Temperature file not found: {temp_file}")
        return None
    
    # Load temperature CSV
    print(f"Loading temperature series from {temp_file}...")
    temp_df = pd.read_csv(temp_file)
    
    # Detect timestamp column (support various names)
    timestamp_col = None
    for col in ['timestamp_utc', 'timestamp', 'datetime', 'date']:
        if col in temp_df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        raise ValueError(f"Temperature CSV {temp_file} must have a timestamp column (timestamp_utc, timestamp, datetime, or date)")
    
    # Detect temperature value column
    temp_col = None
    for col in ['temp_C', 'temperature', 'temp', 'temperature_C']:
        if col in temp_df.columns:
            temp_col = col
            break
    
    if temp_col is None:
        raise ValueError(f"Temperature CSV {temp_file} must have a temperature column (temp_C, temperature, temp, or temperature_C)")
    
    # Filter by site_id if present
    if 'site_id' in temp_df.columns:
        temp_df = temp_df[temp_df['site_id'] == target_site_id].copy()
        if len(temp_df) == 0:
            raise ValueError(f"No temperature data found for site_id={target_site_id} in {temp_file}")
    
    # Parse timestamps to UTC
    print(f"  Parsing timestamps from column '{timestamp_col}'...")
    ts_dt_utc = parse_any_timestamp(temp_df[timestamp_col])
    
    # Log detected frequency
    original_rowcount = len(temp_df)
    print(f"  Original temperature data: {original_rowcount} rows")
    
    # Detect frequency by checking time differences
    if len(ts_dt_utc) > 1:
        time_diffs = ts_dt_utc.diff().dropna()
        median_diff = time_diffs.median()
        if median_diff <= pd.Timedelta('30min'):
            detected_freq = '30min (half-hourly)'
        elif median_diff <= pd.Timedelta('1h'):
            detected_freq = '1h (hourly)'
        else:
            detected_freq = f'{median_diff} (irregular)'
        print(f"  Detected frequency: {detected_freq}")
    
    # Set timestamp as index and resample to hourly
    temp_df_indexed = temp_df.set_index(ts_dt_utc)
    
    # Resample to hourly (mean aggregation)
    print(f"  Resampling to hourly (mean aggregation)...")
    temp_hourly = temp_df_indexed.resample('H', label='left', closed='left').mean(numeric_only=True)
    
    resampled_rowcount = len(temp_hourly)
    print(f"  After resampling: {resampled_rowcount} rows")
    
    if resampled_rowcount != original_rowcount:
        print(f"  [INFO] Resampling occurred: {original_rowcount} -> {resampled_rowcount} rows")
    
    # Reindex to the DemandPack hourly index
    print(f"  Reindexing to DemandPack hourly index ({len(hourly_index)} rows)...")
    temp_hourly_reindexed = temp_hourly[temp_col].reindex(hourly_index)
    
    # Handle missing values
    missing_count = temp_hourly_reindexed.isna().sum()
    if missing_count > 0:
        print(f"  [WARNING] {missing_count} missing values after reindexing, filling with interpolation...")
        # Use time interpolation, fallback to forward fill
        temp_hourly_reindexed = temp_hourly_reindexed.interpolate(method='time', limit_direction='both')
        # Fill any remaining NaNs at the edges with forward/backward fill
        temp_hourly_reindexed = temp_hourly_reindexed.ffill().bfill()
        
        if temp_hourly_reindexed.isna().any():
            raise ValueError(f"Could not fill all missing temperature values. {temp_hourly_reindexed.isna().sum()} NaNs remain.")
    
    print(f"  [OK] Temperature series aligned to hourly index: {len(temp_hourly_reindexed)} rows")
    
    return pd.Series(temp_hourly_reindexed.values, index=hourly_index, name='temperature_C')


def apply_peak_cap(series: pd.Series, cap_mw: float, method: str, target_energy_GWh: float) -> pd.Series:
    """
    Apply peak cap while preserving total annual energy exactly.
    
    Args:
        series: Hourly MW series
        cap_mw: Peak capacity limit
        method: Method to use ('clip_redistribute')
        target_energy_GWh: Target annual energy (for verification)
        
    Returns:
        Capped series with same total energy
    """
    if method != 'clip_redistribute':
        raise ValueError(f"Unknown cap_method: {method}")
    
    result = series.copy()
    max_iterations = 100
    tolerance_MWh = 0.1  # 0.1 MWh tolerance
    
    for iteration in range(max_iterations):
        # Clip to cap
        clipped = result.clip(upper=cap_mw)
        deficit_MWh = (result - clipped).sum()  # Energy that was clipped
        
        if deficit_MWh < tolerance_MWh:
            break
        
        # Find hours with headroom (below cap)
        headroom = cap_mw - clipped
        headroom_mask = headroom > 1e-6
        
        if not headroom_mask.any():
            raise ValueError(f"Cannot redistribute {deficit_MWh:.2f} MWh: no headroom available after capping to {cap_mw} MW")
        
        # Redistribute deficit proportionally to headroom
        total_headroom = headroom[headroom_mask].sum()
        if total_headroom < 1e-6:
            raise ValueError(f"Cannot redistribute {deficit_MWh:.2f} MWh: insufficient headroom")
        
        redistribution = (headroom[headroom_mask] / total_headroom) * deficit_MWh
        result[headroom_mask] = clipped[headroom_mask] + redistribution
        result[~headroom_mask] = clipped[~headroom_mask]
    
    # Final energy closure check
    final_energy_GWh = result.sum() / 1000.0
    energy_error = abs(final_energy_GWh - target_energy_GWh)
    if energy_error > 0.001:  # 1 MWh tolerance
        raise ValueError(f"Peak cap energy closure failed: {final_energy_GWh:.6f} GWh != {target_energy_GWh:.6f} GWh (error: {energy_error:.6f} GWh)")
    
    # Verify peak cap
    max_value = result.max()
    if max_value > cap_mw + 1e-6:
        raise ValueError(f"Peak cap violation: max={max_value:.2f} MW > cap={cap_mw} MW")
    
    print(f"[OK] Peak cap applied: max={max_value:.2f} MW <= {cap_mw} MW, energy preserved ({final_energy_GWh:.6f} GWh)")
    
    return result


def generate_demandpack(config_path: str, cap_peak_mw_override: float = None) -> pd.DataFrame:
    """
    Main function to generate the DemandPack hourly heat demand profile.
    
    Returns DataFrame with timestamp, site_id, heat_demand_MW, and all factors.
    """
    config = load_config(config_path)
    
    # Load annual anchor
    annual_heat_GWh = load_annual_anchor(config)
    target_year = config['general']['target_year']
    target_site_id = config['general']['target_site_id']
    
    # Build hourly index in UTC (handles leap years automatically)
    hourly_index = build_hourly_utc_index(target_year)
    hours_per_year = len(hourly_index)
    
    # Compute average MW
    avg_MW = annual_heat_GWh * 1000.0 / hours_per_year
    
    # Load and compute all factors
    seasonal = load_seasonal_factors(config, hourly_index)
    weekday = load_weekday_factors(config, hourly_index)
    hourly = load_hourly_factors(config, hourly_index)
    weekly = generate_weekly_drift(config, hourly_index)
    noise = generate_hour_noise(config, hourly_index)
    
    # Load and resample temperature if configured
    temperature = load_and_resample_temperature(config, hourly_index, target_site_id)
    
    # Combine factors
    combined_factors = seasonal * weekday * hourly * weekly * noise
    
    # Compute raw MW
    raw_MW = avg_MW * combined_factors
    
    # Energy closure: scale to exact annual target
    raw_GWh = raw_MW.sum() / 1000.0
    scale = annual_heat_GWh / raw_GWh
    final_MW = raw_MW * scale
    
    # Apply peak cap if specified (with energy preservation)
    # CLI override takes precedence over config
    cap_peak_mw = cap_peak_mw_override if cap_peak_mw_override is not None else config['general'].get('cap_peak_mw', None)
    cap_method = config['general'].get('cap_method', 'clip_redistribute')
    
    if cap_peak_mw is not None and cap_peak_mw > 0:
        final_MW = apply_peak_cap(final_MW, cap_peak_mw, cap_method, annual_heat_GWh)
    
    # Build output DataFrame with timestamp_utc in ISO Z format
    # Ensure all columns have the same length as hourly_index
    expected_len = len(hourly_index)
    
    result_dict = {
        'timestamp_utc': to_iso_z(hourly_index),
        'site_id': [target_site_id] * expected_len,  # Broadcast site_id to match length
        'heat_demand_MW': final_MW.values if isinstance(final_MW, pd.Series) else final_MW,
        'seasonal_factor': seasonal.values,
        'weekday_factor': weekday.values,
        'hourly_factor': hourly.values,
        'weekly_factor': weekly.values,
        'noise_factor': noise.values,
    }
    
    # Add temperature if available
    if temperature is not None:
        result_dict['temperature_C'] = temperature.values
    
    # Defensive assertion: all columns must have the same length
    for col_name, col_data in result_dict.items():
        actual_len = len(col_data) if hasattr(col_data, '__len__') else 1
        assert actual_len == expected_len, (
            f"Column '{col_name}' length mismatch: expected {expected_len} (hourly index length), "
            f"got {actual_len}"
        )
    
    # Additional assertion if temperature is used
    if temperature is not None:
        assert len(hourly_index) == len(final_MW) == len(temperature), (
            f"Length mismatch: hourly_index={len(hourly_index)}, "
            f"final_MW={len(final_MW)}, temperature={len(temperature)}"
        )
    
    result = pd.DataFrame(result_dict)
    
    # Verify final DataFrame length
    assert len(result) == expected_len, (
        f"Output DataFrame length mismatch: expected {expected_len}, got {len(result)}"
    )
    
    # Verify energy closure
    final_GWh = result['heat_demand_MW'].sum() / 1000.0
    error = abs(final_GWh - annual_heat_GWh)
    if error > 0.001:  # 1 MWh tolerance
        print(f"WARNING: Energy closure error = {error:.6f} GWh")
    else:
        print(f"[OK] Energy closure verified: {final_GWh:.6f} GWh (target: {annual_heat_GWh} GWh)")
    
    return result


def main():
    """CLI entrypoint."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate DemandPack hourly heat demand')
    parser.add_argument('--config', default='Input/demandpack_config.toml',
                       help='Path to config TOML file')
    parser.add_argument('--cap-peak-mw', type=float, default=None,
                       help='Peak capacity cap (MW). If not specified, uses config value or no cap.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: Output/ or from config)')
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = Path('Output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate demandpack
    print("Generating DemandPack...")
    df = generate_demandpack(args.config, cap_peak_mw_override=args.cap_peak_mw)
    
    # Save CSV (timestamp_utc is already in ISO Z format as string)
    config = load_config(args.config)
    # Use output_dir if provided, otherwise use config value
    if args.output_dir:
        output_csv = output_path / Path(config['general']['output_csv']).name
    else:
        output_csv = config['general']['output_csv']
    df.to_csv(output_csv, index=False)
    print(f"[OK] Saved output to {output_csv}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Mean hourly load: {df['heat_demand_MW'].mean():.2f} MW")
    print(f"  Max hourly load: {df['heat_demand_MW'].max():.2f} MW")
    print(f"  95th percentile: {df['heat_demand_MW'].quantile(0.95):.2f} MW")
    print(f"  Total energy: {df['heat_demand_MW'].sum() / 1000.0:.6f} GWh")
    
    # Monthly totals (parse timestamp_utc for month extraction)
    from src.time_utils import parse_any_timestamp
    df['timestamp'] = parse_any_timestamp(df['timestamp_utc'])
    df['month'] = df['timestamp'].dt.month
    monthly = df.groupby('month')['heat_demand_MW'].sum() / 1000.0
    print("\nMonthly totals (GWh):")
    for month, gwh in monthly.items():
        print(f"  Month {month:2d}: {gwh:6.2f} GWh")
    
    # Sanity check for peak load
    peak_MW = df['heat_demand_MW'].max()
    if peak_MW > 200:  # Rough sanity check for 4-coal-boiler fleet
        print(f"\n[WARNING] Peak hourly load ({peak_MW:.2f} MW) may be implausibly high")
    else:
        print(f"\n[OK] Peak load ({peak_MW:.2f} MW) appears reasonable")


if __name__ == '__main__':
    main()

