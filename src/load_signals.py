"""
SignalsPack-lite Loader

Loads price and emissions signals from TOML configuration for different epochs.
Also provides helpers for loading and validating timestamp alignment with demand data.
"""

from typing import Dict, Any, Optional
import sys
import pandas as pd

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Need tomllib (Python 3.11+) or tomli package")

from src.time_utils import parse_any_timestamp, build_hourly_utc_index, validate_time_alignment


def load_signals_config(config_path: str = "Input/signals_config.toml") -> Dict[str, Any]:
    """
    Load and parse the SignalsPack-lite TOML file.
    
    Args:
        config_path: Path to signals config TOML file
        
    Returns:
        The full parsed dict (top-level 'signals' table)
    """
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    # TOML with [signals.general] and [signals.epochs."2020"] creates nested structure
    if 'signals' not in config:
        raise ValueError(f"Config file {config_path} missing top-level 'signals' table")
    
    return config['signals']


def get_signals_for_epoch(config: Dict[str, Any], epoch_label: str) -> Dict[str, float]:
    """
    Given the parsed signals config and an epoch label (e.g. '2020'),
    return a flat dict with numeric price and emissions parameters for that epoch.
    
    Expected keys in the returned dict:
      - coal_price_nzd_per_MWh_fuel
      - coal_ef_tCO2_per_MWh_fuel
      - biomass_price_nzd_per_MWh_fuel
      - biomass_ef_tCO2_per_MWh_fuel
      - elec_price_flat_nzd_per_MWh
      - ets_price_nzd_per_tCO2
    
    Args:
        config: Parsed signals config dict (from load_signals_config)
        epoch_label: Epoch label string (e.g. '2020', '2024')
        
    Returns:
        Dict with numeric signal values for the epoch
        
    Raises:
        ValueError: If epoch label or any required key is missing
    """
    if 'epochs' not in config:
        raise ValueError("Config missing 'epochs' table")
    
    if epoch_label not in config['epochs']:
        available = list(config['epochs'].keys())
        raise ValueError(f"Epoch '{epoch_label}' not found in config. Available epochs: {available}")
    
    epoch_data = config['epochs'][epoch_label]
    
    # Required keys
    required_keys = [
        'coal_price_nzd_per_MWh_fuel',
        'coal_ef_tCO2_per_MWh_fuel',
        'biomass_price_nzd_per_MWh_fuel',
        'biomass_ef_tCO2_per_MWh_fuel',
        'elec_price_flat_nzd_per_MWh',
        'ets_price_nzd_per_tCO2',
    ]
    
    # Validate all required keys exist
    missing_keys = [key for key in required_keys if key not in epoch_data]
    if missing_keys:
        raise ValueError(
            f"Epoch '{epoch_label}' missing required keys: {missing_keys}. "
            f"Found keys: {list(epoch_data.keys())}"
        )
    
    # Extract and return numeric values
    result = {}
    for key in required_keys:
        value = epoch_data[key]
        if not isinstance(value, (int, float)):
            raise ValueError(f"Epoch '{epoch_label}', key '{key}': expected numeric value, got {type(value)}")
        result[key] = float(value)
    
    return result


def load_gxp_data(gxp_csv_path: str) -> pd.DataFrame:
    """
    Load GXP data CSV and parse timestamp_utc as UTC.
    
    Args:
        gxp_csv_path: Path to GXP CSV file with timestamp_utc column
        
    Returns:
        DataFrame with timestamp_utc parsed as UTC datetime64[ns, UTC]
    """
    df = pd.read_csv(gxp_csv_path)
    
    if 'timestamp_utc' not in df.columns:
        raise ValueError(f"GXP CSV file {gxp_csv_path} must have 'timestamp_utc' column")
    
    # Parse timestamp_utc as UTC
    df['timestamp_utc'] = parse_any_timestamp(df['timestamp_utc'])
    
    return df


def validate_demand_gxp_alignment(demand_df: pd.DataFrame, gxp_df: pd.DataFrame, 
                                   year: Optional[int] = None) -> None:
    """
    Validate that demand and GXP dataframes have aligned timestamps.
    
    Optionally validates that timestamps match the expected hourly UTC index for a given year.
    
    Args:
        demand_df: DataFrame with timestamp_utc column
        gxp_df: DataFrame with timestamp_utc column
        year: Optional year to validate against build_hourly_utc_index(year)
        
    Raises:
        ValueError: If alignment checks fail
    """
    # Use the general validation function
    validate_time_alignment(demand_df, gxp_df)
    
    # If year is provided, also validate against expected index
    if year is not None:
        expected_index = build_hourly_utc_index(year)
        
        # Parse timestamps if needed
        demand_ts = parse_any_timestamp(demand_df['timestamp_utc'])
        gxp_ts = parse_any_timestamp(gxp_df['timestamp_utc'])
        
        # Check against expected index
        if not demand_ts.equals(pd.Series(expected_index)):
            raise ValueError(
                f"demand_df timestamps do not match expected hourly UTC index for year {year}"
            )
        
        if not gxp_ts.equals(pd.Series(expected_index)):
            raise ValueError(
                f"gxp_df timestamps do not match expected hourly UTC index for year {year}"
            )

