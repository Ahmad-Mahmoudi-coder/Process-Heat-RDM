"""
SignalsPack-lite Loader

Loads price and emissions signals from TOML configuration for different epochs.
Also provides helpers for loading and validating timestamp alignment with demand data.
"""

# Bootstrap: allow `python .\src\script.py` (adds repo root to sys.path)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, Any, Optional
import pandas as pd

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Need tomllib (Python 3.11+) or tomli package")

from src.path_utils import repo_root, resolve_path, input_root
from src.time_utils import parse_any_timestamp, build_hourly_utc_index, validate_time_alignment


def map_eval_epoch_to_signals_epoch(eval_epoch: int, signals_config: Dict[str, Any] = None, 
                                     input_dir: Path = None) -> int:
    """
    Map an evaluation epoch to the appropriate signals epoch.
    
    First tries to read Input/signals/epochs_registry.csv with columns eval_epoch, signals_epoch.
    If the registry doesn't exist or eval_epoch isn't found, falls back to finding the closest
    earlier available signals epoch from the signals config.
    
    Args:
        eval_epoch: The evaluation epoch (e.g., 2025, 2028)
        signals_config: Optional parsed signals config dict (if None, will load it)
        input_dir: Optional input directory path (default: input_root())
        
    Returns:
        The signals epoch to use (as int)
        
    Raises:
        ValueError: If no signals epoch can be found
    """
    if input_dir is None:
        input_dir = input_root()
    
    # Try to read epochs_registry.csv
    registry_path = input_dir / 'signals' / 'epochs_registry.csv'
    if registry_path.exists():
        try:
            registry_df = pd.read_csv(registry_path)
            # Check if it has the expected columns
            if 'eval_epoch' in registry_df.columns and 'signals_epoch' in registry_df.columns:
                # Convert to int for comparison
                registry_df['eval_epoch'] = registry_df['eval_epoch'].astype(int)
                registry_df['signals_epoch'] = registry_df['signals_epoch'].astype(int)
                
                # Look up exact match
                matches = registry_df[registry_df['eval_epoch'] == eval_epoch]
                if len(matches) > 0:
                    signals_epoch = int(matches.iloc[0]['signals_epoch'])
                    if signals_epoch != eval_epoch:
                        print(f"[WARN] Epoch mapping: eval_epoch {eval_epoch} -> signals_epoch {signals_epoch} (from registry)")
                    return signals_epoch
        except Exception:
            # If reading registry fails (wrong format, missing columns, etc.), fall through to fallback logic
            pass
    
    # Fallback: find closest earlier available signals epoch from config
    if signals_config is None:
        signals_config = load_signals_config()
    
    if 'epochs' not in signals_config:
        raise ValueError("Signals config missing 'epochs' table")
    
    # Get available signals epochs (as integers)
    available_epochs = [int(k) for k in signals_config['epochs'].keys() if k.isdigit()]
    if not available_epochs:
        raise ValueError("No numeric epoch keys found in signals config")
    
    available_epochs.sort(reverse=True)  # Sort descending
    
    # Find closest earlier epoch
    for signals_epoch in available_epochs:
        if signals_epoch <= eval_epoch:
            if signals_epoch != eval_epoch:
                print(f"[WARN] Epoch mapping: eval_epoch {eval_epoch} -> signals_epoch {signals_epoch} (closest earlier available)")
            return signals_epoch
    
    # No earlier epoch found
    raise ValueError(
        f"No signals epoch found for eval_epoch {eval_epoch}. "
        f"Available signals epochs: {sorted(available_epochs)}"
    )


def load_signals_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load and parse the SignalsPack-lite TOML file.
    
    Args:
        config_path: Path to signals config TOML file (default: Input/signals/signals_config.toml)
        
    Returns:
        The full parsed dict (top-level 'signals' table)
    """
    # Use default if not provided
    if config_path is None:
        config_path = str(input_root() / 'signals' / 'signals_config.toml')
    
    # Resolve config path robustly
    config_path_resolved = resolve_path(config_path)
    if not config_path_resolved.exists():
        raise FileNotFoundError(f"Signals config file not found: {config_path_resolved} (resolved from {config_path})")
    
    with open(config_path_resolved, 'rb') as f:
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
    # Resolve path robustly
    gxp_path_resolved = resolve_path(gxp_csv_path)
    if not gxp_path_resolved.exists():
        raise FileNotFoundError(f"GXP CSV file not found: {gxp_path_resolved} (resolved from {gxp_csv_path})")
    
    df = pd.read_csv(gxp_path_resolved)
    
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

