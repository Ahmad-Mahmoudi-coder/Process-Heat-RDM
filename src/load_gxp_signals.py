"""
Loader for Edendale_GXP module outputs.

Provides functions to load GXP hourly data (tariff, headroom) and grid emissions intensity
for use in site dispatch coupling.
"""

# Bootstrap: allow `python .\src\script.py` (adds repo root to sys.path)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from typing import Tuple, Optional
from src.time_utils import parse_any_timestamp, to_iso_z
from src.path_utils import repo_root


def load_gxp_hourly(epoch: int, modules_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load GXP hourly data for an epoch.
    
    Reads from: modules/edendale_gxp/outputs_latest/gxp_hourly_<epoch>.csv
    
    Expected columns:
    - timestamp_utc
    - headroom_mw (optional)
    - tariff_nzd_per_mwh (required)
    
    Args:
        epoch: Epoch year (e.g., 2020, 2025, 2028, 2035)
        modules_dir: Optional modules directory (default: repo_root/modules)
        
    Returns:
        DataFrame with columns:
        - timestamp_utc (datetime64[ns, UTC])
        - elec_price_nzd_per_MWh (from tariff_nzd_per_mwh)
        - headroom_MW (from headroom_mw, if present)
        
    Raises:
        FileNotFoundError: If gxp_hourly file not found
        ValueError: If required columns missing or timestamp alignment fails
    """
    if modules_dir is None:
        modules_dir = repo_root() / 'modules'
    
    gxp_file = modules_dir / 'edendale_gxp' / 'outputs_latest' / f'gxp_hourly_{epoch}.csv'
    
    if not gxp_file.exists():
        raise FileNotFoundError(
            f"GXP hourly file not found: {gxp_file}\n"
            f"Expected location: modules/edendale_gxp/outputs_latest/gxp_hourly_{epoch}.csv"
        )
    
    df = pd.read_csv(gxp_file)
    
    # Validate required columns
    if 'timestamp_utc' not in df.columns:
        raise ValueError(f"gxp_hourly_{epoch}.csv missing required column: timestamp_utc")
    
    if 'tariff_nzd_per_mwh' not in df.columns:
        raise ValueError(
            f"gxp_hourly_{epoch}.csv missing required column: tariff_nzd_per_mwh. "
            f"Cannot couple ToU pricing without tariff data."
        )
    
    # Parse timestamps
    df['timestamp_utc'] = parse_any_timestamp(df['timestamp_utc'])
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Validate: by default assert exactly 1 unique gxp_id
    if 'gxp_id' in df.columns:
        unique_gxp_ids = df['gxp_id'].unique()
        if len(unique_gxp_ids) > 1:
            raise ValueError(
                f"gxp_hourly_{epoch}.csv contains {len(unique_gxp_ids)} unique GXP IDs: {list(unique_gxp_ids)}. "
                f"By default, exactly 1 unique gxp_id is required. "
                f"If multiple GXPs are intended, use allow_multi_gxp flag (not yet implemented)."
            )
        # If single GXP, proceed normally
        df_agg = df.groupby('timestamp_utc', as_index=False).agg({
            'tariff_nzd_per_mwh': 'first',  # Should be same, but use first for determinism
            'headroom_mw': 'first' if 'headroom_mw' in df.columns else None
        })
        if 'headroom_mw' not in df.columns:
            df_agg = df.groupby('timestamp_utc', as_index=False).agg({'tariff_nzd_per_mwh': 'first'})
    else:
        # No gxp_id column - assume single GXP, aggregate by timestamp
        agg_dict = {'tariff_nzd_per_mwh': 'first'}
        if 'headroom_mw' in df.columns:
            agg_dict['headroom_mw'] = 'first'
        df_agg = df.groupby('timestamp_utc', as_index=False).agg(agg_dict)
    
    # Build result DataFrame
    result = pd.DataFrame({
        'timestamp_utc': df_agg['timestamp_utc'],
        'elec_price_nzd_per_MWh': df_agg['tariff_nzd_per_mwh']
    })
    
    # Add headroom if present
    if 'headroom_mw' in df_agg.columns:
        result['headroom_MW'] = df_agg['headroom_mw']
    
    return result


def load_grid_emissions_intensity(epoch: int, modules_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load grid emissions intensity data for an epoch.
    
    Reads from: modules/edendale_gxp/outputs_latest/grid_emissions_intensity_<epoch>.csv
    
    Expected columns:
    - timestamp_utc
    - grid_co2e_kg_per_mwh_avg
    - grid_co2e_kg_per_mwh_marginal
    
    Args:
        epoch: Epoch year (e.g., 2020, 2025, 2028, 2035)
        modules_dir: Optional modules directory (default: repo_root/modules)
        
    Returns:
        DataFrame with columns:
        - timestamp_utc (datetime64[ns, UTC])
        - grid_co2e_kg_per_mwh_avg
        - grid_co2e_kg_per_mwh_marginal
        
    Raises:
        FileNotFoundError: If grid emissions file not found
        ValueError: If required columns missing
    """
    if modules_dir is None:
        modules_dir = repo_root() / 'modules'
    
    emissions_file = modules_dir / 'edendale_gxp' / 'outputs_latest' / f'grid_emissions_intensity_{epoch}.csv'
    
    if not emissions_file.exists():
        raise FileNotFoundError(
            f"Grid emissions intensity file not found: {emissions_file}\n"
            f"Expected location: modules/edendale_gxp/outputs_latest/grid_emissions_intensity_{epoch}.csv"
        )
    
    df = pd.read_csv(emissions_file)
    
    # Validate required columns
    required_cols = ['timestamp_utc', 'grid_co2e_kg_per_mwh_avg', 'grid_co2e_kg_per_mwh_marginal']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"grid_emissions_intensity_{epoch}.csv missing required columns: {missing}"
        )
    
    # Parse timestamps
    df['timestamp_utc'] = parse_any_timestamp(df['timestamp_utc'])
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Return with required columns
    result = df[required_cols].copy()
    
    return result


def align_signals_to_demand(
    signals_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    signal_name: str = "signals"
) -> pd.DataFrame:
    """
    Align signals DataFrame to demand DataFrame timestamps exactly.
    
    Performs strict merge on timestamp_utc and validates alignment.
    
    Args:
        signals_df: Signals DataFrame with timestamp_utc
        demand_df: Demand DataFrame with timestamp_utc
        signal_name: Name of signal type (for error messages)
        
    Returns:
        Aligned signals DataFrame with same timestamps as demand_df
        
    Raises:
        ValueError: If timestamps don't align exactly
    """
    # Canonicalize timestamps for comparison
    demand_ts = parse_any_timestamp(demand_df['timestamp_utc']).sort_values().unique()
    signals_ts = parse_any_timestamp(signals_df['timestamp_utc']).sort_values().unique()
    
    if len(demand_ts) != len(signals_ts):
        raise ValueError(
            f"Timestamp count mismatch for {signal_name}: "
            f"demand has {len(demand_ts)} timestamps, {signal_name} has {len(signals_ts)}. "
            f"Demand range: {demand_ts[0]} to {demand_ts[-1]}. "
            f"{signal_name.capitalize()} range: {signals_ts[0]} to {signals_ts[-1]}."
        )
    
    # Merge on timestamp_utc
    aligned = demand_df[['timestamp_utc']].merge(
        signals_df,
        on='timestamp_utc',
        how='left',
        validate='one_to_one'
    )
    
    # Check for missing values
    missing = aligned.isna().any(axis=1)
    if missing.any():
        missing_ts = aligned.loc[missing, 'timestamp_utc'].head(5).tolist()
        raise ValueError(
            f"Timestamp alignment failed for {signal_name}: "
            f"{missing.sum()} timestamps from demand not found in {signal_name}. "
            f"Sample missing timestamps: {missing_ts}"
        )
    
    return aligned


