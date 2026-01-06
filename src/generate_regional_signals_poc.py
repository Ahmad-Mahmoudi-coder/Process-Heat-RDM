"""
Generate deterministic PoC regional electricity signals (headroom, tariff, capacity).

This module generates deterministic regional electricity signals for all epochs
using the same logic embedded in RDM screening. These are reporting artefacts only
(no coupling/feedback).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional

from src.path_utils import repo_root, resolve_path
from src.time_utils import parse_any_timestamp, build_hourly_utc_index
from src.load_signals import load_signals_config, get_signals_for_epoch


def generate_deterministic_headroom(timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    Generate deterministic PoC headroom with seasonal pattern.
    
    Uses the same logic as gxp_rdm_screen.load_or_generate_headroom():
    - Base headroom: 50 MW
    - Seasonal variation: ±20 MW (winter lower, summer higher)
    - Weekly pattern: weekend +5 MW
    - Minimum: 10 MW
    
    Args:
        timestamps: DatetimeIndex for headroom series
        
    Returns:
        Series with headroom_MW indexed by timestamp
    """
    n_hours = len(timestamps)
    base_headroom = 50.0
    
    # Seasonal pattern (sine wave, peak in summer)
    day_of_year = timestamps.dayofyear
    seasonal = 20.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)  # Peak around day 80 (March)
    
    # Weekly pattern (weekend +5 MW)
    is_weekend = timestamps.dayofweek >= 5
    weekly = np.where(is_weekend, 5.0, 0.0)
    
    headroom = base_headroom + seasonal + weekly
    headroom = np.maximum(headroom, 10.0)  # Minimum 10 MW
    
    result = pd.Series(headroom, index=timestamps, name='headroom_MW')
    return result


def generate_deterministic_tariff(timestamps: pd.DatetimeIndex, epoch_tag: str) -> pd.Series:
    """
    Generate deterministic PoC tariff.
    
    Uses flat tariff from signals config, or simple time-of-use pattern.
    For PoC, uses flat tariff from signals config for the epoch.
    
    Args:
        timestamps: DatetimeIndex for tariff series
        epoch_tag: Epoch tag (e.g., '2020', '2035_EB')
        
    Returns:
        Series with tariff_nzd_per_MWh indexed by timestamp
    """
    # Extract year from epoch_tag (handle variants like 2035_EB -> 2035)
    if '_' in epoch_tag:
        year = int(epoch_tag.split('_')[0])
    else:
        year = int(epoch_tag)
    
    # Load signals config
    signals_config = load_signals_config()
    
    # Find appropriate epoch in signals config
    # Try exact match first, then find closest year
    epoch_key = None
    if epoch_tag in signals_config.get('epochs', {}):
        epoch_key = epoch_tag
    elif str(year) in signals_config.get('epochs', {}):
        epoch_key = str(year)
    else:
        # Find closest epoch
        available_years = [int(k) for k in signals_config.get('epochs', {}).keys() if k.isdigit()]
        if available_years:
            closest_year = min(available_years, key=lambda x: abs(x - year))
            epoch_key = str(closest_year)
    
    if epoch_key is None:
        # Fallback to default
        print(f"[WARN] No signals config found for {epoch_tag}, using default tariff 120 NZD/MWh")
        base_tariff = 120.0
    else:
        epoch_signals = signals_config['epochs'][epoch_key]
        base_tariff = epoch_signals.get('elec_price_flat_nzd_per_MWh', 120.0)
        print(f"[OK] Using tariff from signals config: {base_tariff} NZD/MWh for epoch {epoch_key}")
    
    # Generate flat tariff (all hours same)
    tariff = pd.Series(base_tariff, index=timestamps, name='tariff_nzd_per_MWh')
    
    return tariff


def generate_gxp_capacity(timestamps: pd.DatetimeIndex, epoch_tag: str) -> pd.Series:
    """
    Generate GXP capacity (scalar or timeseries).
    
    For PoC, uses a constant capacity value. Can be extended to timeseries if needed.
    
    Args:
        timestamps: DatetimeIndex for capacity series
        epoch_tag: Epoch tag
        
    Returns:
        Series with gxp_capacity_MW indexed by timestamp
    """
    # For PoC, use constant capacity
    # This could be made epoch-specific or loaded from config
    gxp_capacity_MW = 200.0  # Default PoC capacity
    
    capacity = pd.Series(gxp_capacity_MW, index=timestamps, name='gxp_capacity_MW')
    return capacity


def plot_headroom_duration_curve(headroom: pd.Series, output_path: Path, epoch_tag: str) -> None:
    """Plot headroom duration curve (sorted descending)."""
    sorted_headroom = np.sort(headroom.values)[::-1]
    hours = np.arange(1, len(sorted_headroom) + 1)
    percent_of_time = (hours / len(sorted_headroom)) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(percent_of_time, sorted_headroom, linewidth=2, color='#2c3e50')
    
    ax.set_xlabel('Percent of Time (%)', fontsize=12)
    ax.set_ylabel('Headroom (MW)', fontsize=12)
    ax.set_title(f'Headroom Duration Curve - {epoch_tag}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_headroom_timeseries_week(headroom: pd.Series, output_path: Path, epoch_tag: str) -> None:
    """Plot one representative week of headroom (first week of February)."""
    # Select first week of February (or first available week)
    feb_mask = headroom.index.month == 2
    feb_week = headroom[feb_mask].head(168)  # 168 hours = 7 days
    
    if len(feb_week) == 0:
        # Fallback to first week
        feb_week = headroom.head(168)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the week
    ax.plot(range(len(feb_week)), feb_week.values, linewidth=1.5, color='#2c3e50', alpha=0.8)
    
    # Add day separators
    for day in range(1, 7):
        ax.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Hour of Week', fontsize=12)
    ax.set_ylabel('Headroom (MW)', fontsize=12)
    ax.set_title(f'Headroom Time Series - {epoch_tag} (Representative Week)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add day labels
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xticks([i * 24 + 12 for i in range(7)])
    ax.set_xticklabels(day_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_tariff_timeseries_week(tariff: pd.Series, output_path: Path, epoch_tag: str) -> None:
    """Plot one representative week of tariff (first week of February)."""
    # Select first week of February (or first available week)
    feb_mask = tariff.index.month == 2
    feb_week = tariff[feb_mask].head(168)  # 168 hours = 7 days
    
    if len(feb_week) == 0:
        # Fallback to first week
        feb_week = tariff.head(168)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the week
    ax.plot(range(len(feb_week)), feb_week.values, linewidth=1.5, color='#3498db', alpha=0.8)
    
    # Add day separators
    for day in range(1, 7):
        ax.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Hour of Week', fontsize=12)
    ax.set_ylabel('Tariff (NZD/MWh)', fontsize=12)
    ax.set_title(f'Tariff Time Series - {epoch_tag} (Representative Week)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add day labels
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xticks([i * 24 + 12 for i in range(7)])
    ax.set_xticklabels(day_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def main():
    """CLI entrypoint for regional signals generation."""
    parser = argparse.ArgumentParser(
        description='Generate deterministic PoC regional electricity signals'
    )
    parser.add_argument('--epoch-tag', type=str, required=True,
                       help='Epoch tag (e.g., 2020, 2025, 2035_EB, 2035_BB)')
    parser.add_argument('--outdir', type=str, required=True,
                       help='Output directory (signals and figures subdirectories will be created)')
    
    args = parser.parse_args()
    
    # Resolve paths
    outdir = Path(resolve_path(args.outdir))
    signals_dir = outdir / 'signals'
    figures_dir = outdir / 'figures'
    
    signals_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract year from epoch_tag
    if '_' in args.epoch_tag:
        year = int(args.epoch_tag.split('_')[0])
    else:
        year = int(args.epoch_tag)
    
    # Build hourly index for the year
    print(f"[Regional Signals] Generating signals for {args.epoch_tag} (year {year})...")
    timestamps = build_hourly_utc_index(year)
    timestamps = pd.DatetimeIndex(timestamps)
    
    # Generate signals
    print(f"[Regional Signals] Generating headroom...")
    headroom = generate_deterministic_headroom(timestamps)
    
    print(f"[Regional Signals] Generating tariff...")
    tariff = generate_deterministic_tariff(timestamps, args.epoch_tag)
    
    print(f"[Regional Signals] Generating GXP capacity...")
    gxp_capacity = generate_gxp_capacity(timestamps, args.epoch_tag)
    
    # Save CSVs
    print(f"[Regional Signals] Saving CSVs...")
    
    # Headroom CSV
    headroom_df = pd.DataFrame({
        'timestamp_utc': headroom.index,
        'headroom_MW': headroom.values
    })
    headroom_csv = signals_dir / f'headroom_MW_{args.epoch_tag}.csv'
    headroom_df.to_csv(headroom_csv, index=False)
    print(f"[OK] Saved {headroom_csv.name}")
    
    # Tariff CSV
    tariff_df = pd.DataFrame({
        'timestamp_utc': tariff.index,
        'tariff_nzd_per_MWh': tariff.values
    })
    tariff_csv = signals_dir / f'tariff_nzd_per_MWh_{args.epoch_tag}.csv'
    tariff_df.to_csv(tariff_csv, index=False)
    print(f"[OK] Saved {tariff_csv.name}")
    
    # GXP capacity CSV (can be scalar or timeseries)
    gxp_capacity_df = pd.DataFrame({
        'timestamp_utc': gxp_capacity.index,
        'gxp_capacity_MW': gxp_capacity.values
    })
    gxp_capacity_csv = signals_dir / f'gxp_capacity_MW_{args.epoch_tag}.csv'
    gxp_capacity_df.to_csv(gxp_capacity_csv, index=False)
    print(f"[OK] Saved {gxp_capacity_csv.name}")
    
    # Generate figures
    print(f"[Regional Signals] Generating figures...")
    
    plot_headroom_duration_curve(headroom, figures_dir / f'headroom_duration_curve_{args.epoch_tag}.png', args.epoch_tag)
    plot_headroom_timeseries_week(headroom, figures_dir / f'headroom_timeseries_week_{args.epoch_tag}.png', args.epoch_tag)
    plot_tariff_timeseries_week(tariff, figures_dir / f'tariff_timeseries_week_{args.epoch_tag}.png', args.epoch_tag)
    
    print(f"\n[OK] Regional signals generation complete for {args.epoch_tag}")
    print(f"  Signals: {signals_dir}")
    print(f"  Figures: {figures_dir}")


if __name__ == '__main__':
    main()

