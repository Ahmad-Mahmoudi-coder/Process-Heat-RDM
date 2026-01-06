"""
Plot DemandPack for thesis-ready figures.

Generates thesis-ready plots for DemandPack hourly heat demand profiles.
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
from src.time_utils import parse_any_timestamp


def load_demandpack_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load DemandPack CSV and prepare for plotting.
    
    Args:
        csv_path: Path to hourly_heat_demand CSV
        
    Returns:
        DataFrame with timestamp and heat_demand_MW columns
    """
    df = pd.read_csv(csv_path)
    
    # Parse timestamp (support both timestamp_utc and timestamp)
    if 'timestamp_utc' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp_utc'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp'])
    else:
        raise ValueError("CSV must have either 'timestamp' or 'timestamp_utc' column")
    
    # Ensure heat_demand_MW column exists
    if 'heat_demand_MW' not in df.columns:
        raise ValueError("CSV must have 'heat_demand_MW' column")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Add derived columns for plotting
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['date'] = df['timestamp'].dt.date
    
    return df


def plot_timeseries_week(df: pd.DataFrame, output_path: Path, epoch_tag: str) -> None:
    """
    Plot one representative week (e.g., first week of February).
    
    Args:
        df: DataFrame with timestamp and heat_demand_MW
        output_path: Output PNG path
        epoch_tag: Epoch tag for title
    """
    # Select first week of February (or first available week if Feb doesn't exist)
    feb_week = df[df['month'] == 2].head(168)  # First week of February (168 hours = 7 days)
    
    if len(feb_week) == 0:
        # Fallback to first week of any month
        feb_week = df.head(168)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the week
    ax.plot(range(len(feb_week)), feb_week['heat_demand_MW'].values,
           linewidth=1.5, color='#2c3e50', alpha=0.8)
    
    # Add day separators
    for day in range(1, 7):
        ax.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Hour of Week', fontsize=12)
    ax.set_ylabel('Heat Demand (MW)', fontsize=12)
    ax.set_title(f'Hourly Heat Demand - {epoch_tag} (Representative Week)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add day labels
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xticks([i * 24 + 12 for i in range(7)])
    ax.set_xticklabels(day_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_timeseries_sample(df: pd.DataFrame, output_path: Path, epoch_tag: str) -> None:
    """
    Plot time series sample showing representative weeks (e.g., Jan + Jun + Oct).
    
    Args:
        df: DataFrame with timestamp and heat_demand_MW
        output_path: Output PNG path
        epoch_tag: Epoch tag for title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Select representative weeks: January (week 1), June (week ~24), October (week ~40)
    jan_week = df[df['month'] == 1].head(168)  # First week of January
    jun_week = df[df['month'] == 6].head(168)   # First week of June
    oct_week = df[df['month'] == 10].head(168)  # First week of October
    
    # Plot each week with offset x-axis
    if len(jan_week) > 0:
        x_offset = 0
        ax.plot(range(x_offset, x_offset + len(jan_week)), jan_week['heat_demand_MW'].values,
               linewidth=1.5, color='#2c3e50', label='January (sample week)', alpha=0.8)
    
    if len(jun_week) > 0:
        x_offset = 168
        ax.plot(range(x_offset, x_offset + len(jun_week)), jun_week['heat_demand_MW'].values,
               linewidth=1.5, color='#3498db', label='June (sample week)', alpha=0.8)
    
    if len(oct_week) > 0:
        x_offset = 336
        ax.plot(range(x_offset, x_offset + len(oct_week)), oct_week['heat_demand_MW'].values,
               linewidth=1.5, color='#e74c3c', label='October (sample week)', alpha=0.8)
    
    ax.set_xlabel('Hour (sample weeks)', fontsize=12)
    ax.set_ylabel('Heat Demand (MW)', fontsize=12)
    ax.set_title(f'Hourly Heat Demand - {epoch_tag} (Sample Weeks)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add week separators
    ax.axvline(x=168, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(x=336, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_duration_curve(df: pd.DataFrame, output_path: Path, epoch_tag: str) -> None:
    """
    Plot load duration curve (sorted descending).
    
    Args:
        df: DataFrame with heat_demand_MW
        output_path: Output PNG path
        epoch_tag: Epoch tag for title
    """
    sorted_load = np.sort(df['heat_demand_MW'].values)[::-1]
    hours = np.arange(1, len(sorted_load) + 1)
    percent_of_time = (hours / len(sorted_load)) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(percent_of_time, sorted_load, linewidth=2, color='#2c3e50')
    
    ax.set_xlabel('Percent of Time (%)', fontsize=12)
    ax.set_ylabel('Heat Demand (MW)', fontsize=12)
    ax.set_title(f'Load Duration Curve - {epoch_tag}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_heatmap_day_hour(df: pd.DataFrame, output_path: Path, epoch_tag: str) -> None:
    """
    Plot heatmap of average demand by day of week and hour of day.
    
    Args:
        df: DataFrame with timestamp, heat_demand_MW, hour, day_of_week
        output_path: Output PNG path
        epoch_tag: Epoch tag for title
    """
    # Group by day of week and hour
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=weekday_order, ordered=True)
    
    heatmap_data = df.groupby(['day_of_week', 'hour'])['heat_demand_MW'].mean().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(heatmap_data.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(len(weekday_order)))
    ax.set_yticklabels([d[:3] for d in weekday_order])  # Mon, Tue, Wed, etc.
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)
    ax.set_title(f'Average Heat Demand by Day and Hour - {epoch_tag}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Heat Demand (MW)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_monthly_totals(df: pd.DataFrame, output_path: Path, epoch_tag: str) -> None:
    """
    Plot monthly total heat (GWh) as bar chart.
    
    Args:
        df: DataFrame with timestamp and heat_demand_MW
        output_path: Output PNG path
        epoch_tag: Epoch tag for title
    """
    monthly_totals = df.groupby('month')['heat_demand_MW'].sum() / 1000.0  # Convert to GWh
    
    fig, ax = plt.subplots(figsize=(12, 6))
    months = monthly_totals.index
    bars = ax.bar(months, monthly_totals.values, color='#3498db', alpha=0.7, edgecolor='#2c3e50', linewidth=1)
    
    # Add value labels on bars
    for month, value in zip(months, monthly_totals.values):
        ax.text(month, value, f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Total Heat Demand (GWh)', fontsize=12)
    ax.set_title(f'Monthly Total Heat Demand - {epoch_tag}', fontsize=14, fontweight='bold')
    ax.set_xticks(months)
    ax.set_xticklabels([f'M{m}' for m in months])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_avg_diurnal_by_season(df: pd.DataFrame, output_path: Path, epoch_tag: str) -> None:
    """
    Plot average diurnal profile by season (or weekday/weekend split).
    
    Uses simple season definition: Dec-Feb (summer), Mar-May (autumn), Jun-Aug (winter), Sep-Nov (spring).
    
    Args:
        df: DataFrame with timestamp, heat_demand_MW, hour
        output_path: Output PNG path
        epoch_tag: Epoch tag for title
    """
    # Define seasons based on month
    def get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:  # 9, 10, 11
            return 'Spring'
    
    df['season'] = df['month'].apply(get_season)
    
    # Compute average hourly profile by season
    hourly_by_season = df.groupby(['season', 'hour'])['heat_demand_MW'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
    colors = {'Summer': '#e74c3c', 'Autumn': '#f39c12', 'Winter': '#3498db', 'Spring': '#2ecc71'}
    
    for season in seasons:
        season_data = hourly_by_season[hourly_by_season['season'] == season]
        if len(season_data) > 0:
            ax.plot(season_data['hour'], season_data['heat_demand_MW'],
                   marker='o', linewidth=2, label=season, color=colors[season], markersize=4)
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Average Heat Demand (MW)', fontsize=12)
    ax.set_title(f'Average Diurnal Profile by Season - {epoch_tag}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def generate_demandpack_figures(csv_path: Path, figures_dir: Path, epoch: int) -> None:
    """
    Generate all required DemandPack figures during generation.
    
    Args:
        csv_path: Path to hourly_heat_demand CSV file
        figures_dir: Output directory for figures (e.g., <demandpack_run_dir>/demandpack/figures/)
        epoch: Epoch year (e.g., 2020, 2025, 2028)
    """
    # Create output directory if needed
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"[DemandPack Figures] Loading data from {csv_path}...")
    df = load_demandpack_csv(csv_path)
    print(f"[DemandPack Figures] Loaded {len(df)} hourly records")
    
    # Generate all required figures
    print(f"[DemandPack Figures] Generating figures for epoch {epoch}...")
    
    # 1. Time series week (one representative week, e.g., first week of Feb)
    plot_timeseries_week(df, figures_dir / f'demand_{epoch}_timeseries_week.png', str(epoch))
    
    # 2. Duration curve
    plot_duration_curve(df, figures_dir / f'demand_{epoch}_duration_curve.png', str(epoch))
    
    # 3. Monthly energy (totals)
    plot_monthly_totals(df, figures_dir / f'demand_{epoch}_monthly_energy.png', str(epoch))
    
    # 4. Daily profile seasonal (mean diurnal by season)
    plot_avg_diurnal_by_season(df, figures_dir / f'demand_{epoch}_daily_profile_seasonal.png', str(epoch))
    
    print(f"[DemandPack Figures] All figures generated in {figures_dir}")


def main():
    """CLI entrypoint for DemandPack plotting."""
    parser = argparse.ArgumentParser(
        description='Generate thesis-ready plots for DemandPack hourly heat demand'
    )
    parser.add_argument('--input-csv', type=str, required=True,
                       help='Path to hourly_heat_demand CSV file')
    parser.add_argument('--epoch-tag', type=str, required=True,
                       help='Epoch tag (e.g., 2020, 2025, 2035_EB, 2035_BB)')
    parser.add_argument('--outdir', type=str, required=True,
                       help='Output directory for figures (e.g., bundle epoch demandpack/figures)')
    
    args = parser.parse_args()
    
    # Resolve paths
    input_csv = Path(resolve_path(args.input_csv))
    outdir = Path(resolve_path(args.outdir))
    
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    # Create output directory if needed
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading DemandPack data from {input_csv}...")
    df = load_demandpack_csv(input_csv)
    print(f"Loaded {len(df)} hourly records")
    
    # Generate all figures
    print(f"Generating figures for {args.epoch_tag}...")
    
    # 1. Time series sample
    plot_timeseries_sample(df, outdir / f'demand_{args.epoch_tag}_timeseries_sample.png', args.epoch_tag)
    
    # 2. Duration curve
    plot_duration_curve(df, outdir / f'demand_{args.epoch_tag}_duration_curve.png', args.epoch_tag)
    
    # 3. Heatmap day-hour
    plot_heatmap_day_hour(df, outdir / f'demand_{args.epoch_tag}_heatmap_day_hour.png', args.epoch_tag)
    
    # 4. Monthly totals
    plot_monthly_totals(df, outdir / f'demand_{args.epoch_tag}_monthly_totals.png', args.epoch_tag)
    
    # 5. Average diurnal by season
    plot_avg_diurnal_by_season(df, outdir / f'demand_{args.epoch_tag}_avg_diurnal_by_season.png', args.epoch_tag)
    
    print(f"\n[OK] All figures generated in {outdir}")


if __name__ == '__main__':
    main()

