"""
Diagnostic plotting for DemandPack output.

Generates publication-ready figures for the synthetic hourly heat demand profile.
"""

# Bootstrap: allow `python .\src\script.py` (adds repo root to sys.path)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Any

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Need tomllib (Python 3.11+) or tomli package")

from src.path_utils import repo_root, resolve_path, resolve_cfg_path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse TOML configuration file."""
    with open(config_path, 'rb') as f:
        return tomllib.load(f)


def load_seasonal_labels(config: Dict[str, Any], config_path: Path) -> Dict[int, str]:
    """Load seasonal labels from seasonal_factors.csv."""
    seasonality = config['seasonality']
    season_file_path = resolve_cfg_path(config_path, seasonality['file'])
    if not season_file_path.exists():
        raise FileNotFoundError(f"Seasonal factors file not found: {season_file_path} (resolved from {seasonality['file']})")
    df = pd.read_csv(season_file_path)
    
    month_col = seasonality['month_col']
    factor_col = seasonality['factor_col']
    season_col = 'season'  # Column name in CSV
    
    if season_col not in df.columns:
        # Fallback: assign based on scaling_factor
        df[season_col] = 'shoulder_season'
        df.loc[df[factor_col] >= 0.9, season_col] = 'peak_season'
        df.loc[df[factor_col] < 0.5, season_col] = 'off_season'
    
    return dict(zip(df[month_col], df[season_col]))


def plot_hourly_timeseries_core(df: pd.DataFrame, output_path: str):
    """
    Core function to plot hourly heat demand time series with 7-day rolling mean.
    Dark, thick line for hourly values; thin, dashed line for rolling mean.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Dark, thick line for hourly values
    ax.plot(
        df['timestamp'],
        df['heat_demand_MW'],
        color='#4a90e2',      # darker blue
        alpha=0.8,
        linewidth=1.0,        # thicker
        label='Hourly',
        zorder=1,
    )

    # Thin, dashed line for 7-day rolling mean
    rolling_7d = df['heat_demand_MW'].rolling(window=7*24, center=True).mean()
    ax.plot(
        df['timestamp'],
        rolling_7d,
        color='navy',
        linewidth=1.2,        # thinner
        linestyle='--',       # dashed
        alpha=0.9,
        label='7-day rolling mean',
        zorder=2,
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Heat Demand (MW)', fontsize=12)
    ax.set_title('Hourly Heat Demand 2020 - Site 11_031', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_hourly_timeseries(df: pd.DataFrame, output_path: str):
    """Wrapper for core timeseries plot (for backward compatibility)."""
    plot_hourly_timeseries_core(df, output_path)


def plot_daily_envelope(df: pd.DataFrame, output_path: str):
    """
    Plot daily min/max envelope across the year.
    
    Args:
        df: DataFrame with 'timestamp' and 'heat_demand_MW' columns
        output_path: Full path to save the figure (e.g., "Output/heat_2020_daily_envelope.png")
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['date'] = df['timestamp'].dt.date
    daily_stats = df.groupby('date')['heat_demand_MW'].agg(['min', 'max'])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    dates = pd.to_datetime(daily_stats.index)
    ax.fill_between(dates, daily_stats['min'], daily_stats['max'],
                    alpha=0.3, color='steelblue', label='Daily min/max envelope')
    ax.plot(dates, daily_stats['min'], color='darkblue', linewidth=0.8, alpha=0.7)
    ax.plot(dates, daily_stats['max'], color='darkblue', linewidth=0.8, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Heat Demand (MW)', fontsize=12)
    ax.set_title('Daily Min/Max Envelope - 2020', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_hourly_means_by_season(df: pd.DataFrame, config: Dict[str, Any], 
                                 output_path: str, config_path: Path):
    """Plot average hourly profile grouped by season."""
    # Ensure timestamp is datetime and extract hour
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    
    # Load seasonal labels
    seasonal_labels = load_seasonal_labels(config, config_path)
    
    # Map months to seasons
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['month'].map(seasonal_labels)
    
    # Compute average hourly profile by season
    hourly_by_season = df.groupby(['season', 'hour'])['heat_demand_MW'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    seasons = sorted(hourly_by_season['season'].unique())
    colors = {'peak_season': 'red', 'shoulder_season': 'orange', 'off_season': 'blue'}
    
    for season in seasons:
        season_data = hourly_by_season[hourly_by_season['season'] == season]
        label = season.replace('_', ' ').title()
        color = colors.get(season, 'gray')
        ax.plot(season_data['hour'], season_data['heat_demand_MW'],
               marker='o', linewidth=2, label=label, color=color)
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Average Heat Demand (MW)', fontsize=12)
    ax.set_title('Average Hourly Profile by Season - 2020', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_load_duration_curve(df: pd.DataFrame, output_path: str):
    """Plot load duration curve (sorted descending)."""
    sorted_load = np.sort(df['heat_demand_MW'].values)[::-1]
    hours = np.arange(1, len(sorted_load) + 1)
    percent_of_time = (hours / len(sorted_load)) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(percent_of_time, sorted_load, linewidth=2, color='darkblue')
    
    ax.set_xlabel('Percent of Time (%)', fontsize=12)
    ax.set_ylabel('Heat Demand (MW)', fontsize=12)
    ax.set_title('Load Duration Curve - 2020', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_monthly_totals(df: pd.DataFrame, output_path: str):
    """Plot monthly total heat (GWh) as bar chart."""
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.month
    
    monthly_totals = df.groupby('month')['heat_demand_MW'].sum() / 1000.0  # Convert to GWh
    
    fig, ax = plt.subplots(figsize=(12, 6))
    months = monthly_totals.index
    bars = ax.bar(months, monthly_totals.values, color='steelblue', alpha=0.7)
    
    # Add value labels on bars (1 decimal place)
    for month, value in zip(months, monthly_totals.values):
        ax.text(month, value, f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Total Heat Demand (GWh)', fontsize=12)
    ax.set_title('Monthly Total Heat Demand - 2020', fontsize=14, fontweight='bold')
    ax.set_xticks(months)
    ax.set_xticklabels([f'M{m}' for m in months])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_weekday_profiles_for_month(df: pd.DataFrame, month: int, output_path: str):
    """
    Plot weekday profiles for a specific month.
    
    Groups by day-of-week and hour, then plots Monday-Sunday as separate lines
    on a single 24-hour plot.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for the specified month
    month_data = df[df['timestamp'].dt.month == month].copy()
    
    if len(month_data) == 0:
        print(f"[WARNING] No data found for month {month}")
        return
    
    # Extract day of week and hour
    month_data['day_of_week'] = month_data['timestamp'].dt.day_name()
    if 'hour' not in month_data.columns:
        month_data['hour'] = month_data['timestamp'].dt.hour
    
    # Group by day of week and hour, compute mean
    weekday_hourly = month_data.groupby(['day_of_week', 'hour'])['heat_demand_MW'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Order weekdays
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    colors = plt.cm.tab10(range(7))
    
    for i, day in enumerate(weekday_order):
        day_data = weekday_hourly[weekday_hourly['day_of_week'] == day]
        if len(day_data) > 0:
            ax.plot(day_data['hour'], day_data['heat_demand_MW'],
                   linewidth=2, label=day, color=colors[i])  # removed marker='o'
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Average Heat Demand (MW)', fontsize=12)
    month_name = pd.Timestamp(2020, month, 1).strftime('%B')
    ax.set_title(f'Weekday Profiles - {month_name} 2020', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_load_histogram(df: pd.DataFrame, output_path: str):
    """Plot histogram of hourly load with mean line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram (30-40 bins)
    n_bins = 35
    ax.hist(df['heat_demand_MW'], bins=n_bins, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add vertical line for mean
    mean_load = df['heat_demand_MW'].mean()
    ax.axvline(mean_load, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_load:.2f} MW')
    
    ax.set_xlabel('Heat Demand (MW)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Hourly Heat Demand - 2020', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def main():
    """CLI entrypoint for full diagnostics mode."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate DemandPack diagnostic plots')
    parser.add_argument('--config', default='Input/demandpack_config.toml',
                       help='Path to config TOML file')
    parser.add_argument('--data', default='Output/hourly_heat_demand_2020.csv',
                       help='Path to generated demand CSV')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: Output/)')
    args = parser.parse_args()
    
    # Resolve paths
    ROOT = repo_root()
    config_path_resolved = resolve_path(args.config)
    data_path_resolved = resolve_path(args.data)
    
    print(f"Repository root: {ROOT}")
    print(f"Config path: {config_path_resolved}")
    
    if not config_path_resolved.exists():
        print(f"[ERROR] Config file not found: {config_path_resolved}")
        sys.exit(1)
    
    # Load config
    config = load_config(str(config_path_resolved))
    
    # Determine output directory
    if args.output_dir:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = ROOT / 'Output'
    figures_dir = output_dir / 'Figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path_resolved}...")
    if not data_path_resolved.exists():
        print(f"[ERROR] Data file not found: {data_path_resolved}")
        sys.exit(1)
    df = pd.read_csv(data_path_resolved)
    # Support both timestamp and timestamp_utc for backward compatibility
    from src.time_utils import parse_any_timestamp
    if 'timestamp_utc' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp_utc'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp'])
    else:
        raise ValueError("CSV must have either 'timestamp' or 'timestamp_utc' column")
    df = df.sort_values('timestamp').reset_index(drop=True)
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    
    print("Generating all diagnostic plots...")
    
    # Track generated files
    generated_files = []
    
    # 1. Annual hourly time series
    output_path = str(figures_dir / 'heat_2020_timeseries.png')
    plot_hourly_timeseries_core(df, output_path)
    generated_files.append(output_path)
    
    # 2. Daily envelope
    output_path = str(figures_dir / 'heat_2020_daily_envelope.png')
    plot_daily_envelope(df, output_path)
    generated_files.append(output_path)
    
    # 3. Average hourly profile by season
    output_path = str(figures_dir / 'heat_2020_hourly_means_by_season.png')
    plot_hourly_means_by_season(df, config, output_path, config_path_resolved)
    generated_files.append(output_path)
    
    # 4. Monthly totals
    output_path = str(figures_dir / 'heat_2020_monthly_totals.png')
    plot_monthly_totals(df, output_path)
    generated_files.append(output_path)
    
    # 5. Load duration curve
    output_path = str(figures_dir / 'heat_2020_LDC.png')
    plot_load_duration_curve(df, output_path)
    generated_files.append(output_path)
    
    # 6. Weekday profiles for February (peak season)
    output_path = str(figures_dir / 'heat_2020_weekday_profiles_Feb.png')
    plot_weekday_profiles_for_month(df, 2, output_path)
    generated_files.append(output_path)
    
    # 7. Weekday profiles for June (low season)
    output_path = str(figures_dir / 'heat_2020_weekday_profiles_Jun.png')
    plot_weekday_profiles_for_month(df, 6, output_path)
    generated_files.append(output_path)
    
    # 8. Load histogram
    output_path = str(figures_dir / 'heat_2020_load_histogram.png')
    plot_load_histogram(df, output_path)
    generated_files.append(output_path)
    
    print("\n" + "="*60)
    print("All diagnostic plots generated successfully!")
    print("="*60)
    print("\nGenerated figures:")
    for fpath in generated_files:
        print(f"  - {fpath}")
    print("="*60)


if __name__ == '__main__':
    # Default behavior: simple plotting mode (hourly timeseries and daily envelope)
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot DemandPack hourly timeseries and daily envelope')
    parser.add_argument('--data', default='Output/hourly_heat_demand_2020.csv',
                       help='Path to generated demand CSV')
    parser.add_argument('--full-diagnostics', action='store_true',
                       help='Generate all diagnostic plots (default: just timeseries and envelope)')
    parser.add_argument('--config', default='Input/demandpack_config.toml',
                       help='Path to config TOML file (for full diagnostics)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: Output/)')
    args = parser.parse_args()
    
    if args.full_diagnostics:
        # Run full diagnostic suite - need to pass args to main
        import sys
        # Temporarily replace sys.argv to pass args to main()
        old_argv = sys.argv
        sys.argv = [sys.argv[0], '--config', args.config, '--data', args.data]
        if args.output_dir:
            sys.argv.extend(['--output-dir', args.output_dir])
        try:
            main()
        finally:
            sys.argv = old_argv
    else:
        # Simple mode: load data and generate two plots
        ROOT = repo_root()
        data_path_resolved = resolve_path(args.data)
        print(f"Loading data from {data_path_resolved}...")
        if not data_path_resolved.exists():
            print(f"[ERROR] Data file not found: {data_path_resolved}")
            sys.exit(1)
        df = pd.read_csv(data_path_resolved)
        # Support both timestamp and timestamp_utc for backward compatibility
        from src.time_utils import parse_any_timestamp
        if 'timestamp_utc' in df.columns:
            df['timestamp'] = parse_any_timestamp(df['timestamp_utc'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = parse_any_timestamp(df['timestamp'])
        else:
            raise ValueError("CSV must have either 'timestamp' or 'timestamp_utc' column")
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Determine output directory
        ROOT = repo_root()
        if args.output_dir:
            output_dir = resolve_path(args.output_dir)
        else:
            output_dir = ROOT / 'Output'
        figures_dir = output_dir / 'Figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate the two requested plots
        print("Generating plots...")
        plot_hourly_timeseries(df, str(figures_dir / "heat_2020_timeseries.png"))
        plot_daily_envelope(df, str(figures_dir / "heat_2020_daily_envelope.png"))
        
        # Print summary statistics
        print("\n" + "="*50)
        print("Data Summary (for sanity checking):")
        print("="*50)
        min_MW = df['heat_demand_MW'].min()
        mean_MW = df['heat_demand_MW'].mean()
        max_MW = df['heat_demand_MW'].max()
        total_GWh = df['heat_demand_MW'].sum() / 1000.0
        
        print(f"  Min hourly load:  {min_MW:.2f} MW")
        print(f"  Mean hourly load: {mean_MW:.2f} MW")
        print(f"  Max hourly load:  {max_MW:.2f} MW")
        print(f"  Total energy:     {total_GWh:.6f} GWh")
        print("="*50)
        
        if abs(total_GWh - 819.0) < 0.1:
            print("[OK] Total energy matches target (~819 GWh)")
        else:
            print(f"[WARNING] Total energy ({total_GWh:.2f} GWh) differs from target (819 GWh)")

