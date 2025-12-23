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
from typing import Dict, Any, Union

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Need tomllib (Python 3.11+) or tomli package")

from src.path_utils import repo_root, resolve_path, resolve_cfg_path
import re


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse TOML configuration file."""
    with open(config_path, 'rb') as f:
        return tomllib.load(f)


def load_seasonal_labels(config: Dict[str, Any], config_path: Union[str, Path]) -> Dict[int, str]:
    """
    Load seasonal labels from seasonal_factors.csv.
    
    Args:
        config: Parsed config dict
        config_path: Path to config file (str or Path)
        
    Returns:
        Dict mapping month number to season label
        
    Raises:
        FileNotFoundError: If seasonal factors file not found
        KeyError: If required config keys missing
    """
    # Normalize config_path to Path
    config_path = Path(config_path).resolve() if isinstance(config_path, str) else config_path
    
    if 'seasonality' not in config:
        raise KeyError("Config missing 'seasonality' section")
    
    seasonality = config['seasonality']
    if 'file' not in seasonality:
        raise KeyError("Config 'seasonality' section missing 'file' key")
    
    season_file_path = resolve_cfg_path(config_path, seasonality['file'])
    if not season_file_path.exists():
        raise FileNotFoundError(f"Seasonal factors file not found: {season_file_path} (resolved from {seasonality['file']})")
    df = pd.read_csv(season_file_path)
    
    month_col = seasonality.get('month_col', 'month')
    factor_col = seasonality.get('factor_col', 'scaling_factor')
    season_col = 'season'  # Column name in CSV
    
    if season_col not in df.columns:
        # Fallback: assign based on scaling_factor
        df[season_col] = 'shoulder_season'
        df.loc[df[factor_col] >= 0.9, season_col] = 'peak_season'
        df.loc[df[factor_col] < 0.5, season_col] = 'off_season'
    
    return dict(zip(df[month_col], df[season_col]))


def plot_hourly_timeseries_core(df: pd.DataFrame, output_path: str, epoch: int = None):
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
    if epoch:
        ax.set_title(f'Hourly Heat Demand {epoch} - Site 11_031', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Hourly Heat Demand - Site 11_031', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_hourly_timeseries(df: pd.DataFrame, output_path: str, epoch: int = None):
    """Wrapper for core timeseries plot (for backward compatibility)."""
    plot_hourly_timeseries_core(df, output_path, epoch)


def plot_daily_envelope(df: pd.DataFrame, output_path: str, epoch: int = None):
    """
    Plot daily min/max envelope across the year.
    
    Args:
        df: DataFrame with 'timestamp' and 'heat_demand_MW' columns
        output_path: Full path to save the figure (e.g., "Output/heat_2020_daily_envelope.png")
        epoch: Optional epoch year for title
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
    if epoch:
        ax.set_title(f'Daily Min/Max Envelope - {epoch}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Daily Min/Max Envelope', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_hourly_means_by_season(df: pd.DataFrame, config: Dict[str, Any], 
                                 output_path: str, config_path: Union[str, Path], epoch: int = None):
    """Plot average hourly profile grouped by season."""
    # Ensure timestamp is datetime and extract hour
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    
    # Load seasonal labels (with error handling)
    try:
        seasonal_labels = load_seasonal_labels(config, config_path)
    except (FileNotFoundError, KeyError) as e:
        raise ValueError(f"Cannot load seasonal labels: {e}") from e
    
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
    if epoch:
        ax.set_title(f'Average Hourly Profile by Season - {epoch}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Average Hourly Profile by Season', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_load_duration_curve(df: pd.DataFrame, output_path: str, epoch: int = None):
    """Plot load duration curve (sorted descending)."""
    sorted_load = np.sort(df['heat_demand_MW'].values)[::-1]
    hours = np.arange(1, len(sorted_load) + 1)
    percent_of_time = (hours / len(sorted_load)) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(percent_of_time, sorted_load, linewidth=2, color='darkblue')
    
    ax.set_xlabel('Percent of Time (%)', fontsize=12)
    ax.set_ylabel('Heat Demand (MW)', fontsize=12)
    if epoch:
        ax.set_title(f'Load Duration Curve - {epoch}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Load Duration Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_monthly_totals(df: pd.DataFrame, output_path: str, epoch: int = None):
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
    if epoch:
        ax.set_title(f'Monthly Total Heat Demand - {epoch}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Monthly Total Heat Demand', fontsize=14, fontweight='bold')
    ax.set_xticks(months)
    ax.set_xticklabels([f'M{m}' for m in months])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_weekday_profiles_for_month(df: pd.DataFrame, month: int, output_path: str, epoch: int = None):
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
    # Infer year from data if epoch not provided
    if epoch is None:
        year = df['timestamp'].iloc[0].year if len(df) > 0 else 2020
    else:
        year = epoch
    month_name = pd.Timestamp(year, month, 1).strftime('%B')
    if epoch:
        ax.set_title(f'Weekday Profiles - {month_name} {epoch}', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Weekday Profiles - {month_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def plot_load_histogram(df: pd.DataFrame, output_path: str, epoch: int = None):
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
    if epoch:
        ax.set_title(f'Distribution of Hourly Heat Demand - {epoch}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Distribution of Hourly Heat Demand', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


def infer_epoch_from_filename(path: Path) -> int:
    """Infer epoch from filename (e.g., hourly_heat_demand_2020.csv -> 2020)."""
    name = path.stem  # e.g., "hourly_heat_demand_2020"
    match = re.search(r'(\d{4})', name)
    if match:
        return int(match.group(1))
    return None


def main():
    """CLI entrypoint for full diagnostics mode."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate DemandPack diagnostic plots',
        epilog="""
Precedence rules for output paths:
  1. If --run-id is provided:
     - Uses resolve_run_paths() to get run_demandpack_dir, run_figures_dir
     - Default data: run_demandpack_dir/hourly_heat_demand_<epoch>.csv
     - Writes figures to: run_figures_dir
  2. Else if --output-dir is provided (legacy):
     - Uses --output-dir for writing figures
     - Default data: must be provided via --data (no default)
  3. Else (no run-id, no output-dir):
     - Error: must provide either --run-id or --output-dir
        """
    )
    parser.add_argument('--config', default='Input/demandpack_config.toml',
                       help='Path to config TOML file')
    parser.add_argument('--data', default=None,
                       help='Path to generated demand CSV (if not provided, auto-resolved based on precedence)')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Epoch year (inferred from data filename or config name if not provided)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (legacy: use --output-root and --run-id for new system)')
    parser.add_argument('--output-root', type=str, default=None,
                       help='Output root directory (default: repo_root/Output)')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run ID (if provided, uses new output path system with run_dir and latest_dir)')
    parser.add_argument('--full-diagnostics', action='store_true',
                       help='Generate all diagnostic plots (default: just timeseries and envelope)')
    args = parser.parse_args()
    
    # Resolve paths
    ROOT = repo_root()
    # Normalize config_path to Path object early
    cfg_path = Path(resolve_path(args.config)).resolve()
    
    # Infer epoch if not provided
    epoch = args.epoch
    if epoch is None:
        # Try to infer from data filename if provided
        if args.data:
            data_path_resolved = resolve_path(args.data)
            epoch = infer_epoch_from_filename(data_path_resolved)
            if epoch:
                print(f"[OK] Inferred epoch {epoch} from data filename")
        # Try to infer from config filename
        if epoch is None:
            try:
                from src.generate_demandpack import infer_epoch_from_config_name
                epoch = infer_epoch_from_config_name(cfg_path)
                print(f"[OK] Inferred epoch {epoch} from config filename")
            except (ValueError, AttributeError):
                print("[WARN] Could not infer epoch, using default 2020")
                epoch = 2020
    
    print(f"Repository root: {ROOT}")
    print(f"Config path: {cfg_path}")
    print(f"Epoch: {epoch}")
    
    if not cfg_path.exists():
        print(f"[ERROR] Config file not found: {cfg_path}")
        sys.exit(1)
    
    # Load config
    config = load_config(str(cfg_path))
    
    # Resolve output paths based on precedence rules
    if args.run_id:
        # Precedence 1: New output path system with --run-id
        from src.output_paths import resolve_run_paths
        
        # Determine output root
        if args.output_root:
            output_root = resolve_path(args.output_root)
        else:
            output_root = ROOT / 'Output'
        
        output_paths = resolve_run_paths(
            output_root=output_root,
            epoch=epoch,
            config_path=cfg_path,
            run_id=args.run_id
        )
        
        # Determine data path
        if args.data:
            data_path_resolved = resolve_path(args.data)
        else:
            data_path_resolved = output_paths['run_demandpack_dir'] / f'hourly_heat_demand_{epoch}.csv'
        
        # Figures directory
        run_figures_dir = output_paths['run_figures_dir']
        
    elif args.output_dir:
        # Precedence 2: Legacy --output-dir
        run_figures_dir = resolve_path(args.output_dir)
        if not run_figures_dir.exists():
            run_figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine data path (must be provided)
        if args.data:
            data_path_resolved = resolve_path(args.data)
        else:
            print(f"[ERROR] --data must be provided when using --output-dir (legacy mode)")
            print(f"  Example: --data Output/runs/<run_id>/demandpack/hourly_heat_demand_{epoch}.csv")
            sys.exit(1)
        
        output_paths = None  # Not using new system
        
    else:
        # No run-id, no output-dir: error
        print(f"[ERROR] Must provide either --run-id or --output-dir")
        print(f"  Use --run-id for new output system")
        print(f"  Use --output-dir for legacy mode (also requires --data)")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from {data_path_resolved}...")
    if not data_path_resolved.exists():
        print(f"[ERROR] Data file not found: {data_path_resolved}")
        print(f"  Resolved from: {args.data if args.data else 'auto-resolved based on precedence rules'}")
        if args.run_id:
            print(f"  Expected location with --run-id: {output_paths['run_demandpack_dir'] / f'hourly_heat_demand_{epoch}.csv'}")
        print(f"  Please provide --data with explicit path, or ensure demandpack CSV exists in expected location")
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
    
    print(f"Generating all diagnostic plots for epoch {epoch}...")
    
    # Track generated files
    generated_files = []
    
    # Helper to write figures
    def write_figure(filename: str, plot_func, *plot_args, **plot_kwargs):
        """
        Write figure to run_figures_dir.
        
        Args:
            filename: Output filename (e.g., 'heat_2020_timeseries.png')
            plot_func: Plot function to call
            *plot_args: Positional arguments to pass to plot_func (before output_path)
            **plot_kwargs: Keyword arguments to pass to plot_func
        """
        # Ensure filename has valid image extension
        valid_extensions = {'.png', '.pdf', '.svg', '.jpg', '.jpeg'}
        file_ext = Path(filename).suffix.lower()
        if file_ext not in valid_extensions:
            raise ValueError(
                f"Invalid figure filename extension: {file_ext}. "
                f"Must be one of {valid_extensions}. Got: {filename}"
            )
        
        # Build output path
        output_path = run_figures_dir / filename
        
        # Ensure directory exists
        run_figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Guard: ensure we're writing to a figures directory, not a config path
        assert 'figures' in str(output_path) or 'Figures' in str(output_path), (
            f"write_figure: output path must be in a figures directory. "
            f"Got: {output_path}"
        )
        
        # Write figure
        # Most plot functions have signature: (df, output_path, epoch=...)
        # So we insert output_path after plot_args: plot_func(df, output_path, epoch=epoch)
        # For plot_hourly_means_by_season wrapper: (df, config, output_path, config_path, epoch=...)
        # We pass (df, config, cfg_path) and insert output_path: wrapper(df, config, output_path, cfg_path, epoch=epoch)
        # This correctly calls: plot_hourly_means_by_season(df, config, output_path, cfg_path, epoch=epoch)
        plot_func(*plot_args, str(output_path), **plot_kwargs)
        generated_files.append(str(output_path))
    
    # 1. Annual hourly time series
    write_figure(f'heat_{epoch}_timeseries.png', plot_hourly_timeseries_core, df, epoch=epoch)
    
    # 2. Daily envelope
    write_figure(f'heat_{epoch}_daily_envelope.png', plot_daily_envelope, df, epoch=epoch)
    
    # 3. Average hourly profile by season (with error handling)
    # Note: plot_hourly_means_by_season signature is: (df, config, output_path, config_path, epoch=...)
    # write_figure calls: plot_func(*plot_args, str(run_path), **plot_kwargs)
    # So if we pass plot_args=(df, config, cfg_path), it becomes: plot_func(df, config, cfg_path, run_path, epoch=epoch)
    # But we need: plot_hourly_means_by_season(df, config, run_path, cfg_path, epoch=epoch)
    # Solution: Create a wrapper that reorders arguments
    try:
        # Wrapper: accepts (df, config, cfg_path, output_path) and reorders to (df, config, output_path, cfg_path)
        def plot_hourly_means_wrapper(df, config, config_path_arg, output_path, epoch=None):
            """Wrapper that reorders arguments: write_figure passes (df, config, cfg_path, run_path) -> (df, config, run_path, cfg_path)"""
            return plot_hourly_means_by_season(df, config, output_path, config_path_arg, epoch=epoch)
        
        # Pass: df, config, cfg_path
        # write_figure will call: wrapper(df, config, cfg_path, run_path, epoch=epoch)
        # Wrapper reorders to: plot_hourly_means_by_season(df, config, run_path, cfg_path, epoch=epoch) ✓
        write_figure(f'heat_{epoch}_hourly_means_by_season.png', plot_hourly_means_wrapper, 
                     df, config, cfg_path, epoch=epoch)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"[WARN] Skipping hourly_means_by_season plot: {e}")
        print("  Continuing with remaining plots...")
    
    # 4. Monthly totals
    write_figure(f'heat_{epoch}_monthly_totals.png', plot_monthly_totals, df, epoch=epoch)
    
    # 5. Load duration curve
    write_figure(f'heat_{epoch}_LDC.png', plot_load_duration_curve, df, epoch=epoch)
    
    # 6. Weekday profiles for February (peak season)
    write_figure(f'heat_{epoch}_weekday_profiles_Feb.png', plot_weekday_profiles_for_month, 
                 df, 2, epoch=epoch)
    
    # 7. Weekday profiles for June (low season)
    write_figure(f'heat_{epoch}_weekday_profiles_Jun.png', plot_weekday_profiles_for_month, 
                 df, 6, epoch=epoch)
    
    # 8. Load histogram
    write_figure(f'heat_{epoch}_load_histogram.png', plot_load_histogram, df, epoch=epoch)
    
    print("\n" + "="*60)
    print("All diagnostic plots generated successfully!")
    print("="*60)
    print(f"\nGenerated figures: {run_figures_dir}")
    print(f"\nTotal figures: {len(generated_files)}")
    print("="*60)


if __name__ == '__main__':
    # Default behavior: simple plotting mode (hourly timeseries and daily envelope)
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Plot DemandPack hourly timeseries and daily envelope',
        epilog="""
Precedence rules for output paths:
  1. If --run-id is provided: Uses new output path system
  2. Else if --output-dir is provided (legacy): Uses --output-dir
  3. Else: Defaults to Output/latest/figures
        """
    )
    parser.add_argument('--data', default=None,
                       help='Path to generated demand CSV (if not provided, auto-resolved)')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Epoch year (inferred from data filename if not provided)')
    parser.add_argument('--full-diagnostics', action='store_true',
                       help='Generate all diagnostic plots (default: just timeseries and envelope)')
    parser.add_argument('--config', default='Input/demandpack_config.toml',
                       help='Path to config TOML file (for full diagnostics)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (legacy: use --output-root and --run-id for new system)')
    parser.add_argument('--output-root', type=str, default=None,
                       help='Output root directory (default: repo_root/Output)')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run ID (if provided, uses new output path system)')
    args = parser.parse_args()
    
    if args.full_diagnostics:
        # Run full diagnostic suite - call main() directly
        main()
    else:
        # Simple mode: load data and generate two plots
        ROOT = repo_root()
        
        # Resolve data path
        if args.data:
            data_path_resolved = resolve_path(args.data)
        else:
            # Auto-resolve based on precedence
            if args.run_id:
                from src.output_paths import resolve_run_paths
                if args.output_root:
                    output_root = resolve_path(args.output_root)
                else:
                    output_root = ROOT / 'Output'
                output_paths = resolve_run_paths(
                    output_root=output_root,
                    epoch=args.epoch or 2020,
                    config_path=None,
                    run_id=args.run_id
                )
                data_path_resolved = output_paths['run_demandpack_dir'] / f'hourly_heat_demand_{args.epoch or 2020}.csv'
            else:
                output_root = ROOT / 'Output'
                latest_demandpack_dir = output_root / 'latest' / 'demandpack'
                data_path_resolved = latest_demandpack_dir / f'hourly_heat_demand_{args.epoch or 2020}.csv'
        
        # Infer epoch if not provided
        epoch = args.epoch
        if epoch is None:
            epoch = infer_epoch_from_filename(data_path_resolved)
            if epoch:
                print(f"[OK] Inferred epoch {epoch} from data filename")
            else:
                epoch = 2020  # Default fallback
        
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
        
        # Determine output directory based on precedence
        if args.run_id:
            from src.output_paths import resolve_run_paths
            if args.output_root:
                output_root = resolve_path(args.output_root)
            else:
                output_root = ROOT / 'Output'
            output_paths = resolve_run_paths(
                output_root=output_root,
                epoch=epoch,
                config_path=None,
                run_id=args.run_id
            )
            figures_dir = output_paths['run_figures_dir']
        elif args.output_dir:
            figures_dir = resolve_path(args.output_dir)
            figures_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[ERROR] Must provide either --run-id or --output-dir")
            sys.exit(1)
        
        # Generate the two requested plots
        print(f"Generating plots for epoch {epoch}...")
        output_path = str(figures_dir / f"heat_{epoch}_timeseries.png")
        plot_hourly_timeseries_core(df, output_path, epoch)
        
        output_path = str(figures_dir / f"heat_{epoch}_daily_envelope.png")
        plot_daily_envelope(df, output_path, epoch)
        
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

