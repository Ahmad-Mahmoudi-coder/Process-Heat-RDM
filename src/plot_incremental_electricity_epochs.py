"""
Additive plotting script for cross-epoch comparison of incremental electricity (ΔE_t).

Generates thesis-ready duration curves comparing incremental electricity across epochs.
Read-only: does not modify any existing outputs.
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
from typing import Dict, List, Optional, Tuple
import warnings

from src.path_utils import repo_root
from src.time_utils import parse_any_timestamp


# Matplotlib style settings for thesis-ready figures
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


# Required epochs (must be present)
REQUIRED_EPOCHS = ['2025', '2028', '2035_EB']
# Optional epochs (include if present)
OPTIONAL_EPOCHS = ['2035_BB', '2020']


def discover_incremental_electricity_files(
    bundle_dir: Path,
    output_root: str = "Output"
) -> Dict[str, Path]:
    """
    Discover incremental electricity CSV files by searching the bundle directory.
    
    Searches: Output/runs/<bundle>/**/signals/incremental_electricity_MW_*.csv
    Parses epoch_tag from filename suffix.
    
    Args:
        bundle_dir: Path to bundle directory (e.g., Output/runs/poc_20260105_release02)
        output_root: Output root directory name (default: "Output")
        
    Returns:
        Dictionary mapping epoch_tag -> resolved Path to CSV file
    """
    found_files = {}
    
    # Search pattern: **/signals/incremental_electricity_MW_*.csv
    signals_pattern = bundle_dir / "**" / "signals" / "incremental_electricity_MW_*.csv"
    
    for csv_path in bundle_dir.glob("**/signals/incremental_electricity_MW_*.csv"):
        # Extract epoch_tag from filename
        # Format: incremental_electricity_MW_<epoch_tag>.csv
        filename = csv_path.name
        if filename.startswith("incremental_electricity_MW_") and filename.endswith(".csv"):
            epoch_tag = filename[len("incremental_electricity_MW_"):-len(".csv")]
            found_files[epoch_tag] = csv_path.resolve()
    
    return found_files


def load_incremental_electricity(csv_path: Path, epoch_tag: str) -> pd.DataFrame:
    """
    Load incremental electricity CSV with robust column name handling.
    
    Args:
        csv_path: Path to CSV file
        epoch_tag: Epoch tag for error messages
        
    Returns:
        DataFrame with columns: timestamp_utc, incremental_electricity_MW
        
    Raises:
        ValueError: If required columns are missing
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Incremental electricity CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Find timestamp column
    ts_col = None
    for candidate in ['timestamp_utc', 'timestamp']:
        if candidate in df.columns:
            ts_col = candidate
            break
    
    if ts_col is None:
        raise ValueError(
            f"No timestamp column found in {csv_path} for epoch {epoch_tag}. "
            f"Expected 'timestamp_utc' or 'timestamp'"
        )
    
    # Find incremental column
    inc_col = None
    # Try standard name first
    if 'incremental_electricity_MW' in df.columns:
        inc_col = 'incremental_electricity_MW'
    else:
        # Try epoch-specific name: incremental_electricity_MW_<epoch_tag>
        epoch_specific = f'incremental_electricity_MW_{epoch_tag}'
        if epoch_specific in df.columns:
            inc_col = epoch_specific
        else:
            # Try alternative names
            for candidate in ['incremental_MW', 'incremental_electricity']:
                if candidate in df.columns:
                    inc_col = candidate
                    break
    
    if inc_col is None:
        raise ValueError(
            f"No incremental electricity column found in {csv_path} for epoch {epoch_tag}"
        )
    
    # Parse timestamps
    df[ts_col] = parse_any_timestamp(df[ts_col])
    
    # Select and rename columns
    result = df[[ts_col, inc_col]].copy()
    result.columns = ['timestamp_utc', 'incremental_electricity_MW']
    
    # Ensure numeric
    result['incremental_electricity_MW'] = pd.to_numeric(
        result['incremental_electricity_MW'], 
        errors='coerce'
    ).fillna(0.0)
    
    # Sort by timestamp (in case not monotonic)
    result = result.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Validate row count (prefer 8760, but warn if different)
    if len(result) != 8760:
        warnings.warn(
            f"Epoch {epoch_tag}: Expected 8760 rows, found {len(result)}. "
            f"Proceeding with available data."
        )
    
    return result


def compute_duration_curve(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute duration curve (sorted descending values vs exceedance percentile).
    
    Args:
        values: Series of values to sort
        
    Returns:
        Tuple of (sorted_values_descending, exceedance_percentile)
        where exceedance_percentile is 0-100 (percent)
    """
    sorted_vals = np.sort(values.values)[::-1]  # Descending
    n = len(sorted_vals)
    exceedance_pct = np.linspace(0, 100, n)
    return sorted_vals, exceedance_pct


def compute_summary_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics for incremental electricity.
    
    Args:
        df: DataFrame with incremental_electricity_MW column
        
    Returns:
        Dictionary with summary statistics
    """
    values = df['incremental_electricity_MW']
    
    # Annual energy (sum MW * 1h / 1000 = GWh)
    annual_energy_GWh = values.sum() / 1000.0
    
    # Peak
    peak_MW = values.max()
    
    # Percentiles
    p95_MW = values.quantile(0.95)
    p99_MW = values.quantile(0.99)
    
    # Hours above thresholds
    hours_gt_0_1MW = (values > 0.1).sum()
    hours_gt_5MW = (values > 5.0).sum()
    hours_gt_10MW = (values > 10.0).sum()
    hours_gt_20MW = (values > 20.0).sum()
    
    return {
        'annual_energy_GWh': annual_energy_GWh,
        'peak_MW': peak_MW,
        'p95_MW': p95_MW,
        'p99_MW': p99_MW,
        'hours_gt_0.1MW': hours_gt_0_1MW,
        'hours_gt_5MW': hours_gt_5MW,
        'hours_gt_10MW': hours_gt_10MW,
        'hours_gt_20MW': hours_gt_20MW,
    }


def plot_duration_curves(
    data_dict: Dict[str, pd.DataFrame],
    output_path: Path
) -> None:
    """
    Plot duration curves for incremental electricity across epochs.
    
    Args:
        data_dict: Dictionary mapping epoch_tag -> DataFrame
        output_path: Base path for output (will add .png and .pdf extensions)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color palette for epochs
    colors = {
        '2020': '#1f77b4',      # blue
        '2025': '#ff7f0e',      # orange
        '2028': '#2ca02c',      # green
        '2035_EB': '#d62728',   # red
        '2035_BB': '#9467bd',   # purple
    }
    
    # Plot each epoch
    for epoch_tag, df in sorted(data_dict.items()):
        values = df['incremental_electricity_MW']
        sorted_vals, exceedance_pct = compute_duration_curve(values)
        
        color = colors.get(epoch_tag, '#7f7f7f')  # gray fallback
        ax.plot(
            exceedance_pct,
            sorted_vals,
            label=epoch_tag,
            color=color,
            linewidth=1.5,
            alpha=0.8
        )
    
    ax.set_xlabel('Exceedance Percentile (%)', fontsize=11)
    ax.set_ylabel('Incremental Electricity (MW)', fontsize=11)
    ax.set_title('Duration Curves: Incremental Electricity Across Epochs', fontsize=12)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    # Save both PNG and PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')
    
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Duration curve: {png_path}")
    print(f"  [OK] Duration curve: {pdf_path}")


def plot_summary_bars(
    summary_dict: Dict[str, Dict[str, float]],
    output_path: Path
) -> None:
    """
    Plot summary bar chart comparing annual energy, peak, and p95 across epochs.
    
    Args:
        summary_dict: Dictionary mapping epoch_tag -> summary stats dict
        output_path: Base path for output (will add .png and .pdf extensions)
    """
    epochs = sorted(summary_dict.keys())
    
    # Extract metrics
    annual_energy = [summary_dict[e]['annual_energy_GWh'] for e in epochs]
    peak = [summary_dict[e]['peak_MW'] for e in epochs]
    p95 = [summary_dict[e]['p95_MW'] for e in epochs]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    x_pos = np.arange(len(epochs))
    width = 0.6
    
    # Color palette
    colors = {
        '2020': '#1f77b4',
        '2025': '#ff7f0e',
        '2028': '#2ca02c',
        '2035_EB': '#d62728',
        '2035_BB': '#9467bd',
    }
    bar_colors = [colors.get(e, '#7f7f7f') for e in epochs]
    
    # Annual energy (GWh)
    axes[0].bar(x_pos, annual_energy, width, color=bar_colors, alpha=0.8)
    axes[0].set_ylabel('Annual Energy (GWh)', fontsize=10)
    axes[0].set_title('Annual Energy', fontsize=11)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(epochs, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Peak (MW)
    axes[1].bar(x_pos, peak, width, color=bar_colors, alpha=0.8)
    axes[1].set_ylabel('Peak (MW)', fontsize=10)
    axes[1].set_title('Peak Demand', fontsize=11)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(epochs, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # P95 (MW)
    axes[2].bar(x_pos, p95, width, color=bar_colors, alpha=0.8)
    axes[2].set_ylabel('P95 (MW)', fontsize=10)
    axes[2].set_title('95th Percentile', fontsize=11)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(epochs, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save both PNG and PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')
    
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Summary bars: {png_path}")
    print(f"  [OK] Summary bars: {pdf_path}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready cross-epoch comparison figures for incremental electricity"
    )
    parser.add_argument(
        '--bundle',
        type=str,
        required=True,
        help='Bundle name (e.g., poc_20260105_release02)'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='Output',
        help='Output root directory (default: Output)'
    )
    parser.add_argument(
        '--outdir-rel',
        type=str,
        default='thesis_pack/comparison',
        help='Output directory relative to bundle (default: thesis_pack/comparison)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    root = repo_root()
    bundle_dir = root / args.output_root / 'runs' / args.bundle
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    
    # Discover incremental electricity files
    print(f"[DISCOVERY] Searching for incremental electricity files in {bundle_dir}...")
    found_files = discover_incremental_electricity_files(bundle_dir, args.output_root)
    
    if not found_files:
        raise FileNotFoundError(
            f"No incremental electricity files found in {bundle_dir}. "
            f"Expected pattern: **/signals/incremental_electricity_MW_*.csv"
        )
    
    # Filter to relevant epochs
    relevant_epochs = set(REQUIRED_EPOCHS + OPTIONAL_EPOCHS)
    selected_files = {
        epoch: path
        for epoch, path in found_files.items()
        if epoch in relevant_epochs
    }
    
    # Check for required epochs
    missing_required = set(REQUIRED_EPOCHS) - set(selected_files.keys())
    if missing_required:
        warnings.warn(
            f"Missing required epochs: {sorted(missing_required)}. "
            f"Proceeding with available epochs: {sorted(selected_files.keys())}"
        )
    
    if not selected_files:
        raise ValueError(
            f"No relevant epochs found. Found: {sorted(found_files.keys())}, "
            f"Expected at least one of: {sorted(REQUIRED_EPOCHS + OPTIONAL_EPOCHS)}"
        )
    
    # Print inputs found
    print("\n[INPUTS FOUND]")
    for epoch_tag in sorted(selected_files.keys()):
        print(f"  {epoch_tag:12s} -> {selected_files[epoch_tag]}")
    
    # Load data
    print("\n[LOADING DATA]")
    data_dict = {}
    for epoch_tag, csv_path in sorted(selected_files.items()):
        try:
            df = load_incremental_electricity(csv_path, epoch_tag)
            data_dict[epoch_tag] = df
            print(f"  [OK] {epoch_tag}: {len(df)} rows loaded")
        except Exception as e:
            warnings.warn(f"Failed to load {epoch_tag} from {csv_path}: {e}")
            continue
    
    if not data_dict:
        raise ValueError("No data successfully loaded")
    
    # Compute summary statistics
    print("\n[COMPUTING STATISTICS]")
    summary_dict = {}
    for epoch_tag, df in sorted(data_dict.items()):
        stats = compute_summary_stats(df)
        summary_dict[epoch_tag] = stats
        print(f"  {epoch_tag}: {stats['annual_energy_GWh']:.2f} GWh, "
              f"peak={stats['peak_MW']:.2f} MW, p95={stats['p95_MW']:.2f} MW")
    
    # Prepare output directory
    output_dir = bundle_dir / args.outdir_rel
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    print("\n[GENERATING FIGURES]")
    
    # (1) Duration curves (PRIMARY, MANDATORY)
    duration_curve_path = output_dir / "incremental_electricity_duration_curve_epochs"
    plot_duration_curves(data_dict, duration_curve_path)
    
    # (2) Summary bars (OPTIONAL, if clean)
    summary_bars_path = output_dir / "incremental_electricity_epoch_summary_bars"
    try:
        plot_summary_bars(summary_dict, summary_bars_path)
    except Exception as e:
        warnings.warn(f"Skipping summary bars due to error: {e}")
    
    # (3) Optional summary CSV
    summary_csv_path = output_dir.parent / "incremental_electricity_epoch_summary.csv"
    if summary_csv_path.exists():
        # Avoid overwriting: use suffix
        summary_csv_path = output_dir.parent / "incremental_electricity_epoch_summary__new.csv"
    
    try:
        summary_rows = []
        for epoch_tag in sorted(summary_dict.keys()):
            row = {'epoch_tag': epoch_tag}
            row.update(summary_dict[epoch_tag])
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\n[OK] Summary CSV: {summary_csv_path}")
    except Exception as e:
        warnings.warn(f"Failed to write summary CSV: {e}")
    
    # Print outputs written
    print("\n[OUTPUTS WRITTEN]")
    print(f"  Output directory: {output_dir}")
    print(f"  Duration curve: {duration_curve_path.with_suffix('.png')}")
    print(f"  Duration curve: {duration_curve_path.with_suffix('.pdf')}")
    if summary_bars_path.with_suffix('.png').exists():
        print(f"  Summary bars: {summary_bars_path.with_suffix('.png')}")
        print(f"  Summary bars: {summary_bars_path.with_suffix('.pdf')}")
    if summary_csv_path.exists():
        print(f"  Summary CSV: {summary_csv_path}")
    
    print("\n[DONE]")


if __name__ == '__main__':
    main()
