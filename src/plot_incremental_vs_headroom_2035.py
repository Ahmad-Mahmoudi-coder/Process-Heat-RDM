"""
Additive plotting script for comparing incremental electricity vs headroom for 2035_EB and 2035_BB.

Generates thesis-ready comparison figure showing headroom stress differences between pathways.
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
import matplotlib.dates as mdates
from typing import Dict, Optional, Tuple
import warnings

from src.path_utils import repo_root, canonical_output_path
from src.time_utils import parse_any_timestamp
from src.gxp_rdm_screen import load_or_generate_headroom


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


def find_incremental_csv(bundle_dir: Path, epoch_tag: str, run_id: str, output_root: str = "Output") -> Path:
    """
    Find incremental electricity CSV for an epoch.
    
    Args:
        bundle_dir: Bundle directory
        epoch_tag: Epoch tag (e.g., '2035_EB')
        run_id: Run ID (e.g., 'dispatch_prop_v2_capfix1')
        output_root: Output root directory name (default: "Output")
        
    Returns:
        Path to incremental CSV
        
    Raises:
        FileNotFoundError: If file not found
    """
    bundle_name = bundle_dir.name
    
    # Canonical path: epoch<epoch_tag>/dispatch/<run_id>/signals/incremental_electricity_MW_<epoch_tag>.csv
    canonical_path = canonical_output_path(
        bundle=bundle_name,
        epoch_tag=epoch_tag,
        layer="dispatch",
        runid=run_id,
        filename=f"signals/incremental_electricity_MW_{epoch_tag}.csv",
        output_root=output_root
    )
    
    if canonical_path.exists():
        return canonical_path
    
    # Legacy path: epoch<epoch_tag>/<run_id>/signals/incremental_electricity_MW_<epoch_tag>.csv
    legacy_path = bundle_dir / f'epoch{epoch_tag}' / run_id / 'signals' / f'incremental_electricity_MW_{epoch_tag}.csv'
    
    if legacy_path.exists():
        return legacy_path
    
    # Neither exists - raise error
    raise FileNotFoundError(
        f"Incremental electricity CSV not found for {epoch_tag}.\n"
        f"Attempted canonical path: {canonical_path}\n"
        f"Attempted legacy path: {legacy_path}"
    )


def find_headroom_csv(bundle_dir: Path) -> Optional[Path]:
    """
    Find headroom CSV in bundle rdm folder.
    
    Checks for any CSV matching *headroom*.csv in Output/runs/<bundle>/rdm/
    
    Args:
        bundle_dir: Bundle directory
        
    Returns:
        Path to headroom CSV if found, None otherwise
    """
    rdm_dir = bundle_dir / 'rdm'
    if not rdm_dir.exists():
        return None
    
    # Search for headroom CSV
    for csv_path in rdm_dir.glob("*headroom*.csv"):
        if csv_path.is_file():
            return csv_path
    
    return None


def load_incremental_electricity(csv_path: Path, epoch_tag: str) -> pd.DataFrame:
    """
    Load incremental electricity CSV with tolerant column mapping.
    
    Args:
        csv_path: Path to CSV file
        epoch_tag: Epoch tag for error messages
        
    Returns:
        DataFrame with columns: timestamp_utc, incremental_electricity_MW
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
    if 'incremental_electricity_MW' in df.columns:
        inc_col = 'incremental_electricity_MW'
    else:
        # Try epoch-specific name
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
        # If only one numeric column besides timestamp, use it
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 1:
            inc_col = numeric_cols[0]
        else:
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
    
    return result


def find_stress_window(
    timestamps: pd.DatetimeIndex,
    exceedance_eb: pd.Series,
    exceedance_bb: pd.Series,
    window_days: int = 14
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Find the contiguous window with maximum total exceedance.
    
    Computes hourly exceedance for both EB and BB, then finds the 14-day window
    with the maximum total exceedance (sum of max(EB, BB) exceedance per hour).
    
    Args:
        timestamps: DatetimeIndex for all timestamps
        exceedance_eb: Series of exceedance values for EB (ΔE_t - headroom where positive)
        exceedance_bb: Series of exceedance values for BB
        window_days: Window size in days (default: 14)
        
    Returns:
        Tuple of (start_timestamp, end_timestamp) for the selected window
    """
    # Compute combined exceedance (max of EB and BB per hour)
    combined_exceedance = np.maximum(exceedance_eb.values, exceedance_bb.values)
    
    # Window size in hours
    window_hours = window_days * 24
    
    if len(combined_exceedance) < window_hours:
        # If data is shorter than window, return full range
        return timestamps[0], timestamps[-1]
    
    # Find window with maximum total exceedance
    max_total = -np.inf
    best_start_idx = 0
    
    for start_idx in range(len(combined_exceedance) - window_hours + 1):
        window_total = combined_exceedance[start_idx:start_idx + window_hours].sum()
        if window_total > max_total:
            max_total = window_total
            best_start_idx = start_idx
    
    start_ts = timestamps[best_start_idx]
    end_ts = timestamps[best_start_idx + window_hours - 1]
    
    return start_ts, end_ts


def plot_comparison(
    timestamps: pd.DatetimeIndex,
    headroom: pd.Series,
    incremental_eb: pd.Series,
    incremental_bb: pd.Series,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    output_path: Path
) -> None:
    """
    Plot incremental electricity vs headroom comparison for 2035_EB and 2035_BB.
    
    Args:
        timestamps: Full DatetimeIndex
        headroom: Headroom series (MW)
        incremental_eb: Incremental electricity for EB (MW)
        incremental_bb: Incremental electricity for BB (MW)
        window_start: Start timestamp for zoom window
        window_end: End timestamp for zoom window
        output_path: Base path for output (will add .png and .pdf extensions)
    """
    # Filter to window
    mask = (timestamps >= window_start) & (timestamps <= window_end)
    ts_window = timestamps[mask]
    hr_window = headroom[mask]
    inc_eb_window = incremental_eb[mask]
    inc_bb_window = incremental_bb[mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot headroom line
    ax.plot(
        ts_window,
        hr_window,
        label='GXP Headroom',
        color='#2ca02c',  # green
        linewidth=2.0,
        alpha=0.8,
        linestyle='--'
    )
    
    # Plot incremental electricity lines
    ax.plot(
        ts_window,
        inc_eb_window,
        label='ΔE_t (2035_EB)',
        color='#d62728',  # red
        linewidth=1.5,
        alpha=0.8
    )
    
    ax.plot(
        ts_window,
        inc_bb_window,
        label='ΔE_t (2035_BB)',
        color='#9467bd',  # purple
        linewidth=1.5,
        alpha=0.8
    )
    
    # Add zero line (optional, but helpful)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3, linestyle=':')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Power (MW)', fontsize=11)
    ax.set_title(
        f'Incremental Electricity vs Headroom: 2035_EB vs 2035_BB',
        fontsize=12
    )
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Save both PNG and PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')
    
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Comparison figure: {png_path}")
    print(f"  [OK] Comparison figure: {pdf_path}")


def compute_exceedance_stats(
    timestamps: pd.DatetimeIndex,
    headroom: pd.Series,
    incremental_eb: pd.Series,
    incremental_bb: pd.Series
) -> Dict[str, float]:
    """
    Compute exceedance statistics (console output only).
    
    Args:
        timestamps: DatetimeIndex
        headroom: Headroom series (MW)
        incremental_eb: Incremental electricity for EB (MW)
        incremental_bb: Incremental electricity for BB (MW)
        
    Returns:
        Dictionary with statistics
    """
    exceedance_eb = np.maximum(incremental_eb - headroom, 0.0)
    exceedance_bb = np.maximum(incremental_bb - headroom, 0.0)
    
    # Total exceedance hours
    hours_exceed_eb = (exceedance_eb > 0.1).sum()  # > 0.1 MW threshold
    hours_exceed_bb = (exceedance_bb > 0.1).sum()
    
    # Max exceedance MW
    max_exceed_eb = exceedance_eb.max()
    max_exceed_bb = exceedance_bb.max()
    
    # Total shed MWh proxy (sum of exceedance * 1h)
    total_shed_eb = exceedance_eb.sum()  # MWh (MW * hours)
    total_shed_bb = exceedance_bb.sum()
    
    return {
        'hours_exceed_eb': hours_exceed_eb,
        'hours_exceed_bb': hours_exceed_bb,
        'max_exceed_eb': max_exceed_eb,
        'max_exceed_bb': max_exceed_bb,
        'total_shed_eb': total_shed_eb,
        'total_shed_bb': total_shed_bb,
    }


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready comparison figure: incremental electricity vs headroom for 2035_EB and 2035_BB"
    )
    parser.add_argument(
        '--bundle',
        type=str,
        required=True,
        help='Bundle name (e.g., poc_20260105_release02)'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        required=True,
        help='Run ID (e.g., dispatch_prop_v2_capfix1)'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='Output',
        help='Output root directory (default: Output)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    root = repo_root()
    bundle_dir = root / args.output_root / 'runs' / args.bundle
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    
    # Find incremental electricity CSVs
    print("[INPUTS] Resolving incremental electricity files...")
    inc_eb_path = find_incremental_csv(bundle_dir, '2035_EB', args.run_id, args.output_root)
    inc_bb_path = find_incremental_csv(bundle_dir, '2035_BB', args.run_id, args.output_root)
    
    print(f"  2035_EB: {inc_eb_path}")
    print(f"  2035_BB: {inc_bb_path}")
    
    # Load incremental electricity
    print("\n[LOADING DATA]")
    df_eb = load_incremental_electricity(inc_eb_path, '2035_EB')
    df_bb = load_incremental_electricity(inc_bb_path, '2035_BB')
    
    print(f"  [OK] 2035_EB: {len(df_eb)} rows")
    print(f"  [OK] 2035_BB: {len(df_bb)} rows")
    
    # Parse timestamps
    timestamps_eb = pd.DatetimeIndex(parse_any_timestamp(df_eb['timestamp_utc']))
    timestamps_bb = pd.DatetimeIndex(parse_any_timestamp(df_bb['timestamp_utc']))
    
    # Use EB timestamps as reference (or find common timestamps)
    # For simplicity, use EB timestamps and align BB to them
    timestamps = timestamps_eb
    
    # Create series indexed by timestamp
    incremental_eb = pd.Series(
        df_eb['incremental_electricity_MW'].values,
        index=timestamps_eb,
        name='incremental_eb'
    )
    incremental_bb = pd.Series(
        df_bb['incremental_electricity_MW'].values,
        index=timestamps_bb,
        name='incremental_bb'
    )
    
    # Reindex BB to EB timestamps (nearest neighbor if needed)
    incremental_bb_aligned = incremental_bb.reindex(timestamps, method='nearest').fillna(0.0)
    
    # Find headroom
    print("\n[HEADROOM]")
    headroom_csv = find_headroom_csv(bundle_dir)
    
    if headroom_csv and headroom_csv.exists():
        print(f"  [OK] Source: A (bundle rdm folder)")
        print(f"  [OK] Using headroom from: {headroom_csv}")
        headroom_source = "A"
        headroom_series = load_or_generate_headroom(timestamps, headroom_csv)
    else:
        print(f"  [OK] Source: B (generated deterministic PoC headroom)")
        print(f"  [OK] Headroom CSV not found in bundle rdm folder, generating deterministic PoC headroom")
        headroom_source = "B"
        headroom_series = load_or_generate_headroom(timestamps, None)
    
    # Align headroom to timestamps
    headroom_aligned = headroom_series.reindex(timestamps, method='nearest').fillna(0.0)
    
    # Compute exceedance for window selection
    exceedance_eb = np.maximum(incremental_eb - headroom_aligned, 0.0)
    exceedance_bb = np.maximum(incremental_bb_aligned - headroom_aligned, 0.0)
    
    # Find stress window
    print("\n[STRESS WINDOW SELECTION]")
    window_start, window_end = find_stress_window(
        timestamps,
        pd.Series(exceedance_eb, index=timestamps),
        pd.Series(exceedance_bb, index=timestamps),
        window_days=14
    )
    print(f"  Selected 14-day window: {window_start.strftime('%Y-%m-%d %H:%M')} to {window_end.strftime('%Y-%m-%d %H:%M')}")
    
    # Compute exceedance statistics (console only)
    print("\n[EXCEEDANCE STATISTICS]")
    stats = compute_exceedance_stats(
        timestamps,
        headroom_aligned,
        incremental_eb,
        incremental_bb_aligned
    )
    print(f"  2035_EB: {stats['hours_exceed_eb']} exceedance hours, "
          f"max={stats['max_exceed_eb']:.2f} MW, "
          f"total_shed={stats['total_shed_eb']:.2f} MWh")
    print(f"  2035_BB: {stats['hours_exceed_bb']} exceedance hours, "
          f"max={stats['max_exceed_bb']:.2f} MW, "
          f"total_shed={stats['total_shed_bb']:.2f} MWh")
    
    # Generate figure
    print("\n[GENERATING FIGURE]")
    output_dir = bundle_dir / 'thesis_pack' / 'comparison'
    output_path = output_dir / 'incremental_electricity_vs_headroom_2035_EB_vs_BB'
    
    plot_comparison(
        timestamps,
        headroom_aligned,
        incremental_eb,
        incremental_bb_aligned,
        window_start,
        window_end,
        output_path
    )
    
    # Print outputs
    print("\n[OUTPUTS WRITTEN]")
    print(f"  Output directory: {output_dir}")
    print(f"  Comparison figure: {output_path.with_suffix('.png')}")
    print(f"  Comparison figure: {output_path.with_suffix('.pdf')}")
    print(f"\n[DONE]")


if __name__ == '__main__':
    main()
