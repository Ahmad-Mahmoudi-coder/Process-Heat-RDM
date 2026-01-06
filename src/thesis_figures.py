"""
Thesis-ready figure generation module.

Generates curated figures for thesis from existing outputs (no re-solving, no coupling).
Reads dispatch outputs, RDM tables, regional signals, and DemandPack CSVs.
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
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from src.path_utils import repo_root, resolve_path
from src.time_utils import parse_any_timestamp


# Matplotlib style settings for thesis-ready figures
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')  # Fallback to default style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_dispatch_wide(dispatch_wide_path: Path) -> pd.DataFrame:
    """Load dispatch wide-form CSV."""
    df = pd.read_csv(dispatch_wide_path)
    if 'timestamp_utc' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp_utc'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp'])
    else:
        raise ValueError("Dispatch wide CSV must have timestamp_utc or timestamp column")
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_regional_signal(csv_path: Path, value_col: str) -> pd.DataFrame:
    """Load regional signal CSV (headroom, tariff, etc.)."""
    df = pd.read_csv(csv_path)
    if 'timestamp_utc' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp_utc'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp'])
    else:
        raise ValueError(f"Signal CSV must have timestamp_utc or timestamp column: {csv_path}")
    
    if value_col not in df.columns:
        raise ValueError(f"Signal CSV must have {value_col} column: {csv_path}")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df[['timestamp', value_col]]


def load_rdm_matrix(csv_path: Path) -> pd.DataFrame:
    """Load RDM matrix CSV."""
    return pd.read_csv(csv_path)


def load_rdm_compare(csv_path: Path) -> pd.DataFrame:
    """Load RDM comparison CSV."""
    return pd.read_csv(csv_path)


def load_incremental_electricity(csv_path: Path) -> pd.DataFrame:
    """Load incremental electricity CSV."""
    df = pd.read_csv(csv_path)
    if 'timestamp_utc' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp_utc'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = parse_any_timestamp(df['timestamp'])
    else:
        raise ValueError("Incremental electricity CSV must have timestamp_utc or timestamp column")
    
    if 'incremental_electricity_MW' not in df.columns:
        raise ValueError("Incremental electricity CSV must have incremental_electricity_MW column")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


# ============================================================================
# 2020 Deep-Dive Figures (Appendix)
# ============================================================================

def plot_2020_timeseries_week(dispatch_wide_path: Path, output_path: Path) -> None:
    """Plot hourly heat demand/dispatch for one representative week (2020)."""
    df = load_dispatch_wide(dispatch_wide_path)
    
    # Select first week of February (or first available week)
    feb_mask = df['timestamp'].dt.month == 2
    if feb_mask.any():
        feb_start = df[feb_mask].index[0]
        feb_week = df.iloc[feb_start:feb_start+168]  # 168 hours = 7 days
    else:
        feb_week = df.head(168)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot total heat
    ax.plot(feb_week['timestamp'], feb_week['total_heat_MW'].values,
           linewidth=1.5, color='#2c3e50', label='Total Heat Demand')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Heat (MW)', fontsize=11)
    ax.set_title('Hourly Heat Demand - 2020 (Representative Week)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_2020_daily_envelope(dispatch_wide_path: Path, output_path: Path) -> None:
    """Plot daily p10/p50/p90 envelope (2020)."""
    df = load_dispatch_wide(dispatch_wide_path)
    
    # Group by day and compute percentiles
    df['date'] = df['timestamp'].dt.date
    daily_stats = df.groupby('date')['total_heat_MW'].agg([
        ('p10', lambda x: np.percentile(x, 10)),
        ('p50', lambda x: np.percentile(x, 50)),
        ('p90', lambda x: np.percentile(x, 90)),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot envelope
    ax.fill_between(daily_stats['date'], daily_stats['p10'], daily_stats['p90'],
                    alpha=0.3, color='#3498db', label='P10-P90 Range')
    ax.plot(daily_stats['date'], daily_stats['p50'], linewidth=1.5, color='#2c3e50',
           label='Median (P50)')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Heat (MW)', fontsize=11)
    ax.set_title('Daily Heat Demand Envelope - 2020 (P10/P50/P90)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_2020_load_histogram(dispatch_wide_path: Path, output_path: Path) -> None:
    """Plot hourly load histogram (2020)."""
    df = load_dispatch_wide(dispatch_wide_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of hourly loads
    ax.hist(df['total_heat_MW'].values, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
    
    ax.set_xlabel('Heat Demand (MW)', fontsize=11)
    ax.set_ylabel('Frequency (Hours)', fontsize=11)
    ax.set_title('Hourly Heat Demand Distribution - 2020', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_2020_weekday_profiles(dispatch_wide_path: Path, output_path: Path, month: int = 2) -> None:
    """Plot weekday diurnal profiles for a specific month (2020)."""
    df = load_dispatch_wide(dispatch_wide_path)
    
    # Filter to month and weekdays (Monday=0, Friday=4)
    # Use .copy() to avoid SettingWithCopyWarning
    month_df = df[df['timestamp'].dt.month == month].copy()
    weekday_df = month_df[month_df['timestamp'].dt.dayofweek < 5].copy()  # Mon-Fri
    
    # Group by hour
    weekday_df['hour'] = weekday_df['timestamp'].dt.hour
    hourly_profile = weekday_df.groupby('hour')['total_heat_MW'].agg([
        ('mean', 'mean'),
        ('p10', lambda x: np.percentile(x, 10)),
        ('p90', lambda x: np.percentile(x, 90))
    ]).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot profile with envelope
    ax.fill_between(hourly_profile['hour'], hourly_profile['p10'], hourly_profile['p90'],
                    alpha=0.3, color='#3498db', label='P10-P90 Range')
    ax.plot(hourly_profile['hour'], hourly_profile['mean'], linewidth=2, color='#2c3e50',
           marker='o', markersize=4, label='Mean')
    
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Heat Demand (MW)', fontsize=11)
    month_name = datetime(2020, month, 1).strftime('%B')
    ax.set_title(f'Weekday Diurnal Profile - 2020 ({month_name})', fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


# ============================================================================
# Per-Epoch Utilisation Figures (Curated)
# ============================================================================

def plot_units_online_prop(dispatch_wide_path: Path, output_path: Path, epoch_tag: str) -> None:
    """Plot proportion of units online per hour."""
    df = load_dispatch_wide(dispatch_wide_path)
    
    # Get unit columns (exclude total_heat_MW and unserved_MW)
    unit_cols = [col for col in df.columns if col.endswith('_MW') and 
                 col not in ['total_heat_MW', 'unserved_MW']]
    
    # Count units online (heat_MW > 0) per hour
    units_online = (df[unit_cols] > 0).sum(axis=1)
    total_units = len(unit_cols)
    prop_online = units_online / total_units if total_units > 0 else 0
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot time series
    ax.plot(df['timestamp'], prop_online.values, linewidth=1, color='#2c3e50', alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Proportion of Units Online', fontsize=11)
    ax.set_title(f'Units Online Proportion - {epoch_tag}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_unit_utilisation_duration_prop(dispatch_wide_path: Path, output_path: Path, 
                                        epoch_tag: str, util_df_path: Optional[Path] = None) -> None:
    """Plot duration curve of utilisation per unit (normalized by max capacity if available)."""
    df = load_dispatch_wide(dispatch_wide_path)
    
    # Get unit columns
    unit_cols = [col for col in df.columns if col.endswith('_MW') and 
                 col not in ['total_heat_MW', 'unserved_MW']]
    
    # Load unit capacities if available
    max_capacities = {}
    if util_df_path and util_df_path.exists():
        try:
            util_df = pd.read_csv(util_df_path)
            for unit_id in [col.replace('_MW', '') for col in unit_cols]:
                unit_row = util_df[util_df['unit_id'] == unit_id]
                if len(unit_row) > 0:
                    max_capacities[unit_id] = unit_row.iloc[0].get('max_heat_MW', None)
        except Exception as e:
            print(f"[WARN] Could not load unit capacities from {util_df_path}: {e}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot duration curve for each unit
    for col in unit_cols:
        unit_id = col.replace('_MW', '')
        unit_data = df[col].values
        
        # Normalize by capacity if available
        if unit_id in max_capacities and max_capacities[unit_id] is not None and max_capacities[unit_id] > 0:
            unit_data = unit_data / max_capacities[unit_id]
            ylabel = 'Utilisation (proportion of max capacity)'
        else:
            ylabel = 'Heat Output (MW)'
        
        # Sort descending for duration curve
        sorted_data = np.sort(unit_data)[::-1]
        hours = np.arange(1, len(sorted_data) + 1)
        hours_pct = (hours / len(sorted_data)) * 100
        
        ax.plot(hours_pct, sorted_data, linewidth=1.5, alpha=0.7, label=unit_id)
    
    ax.set_xlabel('Duration (% of hours)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'Unit Utilisation Duration Curve - {epoch_tag}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if len(unit_cols) <= 10:  # Only show legend if not too many units
        ax.legend(fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


# ============================================================================
# RDM Comparison Figures (Curated)
# ============================================================================

def plot_rdm_failure_rate_bar(rdm_compare_path: Path, output_path: Path) -> None:
    """Plot failure rate (shed > 0 or exceed > 0) comparison bar chart."""
    df = load_rdm_compare(rdm_compare_path)
    
    # Compute failure rates
    # Failure = shed > 0 OR hours_exceed > 0
    if 'annual_shed_MWh_EB' in df.columns and 'annual_shed_MWh_BB' in df.columns:
        fail_eb = ((df['annual_shed_MWh_EB'] > 1e-6) | 
                   (df.get('hours_exceed_EB', 0) > 0)).sum()
        fail_bb = ((df['annual_shed_MWh_BB'] > 1e-6) | 
                   (df.get('hours_exceed_BB', 0) > 0)).sum()
    else:
        # Fallback: use satisficing_pass if available
        if 'satisficing_pass_EB' in df.columns:
            fail_eb = (~df['satisficing_pass_EB']).sum()
            fail_bb = (~df['satisficing_pass_BB']).sum()
        else:
            print("[WARN] Cannot compute failure rate: missing columns")
            return
    
    total = len(df)
    fail_rate_eb = (fail_eb / total * 100) if total > 0 else 0
    fail_rate_bb = (fail_bb / total * 100) if total > 0 else 0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bar chart
    pathways = ['EB', 'BB']
    fail_rates = [fail_rate_eb, fail_rate_bb]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(pathways, fail_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, rate in zip(bars, fail_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Failure Rate (%)', fontsize=11)
    ax.set_title('RDM Failure Rate Comparison (2035)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(fail_rates) * 1.2 if max(fail_rates) > 0 else 10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_rdm_regret_cdf(rdm_compare_path: Path, output_path: Path, log_x: bool = False) -> None:
    """Plot CDF of regret (EB vs BB)."""
    df = load_rdm_compare(rdm_compare_path)
    
    # Get regret columns (prefer regret_vs_benchmark, fallback to total_cost difference)
    if 'regret_EB' in df.columns and 'regret_BB' in df.columns:
        regret_eb = df['regret_EB'].values
        regret_bb = df['regret_BB'].values
    elif 'regret_vs_benchmark_nzd_EB' in df.columns and 'regret_vs_benchmark_nzd_BB' in df.columns:
        regret_eb = df['regret_vs_benchmark_nzd_EB'].values
        regret_bb = df['regret_vs_benchmark_nzd_BB'].values
    elif 'total_cost_nzd_EB' in df.columns and 'total_cost_nzd_BB' in df.columns:
        # Use total cost as proxy for regret
        regret_eb = df['total_cost_nzd_EB'].values
        regret_bb = df['total_cost_nzd_BB'].values
    else:
        print("[WARN] Cannot plot regret CDF: missing columns")
        return
    
    # Remove NaN and infinite values
    regret_eb = regret_eb[np.isfinite(regret_eb)]
    regret_bb = regret_bb[np.isfinite(regret_bb)]
    
    if len(regret_eb) == 0 or len(regret_bb) == 0:
        print("[WARN] No valid regret data for CDF")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute CDF
    sorted_eb = np.sort(regret_eb)
    sorted_bb = np.sort(regret_bb)
    p_eb = np.arange(1, len(sorted_eb) + 1) / len(sorted_eb)
    p_bb = np.arange(1, len(sorted_bb) + 1) / len(sorted_bb)
    
    # Plot CDF
    ax.plot(sorted_eb, p_eb * 100, linewidth=2, color='#3498db', label='EB', marker='o', markersize=3)
    ax.plot(sorted_bb, p_bb * 100, linewidth=2, color='#e74c3c', label='BB', marker='s', markersize=3)
    
    if log_x:
        ax.set_xscale('log')
        ax.set_xlabel('Regret (NZD, log scale)', fontsize=11)
        title_suffix = ' (Log Scale)'
    else:
        ax.set_xlabel('Regret (NZD)', fontsize=11)
        title_suffix = ''
    
    ax.set_ylabel('Cumulative Probability (%)', fontsize=11)
    ax.set_title(f'RDM Regret CDF Comparison (2035){title_suffix}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_rdm_total_cost_boxplot(rdm_compare_path: Path, output_path: Path) -> None:
    """Plot boxplot of total cost (EB vs BB)."""
    df = load_rdm_compare(rdm_compare_path)
    
    if 'total_cost_nzd_EB' not in df.columns or 'total_cost_nzd_BB' not in df.columns:
        print("[WARN] Cannot plot total cost boxplot: missing columns")
        return
    
    cost_eb = df['total_cost_nzd_EB'].values
    cost_bb = df['total_cost_nzd_BB'].values
    
    # Remove NaN and infinite
    cost_eb = cost_eb[np.isfinite(cost_eb)]
    cost_bb = cost_bb[np.isfinite(cost_bb)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Boxplot (use tick_labels for Matplotlib 3.9+, fallback to labels for older versions)
    try:
        bp = ax.boxplot([cost_eb, cost_bb], tick_labels=['EB', 'BB'], patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
    except TypeError:
        # Fallback for older Matplotlib versions
        bp = ax.boxplot([cost_eb, cost_bb], labels=['EB', 'BB'], patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
    
    # Color boxes
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Total Cost (NZD)', fontsize=11)
    ax.set_title('RDM Total Cost Distribution Comparison (2035)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_rdm_shed_fraction_hist(rdm_compare_path: Path, output_path: Path) -> None:
    """Plot histogram of shed fraction (EB vs BB)."""
    df = load_rdm_compare(rdm_compare_path)
    
    if 'shed_fraction_EB' not in df.columns or 'shed_fraction_BB' not in df.columns:
        print("[WARN] Cannot plot shed fraction histogram: missing columns")
        return
    
    shed_eb = df['shed_fraction_EB'].values
    shed_bb = df['shed_fraction_BB'].values
    
    # Remove NaN and infinite
    shed_eb = shed_eb[np.isfinite(shed_eb)]
    shed_bb = shed_bb[np.isfinite(shed_bb)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    bins = np.linspace(0, max(shed_eb.max(), shed_bb.max()) if len(shed_eb) > 0 and len(shed_bb) > 0 else 1, 30)
    ax.hist(shed_eb, bins=bins, alpha=0.6, color='#3498db', label='EB', edgecolor='black', linewidth=0.5)
    ax.hist(shed_bb, bins=bins, alpha=0.6, color='#e74c3c', label='BB', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Shed Fraction', fontsize=11)
    ax.set_ylabel('Frequency (Futures)', fontsize=11)
    ax.set_title('RDM Shed Fraction Distribution (2035)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


# ============================================================================
# Tariff/Headroom Validity Figures (Curated)
# ============================================================================

def plot_tariff_duration_curve(tariff_csv_path: Path, output_path: Path, epoch_tag: str) -> None:
    """Plot tariff duration curve (sorted descending)."""
    df = load_regional_signal(tariff_csv_path, 'tariff_nzd_per_MWh')
    
    # Sort descending for duration curve
    sorted_tariff = np.sort(df['tariff_nzd_per_MWh'].values)[::-1]
    hours = np.arange(1, len(sorted_tariff) + 1)
    hours_pct = (hours / len(sorted_tariff)) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(hours_pct, sorted_tariff, linewidth=2, color='#2c3e50')
    
    ax.set_xlabel('Duration (% of hours)', fontsize=11)
    ax.set_ylabel('Tariff (NZD/MWh)', fontsize=11)
    ax.set_title(f'Tariff Duration Curve - {epoch_tag}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_tariff_daily_envelope(tariff_csv_path: Path, output_path: Path, epoch_tag: str) -> None:
    """Plot daily tariff envelope (min/median/max per day)."""
    df = load_regional_signal(tariff_csv_path, 'tariff_nzd_per_MWh')
    
    # Group by day
    df['date'] = df['timestamp'].dt.date
    daily_stats = df.groupby('date')['tariff_nzd_per_MWh'].agg([
        ('min', 'min'),
        ('median', 'median'),
        ('max', 'max')
    ]).reset_index()
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot envelope
    ax.fill_between(daily_stats['date'], daily_stats['min'], daily_stats['max'],
                    alpha=0.3, color='#3498db', label='Min-Max Range')
    ax.plot(daily_stats['date'], daily_stats['median'], linewidth=1.5, color='#2c3e50',
           label='Median')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Tariff (NZD/MWh)', fontsize=11)
    ax.set_title(f'Daily Tariff Envelope - {epoch_tag}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_headroom_daily_envelope(headroom_csv_path: Path, output_path: Path, epoch_tag: str) -> None:
    """Plot daily headroom envelope (min/median/max per day)."""
    df = load_regional_signal(headroom_csv_path, 'headroom_MW')
    
    # Group by day
    df['date'] = df['timestamp'].dt.date
    daily_stats = df.groupby('date')['headroom_MW'].agg([
        ('min', 'min'),
        ('median', 'median'),
        ('max', 'max')
    ]).reset_index()
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot envelope
    ax.fill_between(daily_stats['date'], daily_stats['min'], daily_stats['max'],
                    alpha=0.3, color='#3498db', label='Min-Max Range')
    ax.plot(daily_stats['date'], daily_stats['median'], linewidth=1.5, color='#2c3e50',
           label='Median')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Headroom (MW)', fontsize=11)
    ax.set_title(f'Daily Headroom Envelope - {epoch_tag}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


def plot_incremental_vs_headroom_week(incremental_csv_path: Path, headroom_csv_path: Path,
                                      output_path: Path, epoch_tag: str) -> None:
    """Plot incremental electricity vs headroom for one representative week."""
    inc_df = load_incremental_electricity(incremental_csv_path)
    hr_df = load_regional_signal(headroom_csv_path, 'headroom_MW')
    
    # Merge on timestamp
    merged = pd.merge(inc_df[['timestamp', 'incremental_electricity_MW']],
                     hr_df[['timestamp', 'headroom_MW']],
                     on='timestamp', how='inner')
    
    if len(merged) == 0:
        print(f"[WARN] No overlapping timestamps between incremental and headroom data")
        return
    
    # Select first week of February (or first available week)
    feb_mask = merged['timestamp'].dt.month == 2
    if feb_mask.any():
        feb_start = merged[feb_mask].index[0]
        week_data = merged.iloc[feb_start:feb_start+168]
    else:
        week_data = merged.head(168)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot both series
    ax.plot(week_data['timestamp'], week_data['incremental_electricity_MW'].values,
           linewidth=1.5, color='#e74c3c', label='Incremental Electricity', marker='o', markersize=2)
    ax.plot(week_data['timestamp'], week_data['headroom_MW'].values,
           linewidth=1.5, color='#3498db', label='Headroom', marker='s', markersize=2)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Power (MW)', fontsize=11)
    ax.set_title(f'Incremental Electricity vs Headroom - {epoch_tag} (Representative Week)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path.name}")


# ============================================================================
# Main Orchestrator
# ============================================================================

def generate_thesis_figures(
    bundle_dir: Path,
    run_id: str,
    epoch_tags: List[str],
    output_dir: Path,
    create_appendix: bool = False
) -> None:
    """
    Generate all thesis figures from existing outputs.
    
    Args:
        bundle_dir: Bundle directory (e.g., Output/runs/<bundle>)
        run_id: Run ID (e.g., dispatch_prop_v2_capfix1)
        epoch_tags: List of epoch tags (e.g., ['2020', '2025', '2028', '2035_EB', '2035_BB'])
        output_dir: Output directory for figures (thesis_pack/figures)
        create_appendix: If True, create appendix subdirectory for 2020 deep-dive
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if create_appendix:
        appendix_dir = output_dir / 'appendix'
        appendix_dir.mkdir(parents=True, exist_ok=True)
    else:
        appendix_dir = output_dir
    
    print(f"[Thesis Figures] Generating figures for {len(epoch_tags)} epochs...")
    
    # A) 2020 Deep-Dive (Appendix)
    if '2020' in epoch_tags:
        epoch_tag = '2020'
        dispatch_wide_path = bundle_dir / f'epoch{epoch_tag}' / 'dispatch' / run_id / f'site_dispatch_{epoch_tag}_wide.csv'
        
        if dispatch_wide_path.exists():
            print(f"[Thesis Figures] Generating 2020 deep-dive figures...")
            plot_2020_timeseries_week(dispatch_wide_path, appendix_dir / f'heat_2020_timeseries_week.png')
            plot_2020_daily_envelope(dispatch_wide_path, appendix_dir / f'heat_2020_daily_envelope.png')
            plot_2020_load_histogram(dispatch_wide_path, appendix_dir / f'heat_2020_load_histogram_hourly.png')
            plot_2020_weekday_profiles(dispatch_wide_path, appendix_dir / f'heat_2020_weekday_profiles_Feb.png', month=2)
            plot_2020_weekday_profiles(dispatch_wide_path, appendix_dir / f'heat_2020_weekday_profiles_Jun.png', month=6)
        else:
            print(f"[WARN] 2020 dispatch wide not found: {dispatch_wide_path}")
    
    # B) Per-Epoch Utilisation (Curated)
    print(f"[Thesis Figures] Generating utilisation figures...")
    for epoch_tag in epoch_tags:
        dispatch_wide_path = bundle_dir / f'epoch{epoch_tag}' / 'dispatch' / run_id / f'site_dispatch_{epoch_tag}_wide.csv'
        
        if dispatch_wide_path.exists():
            # Units online proportion
            plot_units_online_prop(dispatch_wide_path, output_dir / f'heat_{epoch_tag}_units_online_prop.png', epoch_tag)
            
            # Unit utilisation duration curve
            # Try to find utilities CSV for capacity info
            util_df_path = None
            for possible_path in [
                bundle_dir / f'epoch{epoch_tag}' / 'dispatch' / run_id / 'utilities.csv',
                bundle_dir / f'epoch{epoch_tag}' / 'utilities.csv',
            ]:
                if possible_path.exists():
                    util_df_path = possible_path
                    break
            
            plot_unit_utilisation_duration_prop(
                dispatch_wide_path,
                output_dir / f'heat_{epoch_tag}_unit_utilisation_duration_prop.png',
                epoch_tag,
                util_df_path
            )
        else:
            print(f"[WARN] Dispatch wide not found for {epoch_tag}: {dispatch_wide_path}")
    
    # C) RDM Comparison (Curated)
    print(f"[Thesis Figures] Generating RDM comparison figures...")
    rdm_compare_path = bundle_dir / 'rdm' / 'rdm_compare_2035_EB_vs_BB.csv'
    
    if rdm_compare_path.exists():
        plot_rdm_failure_rate_bar(rdm_compare_path, output_dir / 'rdm_2035_compare_failure_rate_bar.png')
        plot_rdm_regret_cdf(rdm_compare_path, output_dir / 'rdm_2035_compare_regret_cdf.png', log_x=False)
        plot_rdm_regret_cdf(rdm_compare_path, output_dir / 'rdm_2035_compare_regret_cdf_logx.png', log_x=True)
        plot_rdm_total_cost_boxplot(rdm_compare_path, output_dir / 'rdm_2035_compare_total_cost_boxplot.png')
        plot_rdm_shed_fraction_hist(rdm_compare_path, output_dir / 'rdm_2035_compare_shed_fraction_hist.png')
    else:
        print(f"[WARN] RDM comparison not found: {rdm_compare_path}")
    
    # D) Tariff/Headroom Validity (Curated)
    print(f"[Thesis Figures] Generating tariff/headroom validity figures...")
    for epoch_tag in epoch_tags:
        # Tariff duration curve
        tariff_csv_path = bundle_dir / f'epoch{epoch_tag}' / 'regional_signals' / run_id / 'signals' / f'tariff_nzd_per_MWh_{epoch_tag}.csv'
        if tariff_csv_path.exists():
            plot_tariff_duration_curve(tariff_csv_path, output_dir / f'tariff_duration_curve_{epoch_tag}.png', epoch_tag)
            plot_tariff_daily_envelope(tariff_csv_path, output_dir / f'tariff_daily_envelope_{epoch_tag}.png', epoch_tag)
        else:
            print(f"[WARN] Tariff CSV not found for {epoch_tag}: {tariff_csv_path}")
        
        # Headroom daily envelope
        headroom_csv_path = bundle_dir / f'epoch{epoch_tag}' / 'regional_signals' / run_id / 'signals' / f'headroom_MW_{epoch_tag}.csv'
        if headroom_csv_path.exists():
            plot_headroom_daily_envelope(headroom_csv_path, output_dir / f'headroom_daily_envelope_{epoch_tag}.png', epoch_tag)
        else:
            print(f"[WARN] Headroom CSV not found for {epoch_tag}: {headroom_csv_path}")
        
        # Incremental vs headroom bridge plot
        incremental_csv_path = bundle_dir / f'epoch{epoch_tag}' / 'dispatch' / run_id / 'signals' / f'incremental_electricity_MW_{epoch_tag}.csv'
        if incremental_csv_path.exists() and headroom_csv_path.exists():
            plot_incremental_vs_headroom_week(
                incremental_csv_path,
                headroom_csv_path,
                output_dir / f'incremental_electricity_vs_headroom_week_{epoch_tag}.png',
                epoch_tag
            )
        else:
            print(f"[INFO] Skipping incremental vs headroom plot for {epoch_tag} (missing data)")
    
    print(f"[OK] Thesis figures generation complete")


def main():
    """CLI entrypoint for thesis figure generation."""
    parser = argparse.ArgumentParser(
        description='Generate thesis-ready figures from existing outputs'
    )
    parser.add_argument('--bundle', type=str, required=True,
                       help='Bundle name (e.g., full2035_20251225_170112)')
    parser.add_argument('--run-id', type=str, required=True,
                       help='Run ID (e.g., dispatch_prop_v2_capfix1)')
    parser.add_argument('--epoch-tags', type=str, required=True,
                       help='Comma-separated epoch tags (e.g., "2020,2025,2028,2035_EB,2035_BB")')
    parser.add_argument('--output-root', type=str, default='Output',
                       help='Output root directory (default: Output)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for figures (default: thesis_pack/figures)')
    parser.add_argument('--create-appendix', action='store_true',
                       help='Create appendix subdirectory for 2020 deep-dive figures')
    
    args = parser.parse_args()
    
    # Resolve paths
    output_root = Path(resolve_path(args.output_root))
    bundle_dir = output_root / 'runs' / args.bundle
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    
    # Parse epoch tags
    epoch_tags = [tag.strip() for tag in args.epoch_tags.split(',')]
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(resolve_path(args.output_dir))
    else:
        output_dir = bundle_dir / 'thesis_pack' / 'figures'
    
    # Generate figures
    generate_thesis_figures(
        bundle_dir=bundle_dir,
        run_id=args.run_id,
        epoch_tags=epoch_tags,
        output_dir=output_dir,
        create_appendix=args.create_appendix
    )


if __name__ == '__main__':
    main()

