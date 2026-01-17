"""
Grid RDM Figures Generator

Generates comprehensive thesis-ready figures and summary tables for Grid robustness
screening (RDM) across futures for both 2035_EB and 2035_BB pathways.

Read-only: reads existing RDM CSV outputs, does not modify or regenerate them.
"""

import sys
from pathlib import Path

# Add repo root to path
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


# Matplotlib style settings for thesis-ready figures
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


# ============================================================================
# CENTRAL LABEL MAPPING
# ============================================================================

def map_strategy_label(strategy_label: str) -> str:
    """Map strategy labels to thesis-friendly names."""
    mapping = {
        'Auto-select upgrade (min cost)': 'AUTO (min total cost)',
        'No upgrade (allow shedding)': 'NO UPGRADE (shedding allowed)',
        'Force fonterra_opt1_N_plus21MW': 'FORCE GXP upgrade opt1 (+21 MW)',
        'Force fonterra_opt2_N-1_plus32MW': 'FORCE GXP upgrade opt2 (+32 MW, N-1)',
        'Force fonterra_opt3_N_plus97MW': 'FORCE GXP upgrade opt3 (+97 MW)',
        'Force fonterra_opt4_N_plus150MW': 'FORCE GXP upgrade opt4 (+150 MW)',
    }
    return mapping.get(strategy_label, strategy_label)


def map_upgrade_name(upgrade_name: str) -> str:
    """Map upgrade names to thesis-friendly names."""
    mapping = {
        'none': 'None (allow shedding)',
        'fonterra_opt1_N_plus21MW': 'GXP upgrade opt1 (+21 MW)',
        'fonterra_opt2_N-1_plus32MW': 'GXP upgrade opt2 (+32 MW, N-1)',
        'fonterra_opt3_N_plus97MW': 'GXP upgrade opt3 (+97 MW)',
        'fonterra_opt4_N_plus150MW': 'GXP upgrade opt4 (+150 MW)',
    }
    return mapping.get(upgrade_name, upgrade_name)


def map_upgrade_to_capacity(upgrade_name: str) -> float:
    """Map upgrade name to capacity in MW."""
    mapping = {
        'none': 0.0,
        'fonterra_opt1_N_plus21MW': 21.0,
        'fonterra_opt2_N-1_plus32MW': 32.0,
        'fonterra_opt3_N_plus97MW': 97.0,
        'fonterra_opt4_N_plus150MW': 150.0,
    }
    return mapping.get(upgrade_name, 0.0)


def apply_label_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """Apply label mappings to a DataFrame."""
    df_mapped = df.copy()
    if 'strategy_label' in df_mapped.columns:
        df_mapped['strategy_label'] = df_mapped['strategy_label'].apply(map_strategy_label)
    if 'selected_upgrade_name' in df_mapped.columns:
        df_mapped['selected_upgrade_name'] = df_mapped['selected_upgrade_name'].apply(map_upgrade_name)
    return df_mapped


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_required_columns(df: pd.DataFrame, required_cols: List[str], file_name: str) -> None:
    """Check that required columns exist in DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {file_name}:\n"
            f"  Missing: {missing}\n"
            f"  Available: {list(df.columns)}"
        )


def get_safe_filename(base_path: Path, overwrite: bool = False) -> Path:
    """Get a safe filename, appending _v2, _v3 etc. if file exists and overwrite=False."""
    if overwrite or not base_path.exists():
        return base_path
    
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    
    version = 2
    while True:
        candidate = parent / f"{stem}_v{version}{suffix}"
        if not candidate.exists():
            return candidate
        version += 1


def verify_paired_futures(
    eb_futures: set,
    bb_futures: set,
    context: str = ""
) -> None:
    """Verify that EB and BB have identical future_id sets."""
    if eb_futures != bb_futures:
        missing_eb = bb_futures - eb_futures
        missing_bb = eb_futures - bb_futures
        error_msg = f"Future ID mismatch between EB and BB{context}:\n"
        if missing_eb:
            error_msg += f"  Missing in EB: {sorted(missing_eb)}\n"
        if missing_bb:
            error_msg += f"  Missing in BB: {sorted(missing_bb)}\n"
        error_msg += f"  EB futures: {len(eb_futures)} (IDs: {sorted(eb_futures)[:10]}...)\n"
        error_msg += f"  BB futures: {len(bb_futures)} (IDs: {sorted(bb_futures)[:10]}...)\n"
        raise ValueError(error_msg)


def load_rdm_data(bundle_dir: Path, rdm_subdir: str = "rdm") -> Dict[str, pd.DataFrame]:
    """Load all RDM CSV files."""
    rdm_dir = bundle_dir / rdm_subdir
    
    if not rdm_dir.exists():
        raise FileNotFoundError(f"RDM directory not found: {rdm_dir}")
    
    files = {
        'matrix_eb': rdm_dir / 'rdm_matrix_2035_EB.csv',
        'matrix_bb': rdm_dir / 'rdm_matrix_2035_BB.csv',
        'summary_eb': rdm_dir / 'rdm_summary_2035_EB.csv',
        'summary_bb': rdm_dir / 'rdm_summary_2035_BB.csv',
        'compare': rdm_dir / 'rdm_compare_2035_EB_vs_BB.csv',
    }
    
    data = {}
    for key, path in files.items():
        if path.exists():
            data[key] = pd.read_csv(path)
            print(f"  [OK] {key}: {len(data[key])} rows from {path.name}")
        else:
            if key in ['summary_eb', 'summary_bb']:
                print(f"  [WARN] {key}: {path.name} not found (optional for some figures)")
            else:
                raise FileNotFoundError(f"Required file not found: {path}")
    
    return data


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================

def plot_f1_total_cost_boxplot(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F1: Total cost distributions by strategy."""
    check_required_columns(df_matrix, ['strategy_label', 'total_cost_nzd'], 'rdm_matrix')
    
    # Apply label mappings
    df_mapped = apply_label_mappings(df_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df_mapped['strategy_label'].unique()
    data_by_strategy = [df_mapped[df_mapped['strategy_label'] == s]['total_cost_nzd'].values 
                        for s in strategies]
    
    bp = ax.boxplot(data_by_strategy, patch_artist=True)
    ax.set_xticklabels(strategies)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Total Cost (NZD)', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_title(f'{epoch_tag} | Total Cost by Strategy', fontsize=12)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f2_annual_shed_boxplot(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F2a: Annual shed MWh distributions by strategy."""
    check_required_columns(df_matrix, ['strategy_label', 'annual_shed_MWh'], 'rdm_matrix')
    
    # Apply label mappings
    df_mapped = apply_label_mappings(df_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df_mapped['strategy_label'].unique()
    data_by_strategy = [df_mapped[df_mapped['strategy_label'] == s]['annual_shed_MWh'].values 
                        for s in strategies]
    
    bp = ax.boxplot(data_by_strategy, patch_artist=True)
    ax.set_xticklabels(strategies)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Annual Shed (MWh)', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_title(f'{epoch_tag} | Annual Shedding by Strategy', fontsize=12)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f2_shed_fraction_boxplot(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F2b: Shed fraction distributions by strategy."""
    check_required_columns(df_matrix, ['strategy_label', 'shed_fraction'], 'rdm_matrix')
    
    # Apply label mappings
    df_mapped = apply_label_mappings(df_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df_mapped['strategy_label'].unique()
    data_by_strategy = [df_mapped[df_mapped['strategy_label'] == s]['shed_fraction'].values 
                        for s in strategies]
    
    bp = ax.boxplot(data_by_strategy, patch_artist=True)
    ax.set_xticklabels(strategies)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Shed Fraction', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_title(f'{epoch_tag} | Shed Fraction by Strategy', fontsize=12)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f3_binding_hours_boxplot(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F3: Binding hours by strategy."""
    check_required_columns(df_matrix, ['strategy_label', 'n_hours_binding'], 'rdm_matrix')
    
    # Apply label mappings
    df_mapped = apply_label_mappings(df_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df_mapped['strategy_label'].unique()
    data_by_strategy = [df_mapped[df_mapped['strategy_label'] == s]['n_hours_binding'].values 
                        for s in strategies]
    
    bp = ax.boxplot(data_by_strategy, patch_artist=True)
    ax.set_xticklabels(strategies)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Binding Hours', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_title(f'{epoch_tag} | Binding Hours by Strategy', fontsize=12)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f4_upgrade_choice_frequency(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F4: Upgrade choice frequency by strategy."""
    check_required_columns(df_matrix, ['strategy_label', 'selected_upgrade_name'], 'rdm_matrix')
    
    # Apply label mappings
    df_mapped = apply_label_mappings(df_matrix)
    
    # Create cross-tabulation
    crosstab = pd.crosstab(
        df_mapped['strategy_label'],
        df_mapped['selected_upgrade_name'],
        normalize='index'
    ) * 100  # Convert to percentages
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
    
    ax.set_ylabel('Frequency (%)', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_title(f'{epoch_tag} | Upgrade Choice Frequency by Strategy', fontsize=12)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.legend(title='Upgrade', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f5_tradeoff_scatter(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F5: Strategy trade-off scatter (cost vs shed)."""
    check_required_columns(df_matrix, ['strategy_label', 'total_cost_nzd', 'annual_shed_MWh'], 'rdm_matrix')
    
    # Apply label mappings
    df_mapped = apply_label_mappings(df_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    strategies = df_mapped['strategy_label'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    
    for i, strategy in enumerate(strategies):
        mask = df_mapped['strategy_label'] == strategy
        ax.scatter(
            df_mapped.loc[mask, 'annual_shed_MWh'],
            df_mapped.loc[mask, 'total_cost_nzd'],
            label=strategy,
            color=colors[i],
            marker=markers[i % len(markers)],
            alpha=0.6,
            s=30
        )
    
    ax.set_xlabel('Annual Shed (MWh)', fontsize=11)
    ax.set_ylabel('Total Cost (NZD)', fontsize=11)
    ax.set_title(f'{epoch_tag} | Cost vs Shed Trade-off by Strategy', fontsize=12)
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f6_headroom_vs_capacity(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F6: Driver → upgrade capacity relationship (AUTO only)."""
    check_required_columns(df_matrix, ['U_headroom_mult', 'selected_capacity_MW', 'strategy_id'], 'rdm_matrix')
    
    # Filter to AUTO strategy only
    df_auto = df_matrix[df_matrix['strategy_id'] == 'S_AUTO'].copy()
    
    if len(df_auto) == 0:
        raise ValueError("No AUTO strategy rows found in rdm_matrix")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(
        df_auto['U_headroom_mult'],
        df_auto['selected_capacity_MW'],
        alpha=0.6,
        s=50
    )
    
    ax.set_xlabel('Headroom Multiplier', fontsize=11)
    ax.set_ylabel('Selected Capacity (MW)', fontsize=11)
    ax.set_title(f'{epoch_tag} | Headroom vs Selected Capacity (AUTO)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f7_increment_vs_binding(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F7: Driver → binding hours (AUTO only)."""
    check_required_columns(df_matrix, ['U_inc_mult', 'n_hours_binding', 'strategy_id'], 'rdm_matrix')
    
    # Filter to AUTO strategy only
    df_auto = df_matrix[df_matrix['strategy_id'] == 'S_AUTO'].copy()
    
    if len(df_auto) == 0:
        raise ValueError("No AUTO strategy rows found in rdm_matrix")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(
        df_auto['U_inc_mult'],
        df_auto['n_hours_binding'],
        alpha=0.6,
        s=50
    )
    
    ax.set_xlabel('Incremental Multiplier', fontsize=11)
    ax.set_ylabel('Binding Hours', fontsize=11)
    ax.set_title(f'{epoch_tag} | Increment vs Binding Hours (AUTO)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f8_shed_heatmap(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F8: Vulnerability-style heatmap (AUTO only)."""
    check_required_columns(df_matrix, ['U_headroom_mult', 'U_inc_mult', 'annual_shed_MWh', 'strategy_id'], 'rdm_matrix')
    
    # Filter to AUTO strategy only
    df_auto = df_matrix[df_matrix['strategy_id'] == 'S_AUTO'].copy()
    
    if len(df_auto) == 0:
        raise ValueError("No AUTO strategy rows found in rdm_matrix")
    
    # Create bins
    n_bins = 10
    headroom_bins = np.linspace(df_auto['U_headroom_mult'].min(), df_auto['U_headroom_mult'].max(), n_bins+1)
    inc_bins = np.linspace(df_auto['U_inc_mult'].min(), df_auto['U_inc_mult'].max(), n_bins+1)
    
    # Bin data
    df_auto['headroom_bin'] = pd.cut(df_auto['U_headroom_mult'], bins=headroom_bins)
    df_auto['inc_bin'] = pd.cut(df_auto['U_inc_mult'], bins=inc_bins)
    
    # Compute mean shed per bin
    heatmap_data = df_auto.groupby(['headroom_bin', 'inc_bin'])['annual_shed_MWh'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(heatmap_data.values, aspect='auto', cmap='YlOrRd', origin='lower')
    
    # Set ticks - use bin edges for labels
    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_bins))
    ax.set_xticklabels([f'{inc_bins[i]:.2f}' for i in range(n_bins)], rotation=45, ha='right')
    ax.set_yticklabels([f'{headroom_bins[i]:.2f}' for i in range(n_bins)])
    
    ax.set_xlabel('Incremental Multiplier', fontsize=11)
    ax.set_ylabel('Headroom Multiplier', fontsize=11)
    ax.set_title(f'{epoch_tag} | Mean Annual Shed Heatmap (AUTO)', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Mean Annual Shed (MWh)')
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f9_cost_scatter_eb_vs_bb(
    df_compare: pd.DataFrame,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F9: EB vs BB total cost scatter."""
    check_required_columns(df_compare, ['total_cost_nzd_EB', 'total_cost_nzd_BB'], 'rdm_compare')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(
        df_compare['total_cost_nzd_EB'],
        df_compare['total_cost_nzd_BB'],
        alpha=0.6,
        s=50
    )
    
    # Add 45-degree line
    min_val = min(df_compare['total_cost_nzd_EB'].min(), df_compare['total_cost_nzd_BB'].min())
    max_val = max(df_compare['total_cost_nzd_EB'].max(), df_compare['total_cost_nzd_BB'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='1:1 line')
    
    ax.set_xlabel('Total Cost EB (NZD)', fontsize=11)
    ax.set_ylabel('Total Cost BB (NZD)', fontsize=11)
    ax.set_title('2035_EB vs 2035_BB | Total Cost Comparison', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f10_delta_cost_hist(
    df_compare: pd.DataFrame,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F10: Distribution of Δcost (EB−BB)."""
    check_required_columns(df_compare, ['total_cost_nzd_EB', 'total_cost_nzd_BB'], 'rdm_compare')
    
    delta = df_compare['total_cost_nzd_EB'] - df_compare['total_cost_nzd_BB']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(delta, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero (EB=BB)')
    
    ax.set_xlabel('Δ Cost (EB - BB) (NZD)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('2035_EB vs 2035_BB | Distribution of Cost Difference', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f11_upgrade_confusion(
    df_compare: pd.DataFrame,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F11: Upgrade-choice confusion matrix."""
    check_required_columns(df_compare, ['selected_upgrade_name_EB', 'selected_upgrade_name_BB'], 'rdm_compare')
    
    # Create confusion matrix
    confusion = pd.crosstab(
        df_compare['selected_upgrade_name_EB'],
        df_compare['selected_upgrade_name_BB'],
        margins=False
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(confusion.values, aspect='auto', cmap='Blues', origin='upper')
    
    # Set ticks
    ax.set_xticks(np.arange(len(confusion.columns)))
    ax.set_yticks(np.arange(len(confusion.index)))
    ax.set_xticklabels(confusion.columns, rotation=45, ha='right')
    ax.set_yticklabels(confusion.index)
    
    ax.set_xlabel('BB Selected Upgrade', fontsize=11)
    ax.set_ylabel('EB Selected Upgrade', fontsize=11)
    ax.set_title('2035_EB vs 2035_BB | Upgrade Choice Confusion Matrix', fontsize=12)
    
    # Add text annotations
    for i in range(len(confusion.index)):
        for j in range(len(confusion.columns)):
            text = ax.text(j, i, confusion.iloc[i, j],
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Count')
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f12_shed_scatter_eb_vs_bb(
    df_compare: pd.DataFrame,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F12: Shedding comparison scatter."""
    check_required_columns(df_compare, ['annual_shed_MWh_EB', 'annual_shed_MWh_BB'], 'rdm_compare')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(
        df_compare['annual_shed_MWh_EB'],
        df_compare['annual_shed_MWh_BB'],
        alpha=0.6,
        s=50
    )
    
    # Add 45-degree line
    min_val = min(df_compare['annual_shed_MWh_EB'].min(), df_compare['annual_shed_MWh_BB'].min())
    max_val = max(df_compare['annual_shed_MWh_EB'].max(), df_compare['annual_shed_MWh_BB'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='1:1 line')
    
    ax.set_xlabel('Annual Shed EB (MWh)', fontsize=11)
    ax.set_ylabel('Annual Shed BB (MWh)', fontsize=11)
    ax.set_title('2035_EB vs 2035_BB | Annual Shedding Comparison', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


def plot_f13_binding_hours_scatter_eb_vs_bb(
    df_compare: pd.DataFrame,
    bundle_name: str,
    output_path: Path,
    overwrite: bool
) -> Path:
    """F13: Binding hours comparison scatter."""
    check_required_columns(df_compare, ['n_hours_binding_EB', 'n_hours_binding_BB'], 'rdm_compare')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(
        df_compare['n_hours_binding_EB'],
        df_compare['n_hours_binding_BB'],
        alpha=0.6,
        s=50
    )
    
    # Add 45-degree line
    min_val = min(df_compare['n_hours_binding_EB'].min(), df_compare['n_hours_binding_BB'].min())
    max_val = max(df_compare['n_hours_binding_EB'].max(), df_compare['n_hours_binding_BB'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='1:1 line')
    
    ax.set_xlabel('Binding Hours EB', fontsize=11)
    ax.set_ylabel('Binding Hours BB', fontsize=11)
    ax.set_title('2035_EB vs 2035_BB | Binding Hours Comparison', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_path = get_safe_filename(output_path, overwrite)
    plt.savefig(safe_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return safe_path


# ============================================================================
# TABLE GENERATION FUNCTIONS
# ============================================================================

def generate_t1_strategy_metrics(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    output_path: Path,
    overwrite: bool,
    satisficing_tol: float = 0.001
) -> Path:
    """T1: Strategy robustness summary per epoch."""
    check_required_columns(df_matrix, [
        'strategy_label', 'total_cost_nzd', 'annual_shed_MWh',
        'shed_fraction', 'n_hours_binding'
    ], 'rdm_matrix')
    
    results = []
    
    for strategy in df_matrix['strategy_label'].unique():
        df_strat = df_matrix[df_matrix['strategy_label'] == strategy]
        
        results.append({
            'strategy_label': strategy,
            'mean_total_cost_nzd': df_strat['total_cost_nzd'].mean(),
            'p50_total_cost_nzd': df_strat['total_cost_nzd'].median(),
            'p95_total_cost_nzd': df_strat['total_cost_nzd'].quantile(0.95),
            'mean_annual_shed_MWh': df_strat['annual_shed_MWh'].mean(),
            'p95_annual_shed_MWh': df_strat['annual_shed_MWh'].quantile(0.95),
            'share_satisficing_0.001': (df_strat['shed_fraction'] <= satisficing_tol).mean(),
            'share_satisficing_0.000': (df_strat['shed_fraction'] <= 0.000).mean(),
            'share_zero_binding': (df_strat['n_hours_binding'] == 0).mean(),
        })
    
    result_df = pd.DataFrame(results)
    
    safe_path = get_safe_filename(output_path, overwrite)
    result_df.to_csv(safe_path, index=False)
    
    return safe_path


def generate_t2_figure_manifest(
    figure_paths: List[Path],
    output_path: Path,
    overwrite: bool
) -> Path:
    """T2: Figure manifest for thesis integration."""
    manifest_rows = []
    
    for fig_path in figure_paths:
        filename = fig_path.name
        
        # Detect epoch
        if 'EB_vs_BB' in filename:
            epoch = 'EB_vs_BB'
        elif '2035_EB' in filename:
            epoch = '2035_EB'
        elif '2035_BB' in filename:
            epoch = '2035_BB'
        else:
            epoch = 'unknown'
        
        # Infer data source
        if 'EB_vs_BB' in filename or 'delta' in filename.lower() or 'confusion' in filename.lower():
            data_source = 'rdm_compare'
        elif 'auto' in filename.lower() and ('headroom' in filename.lower() or 'increment' in filename.lower() or 'heatmap' in filename.lower()):
            data_source = 'rdm_matrix'  # AUTO diagnostics use matrix data
        elif 'boxplot' in filename or 'tradeoff' in filename or 'frequency' in filename:
            data_source = 'rdm_matrix'
        else:
            data_source = 'rdm_matrix'  # Default to matrix
        
        # Generate primary message and caption suggestions
        if 'total_cost' in filename:
            primary_message = 'Total cost distribution across strategies'
            caption = f'Distribution of total cost (NZD) by strategy for {epoch}'
        elif 'annual_shed' in filename:
            primary_message = 'Annual shedding distribution across strategies'
            caption = f'Distribution of annual shedding (MWh) by strategy for {epoch}'
        elif 'shed_fraction' in filename:
            primary_message = 'Shed fraction distribution across strategies'
            caption = f'Distribution of shed fraction by strategy for {epoch}'
        elif 'binding_hours' in filename:
            primary_message = 'Binding hours (headroom stress) across strategies'
            caption = f'Distribution of binding hours by strategy for {epoch}'
        elif 'frequency' in filename:
            primary_message = 'Upgrade choice frequency by strategy'
            caption = f'Frequency of upgrade choices by strategy for {epoch}'
        elif 'tradeoff' in filename:
            primary_message = 'Cost vs shedding trade-off by strategy'
            caption = f'Cost vs shedding trade-off scatter by strategy for {epoch}'
        elif 'headroom_vs_selected' in filename:
            primary_message = 'Headroom multiplier vs selected capacity relationship'
            caption = f'Relationship between headroom multiplier and selected capacity (AUTO) for {epoch}'
        elif 'increment_vs_binding' in filename:
            primary_message = 'Incremental multiplier vs binding hours relationship'
            caption = f'Relationship between incremental multiplier and binding hours (AUTO) for {epoch}'
        elif 'heatmap' in filename:
            primary_message = 'Vulnerability heatmap: headroom vs increment drivers'
            caption = f'Mean annual shedding heatmap by headroom and incremental multipliers (AUTO) for {epoch}'
        elif 'EB_vs_BB' in filename and 'total_cost' in filename:
            primary_message = 'EB vs BB total cost comparison'
            caption = 'Total cost comparison between EB and BB pathways (paired futures)'
        elif 'delta_total_cost' in filename:
            primary_message = 'Distribution of cost difference (EB - BB)'
            caption = 'Histogram of cost difference between EB and BB pathways'
        elif 'confusion' in filename:
            primary_message = 'Upgrade choice agreement between EB and BB'
            caption = 'Confusion matrix of upgrade choices between EB and BB pathways'
        elif 'EB_vs_BB' in filename and 'annual_shed' in filename:
            primary_message = 'EB vs BB annual shedding comparison'
            caption = 'Annual shedding comparison between EB and BB pathways'
        elif 'EB_vs_BB' in filename and 'binding_hours' in filename:
            primary_message = 'EB vs BB binding hours comparison'
            caption = 'Binding hours comparison between EB and BB pathways'
        else:
            primary_message = 'RDM analysis figure'
            caption = f'RDM analysis figure for {epoch}'
        
        manifest_rows.append({
            'figure_file': filename,
            'epoch': epoch,
            'data_source_csv': data_source,
            'primary_message': primary_message,
            'suggested_caption': caption
        })
    
    manifest_df = pd.DataFrame(manifest_rows)
    
    safe_path = get_safe_filename(output_path, overwrite)
    manifest_df.to_csv(safe_path, index=False)
    
    return safe_path


# ============================================================================
# MAIN FUNCTION
# ============================================================================

# ============================================================================
# V2 FIGURE GENERATION FUNCTIONS (IMPROVED DESIGNS)
# ============================================================================

def plot_v2_auto_upgrade_frequency(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    output_path: Path
) -> Path:
    """V2A: AUTO-only upgrade choice frequency (simple bar chart)."""
    check_required_columns(df_matrix, ['strategy_id', 'selected_upgrade_name'], 'rdm_matrix')
    
    # Filter to AUTO strategy only
    df_auto = df_matrix[df_matrix['strategy_id'] == 'S_AUTO'].copy()
    if len(df_auto) == 0:
        raise ValueError("No AUTO strategy rows found")
    
    # Apply label mappings
    df_auto = apply_label_mappings(df_auto)
    
    # Count frequencies
    upgrade_counts = df_auto['selected_upgrade_name'].value_counts()
    upgrade_pct = (upgrade_counts / len(df_auto) * 100).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Horizontal bar chart for better readability
    bars = ax.barh(upgrade_pct.index, upgrade_pct.values, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (upgrade, pct) in enumerate(upgrade_pct.items()):
        ax.text(pct + 1, i, f'{pct:.1f}%', va='center', fontsize=10)
    
    ax.set_xlabel('Frequency (%)', fontsize=11)
    ax.set_ylabel('Upgrade Choice', fontsize=11)
    ax.set_title(f'{epoch_tag} | AUTO Strategy Upgrade Choice Frequency', fontsize=12)
    ax.set_xlim(0, max(upgrade_pct.values) * 1.15)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_path


def plot_v2_strategy_upgrade_heatmap(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    output_path: Path
) -> Path:
    """V2B: Strategy × upgrade frequency heatmap."""
    check_required_columns(df_matrix, ['strategy_label', 'selected_upgrade_name'], 'rdm_matrix')
    
    # Apply label mappings
    df_mapped = apply_label_mappings(df_matrix)
    
    # Create cross-tabulation (counts, then convert to percentages per strategy)
    crosstab = pd.crosstab(
        df_mapped['strategy_label'],
        df_mapped['selected_upgrade_name'],
        normalize='index'
    ) * 100
    
    # Sort strategies and upgrades for better readability
    strategy_order = sorted(crosstab.index)
    upgrade_order = sorted(crosstab.columns)
    crosstab = crosstab.loc[strategy_order, upgrade_order]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(crosstab.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(np.arange(len(upgrade_order)))
    ax.set_yticks(np.arange(len(strategy_order)))
    ax.set_xticklabels(upgrade_order, rotation=45, ha='right')
    ax.set_yticklabels(strategy_order)
    
    # Add text annotations with percentages
    for i in range(len(strategy_order)):
        for j in range(len(upgrade_order)):
            value = crosstab.iloc[i, j]
            text_color = 'white' if value > 50 else 'black'
            ax.text(j, i, f'{value:.0f}%', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Upgrade Choice', fontsize=11)
    ax.set_ylabel('Strategy', fontsize=11)
    ax.set_title(f'{epoch_tag} | Strategy × Upgrade Choice Frequency', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Frequency (%)')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_path


def plot_v2_auto_upgrade_vs_headroom(
    df_matrix: pd.DataFrame,
    epoch_tag: str,
    output_path: Path
) -> Path:
    """V2C: AUTO upgrade choice vs headroom driver."""
    check_required_columns(df_matrix, ['strategy_id', 'selected_upgrade_name', 'U_headroom_mult'], 'rdm_matrix')
    
    # Filter to AUTO strategy only
    df_auto = df_matrix[df_matrix['strategy_id'] == 'S_AUTO'].copy()
    if len(df_auto) == 0:
        raise ValueError("No AUTO strategy rows found")
    
    # Map upgrade to capacity
    df_auto['upgrade_capacity'] = df_auto['selected_upgrade_name'].apply(map_upgrade_to_capacity)
    
    # Add slight jitter to y-axis for better visibility
    np.random.seed(42)  # For reproducibility
    jitter = np.random.normal(0, 2, len(df_auto))
    df_auto['upgrade_capacity_jittered'] = df_auto['upgrade_capacity'] + jitter
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by upgrade choice
    upgrade_colors = {
        'none': '#d62728',
        'fonterra_opt1_N_plus21MW': '#ff7f0e',
        'fonterra_opt2_N-1_plus32MW': '#2ca02c',
        'fonterra_opt3_N_plus97MW': '#1f77b4',
        'fonterra_opt4_N_plus150MW': '#9467bd',
    }
    
    for upgrade in df_auto['selected_upgrade_name'].unique():
        mask = df_auto['selected_upgrade_name'] == upgrade
        color = upgrade_colors.get(upgrade, '#7f7f7f')
        label = map_upgrade_name(upgrade)
        ax.scatter(
            df_auto.loc[mask, 'U_headroom_mult'],
            df_auto.loc[mask, 'upgrade_capacity_jittered'],
            label=label,
            color=color,
            alpha=0.6,
            s=50
        )
    
    # Add horizontal lines at actual capacity levels
    for capacity in [0, 21, 32, 97, 150]:
        ax.axhline(y=capacity, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Headroom Multiplier', fontsize=11)
    ax.set_ylabel('Selected Upgrade Capacity (MW)', fontsize=11)
    ax.set_title(f'{epoch_tag} | AUTO Upgrade Choice vs Headroom Driver', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_path


def plot_v2_combined_auto_upgrade_frequency(
    df_matrix_eb: pd.DataFrame,
    df_matrix_bb: pd.DataFrame,
    output_path: Path
) -> Path:
    """V2A (combined): Combined EB vs BB AUTO upgrade frequency."""
    check_required_columns(df_matrix_eb, ['strategy_id', 'selected_upgrade_name'], 'rdm_matrix_EB')
    check_required_columns(df_matrix_bb, ['strategy_id', 'selected_upgrade_name'], 'rdm_matrix_BB')
    
    # Filter to AUTO strategy only
    df_auto_eb = df_matrix_eb[df_matrix_eb['strategy_id'] == 'S_AUTO'].copy()
    df_auto_bb = df_matrix_bb[df_matrix_bb['strategy_id'] == 'S_AUTO'].copy()
    
    # Apply label mappings
    df_auto_eb = apply_label_mappings(df_auto_eb)
    df_auto_bb = apply_label_mappings(df_auto_bb)
    
    # Get all unique upgrades
    all_upgrades = sorted(set(df_auto_eb['selected_upgrade_name'].unique()) | 
                         set(df_auto_bb['selected_upgrade_name'].unique()))
    
    # Compute frequencies
    freq_eb = df_auto_eb['selected_upgrade_name'].value_counts(normalize=True) * 100
    freq_bb = df_auto_bb['selected_upgrade_name'].value_counts(normalize=True) * 100
    
    # Align to all upgrades
    freq_eb = freq_eb.reindex(all_upgrades, fill_value=0)
    freq_bb = freq_bb.reindex(all_upgrades, fill_value=0)
    
    x = np.arange(len(all_upgrades))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, freq_eb.values, width, label='2035_EB', alpha=0.8, color='#d62728')
    bars2 = ax.bar(x + width/2, freq_bb.values, width, label='2035_BB', alpha=0.8, color='#9467bd')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Upgrade Choice', fontsize=11)
    ax.set_ylabel('Frequency (%)', fontsize=11)
    ax.set_title('2035_EB vs 2035_BB | AUTO Strategy Upgrade Choice Frequency', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_upgrades, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_path


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive RDM figures and summary tables"
    )
    parser.add_argument(
        '--bundle',
        type=str,
        default='poc_20260105_release02',
        help='Bundle name (default: poc_20260105_release02)'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='Output',
        help='Output root directory (default: Output)'
    )
    parser.add_argument(
        '--rdm-subdir',
        type=str,
        default='rdm',
        help='RDM subdirectory name (default: rdm)'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Output directory (default: <output-root>/runs/<bundle>/<rdm-subdir>/figures)'
    )
    parser.add_argument(
        '--formats',
        type=str,
        default='png',
        help='Output formats, comma-separated (default: png; options: png,pdf)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files (default: append _v2, _v3, etc.)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    root = repo_root()
    bundle_dir = root / args.output_root / 'runs' / args.bundle
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    
    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        output_dir = bundle_dir / args.rdm_subdir / 'figures'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GRID RDM FIGURES GENERATOR")
    print("=" * 80)
    print(f"\n[CONFIGURATION]")
    print(f"  Bundle: {args.bundle}")
    print(f"  Output root: {args.output_root}")
    print(f"  RDM subdir: {args.rdm_subdir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Formats: {args.formats}")
    print(f"  Overwrite: {args.overwrite}")
    
    # Load RDM data
    print(f"\n[LOADING DATA]")
    data = load_rdm_data(bundle_dir, args.rdm_subdir)
    
    # Verify paired futures
    print(f"\n[VERIFYING PAIRED FUTURES]")
    if 'matrix_eb' in data and 'matrix_bb' in data:
        eb_futures = set(data['matrix_eb']['future_id'].unique())
        bb_futures = set(data['matrix_bb']['future_id'].unique())
        print(f"  EB futures: {len(eb_futures)} unique IDs")
        print(f"  BB futures: {len(bb_futures)} unique IDs")
        verify_paired_futures(eb_futures, bb_futures, " (matrix)")
        print(f"  [OK] Paired futures verified: {len(eb_futures)} futures match")
    
    if 'summary_eb' in data and 'summary_bb' in data:
        eb_futures_summary = set(data['summary_eb']['future_id'].unique())
        bb_futures_summary = set(data['summary_bb']['future_id'].unique())
        print(f"  EB summary futures: {len(eb_futures_summary)} unique IDs")
        print(f"  BB summary futures: {len(bb_futures_summary)} unique IDs")
        verify_paired_futures(eb_futures_summary, bb_futures_summary, " (summary)")
        print(f"  [OK] Paired futures verified in summaries")
    
    if 'compare' in data:
        compare_futures = set(data['compare']['future_id'].unique())
        print(f"  Compare futures: {len(compare_futures)} unique IDs")
        if 'matrix_eb' in data:
            eb_futures = set(data['matrix_eb']['future_id'].unique())
            if compare_futures != eb_futures:
                print(f"  [WARN] Compare futures don't match matrix EB futures")
    
    # Generate figures
    print(f"\n[GENERATING FIGURES]")
    figure_paths = []
    
    # Within-pathway figures (2035_EB and 2035_BB)
    for epoch_tag in ['2035_EB', '2035_BB']:
        epoch_key = epoch_tag.replace('2035_', '').lower()
        matrix_key = f'matrix_{epoch_key}'
        summary_key = f'summary_{epoch_key}'
        
        if matrix_key not in data:
            print(f"  [SKIP] {epoch_tag}: matrix data not found")
            continue
        
        df_matrix = data[matrix_key]
        print(f"\n  [{epoch_tag}] Generating within-pathway figures...")
        
        # F1: Total cost boxplot
        try:
            path = plot_f1_total_cost_boxplot(
                df_matrix, epoch_tag, args.bundle,
                output_dir / f'grid_rdm_{epoch_tag}__total_cost_boxplot.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F1: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F1: {e}")
        
        # F2a: Annual shed boxplot
        try:
            path = plot_f2_annual_shed_boxplot(
                df_matrix, epoch_tag, args.bundle,
                output_dir / f'grid_rdm_{epoch_tag}__annual_shed_MWh_boxplot.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F2a: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F2a: {e}")
        
        # F2b: Shed fraction boxplot
        try:
            path = plot_f2_shed_fraction_boxplot(
                df_matrix, epoch_tag, args.bundle,
                output_dir / f'grid_rdm_{epoch_tag}__shed_fraction_boxplot.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F2b: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F2b: {e}")
        
        # F3: Binding hours boxplot
        try:
            path = plot_f3_binding_hours_boxplot(
                df_matrix, epoch_tag, args.bundle,
                output_dir / f'grid_rdm_{epoch_tag}__binding_hours_boxplot.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F3: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F3: {e}")
        
        # F4: Upgrade choice frequency
        try:
            path = plot_f4_upgrade_choice_frequency(
                df_matrix, epoch_tag, args.bundle,
                output_dir / f'grid_rdm_{epoch_tag}__upgrade_choice_frequency.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F4: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F4: {e}")
        
        # F5: Trade-off scatter
        try:
            path = plot_f5_tradeoff_scatter(
                df_matrix, epoch_tag, args.bundle,
                output_dir / f'grid_rdm_{epoch_tag}__tradeoff_cost_vs_shed_scatter.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F5: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F5: {e}")
        
        # Future-driver diagnostics (AUTO only, from matrix)
        print(f"\n  [{epoch_tag}] Generating driver diagnostics (AUTO)...")
        
        # F6: Headroom vs capacity
        try:
            path = plot_f6_headroom_vs_capacity(
                df_matrix, epoch_tag, args.bundle,
                output_dir / f'grid_rdm_{epoch_tag}__auto_headroom_vs_selected_capacity.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F6: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F6: {e}")
        
        # F7: Increment vs binding
        try:
            path = plot_f7_increment_vs_binding(
                df_matrix, epoch_tag, args.bundle,
                output_dir / f'grid_rdm_{epoch_tag}__auto_increment_vs_binding_hours.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F7: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F7: {e}")
        
        # F8: Shed heatmap
        try:
            path = plot_f8_shed_heatmap(
                df_matrix, epoch_tag, args.bundle,
                output_dir / f'grid_rdm_{epoch_tag}__auto_shed_heatmap_headroom_vs_increment.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F8: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F8: {e}")
    
    # EB vs BB comparison figures
    if 'compare' in data:
        print(f"\n  [EB vs BB] Generating comparison figures...")
        df_compare = data['compare']
        
        # F9: Cost scatter
        try:
            path = plot_f9_cost_scatter_eb_vs_bb(
                df_compare, args.bundle,
                output_dir / 'grid_rdm_EB_vs_BB__total_cost_scatter.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F9: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F9: {e}")
        
        # F10: Delta cost hist
        try:
            path = plot_f10_delta_cost_hist(
                df_compare, args.bundle,
                output_dir / 'grid_rdm_EB_vs_BB__delta_total_cost_hist.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F10: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F10: {e}")
        
        # F11: Upgrade confusion
        try:
            path = plot_f11_upgrade_confusion(
                df_compare, args.bundle,
                output_dir / 'grid_rdm_EB_vs_BB__upgrade_choice_confusion.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F11: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F11: {e}")
        
        # F12: Shed scatter
        try:
            path = plot_f12_shed_scatter_eb_vs_bb(
                df_compare, args.bundle,
                output_dir / 'grid_rdm_EB_vs_BB__annual_shed_scatter.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F12: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F12: {e}")
        
        # F13: Binding hours scatter
        try:
            path = plot_f13_binding_hours_scatter_eb_vs_bb(
                df_compare, args.bundle,
                output_dir / 'grid_rdm_EB_vs_BB__binding_hours_scatter.png',
                args.overwrite
            )
            figure_paths.append(path)
            print(f"    [OK] F13: {path.name}")
        except Exception as e:
            print(f"    [ERROR] F13: {e}")
    
    # Generate tables
    print(f"\n[GENERATING TABLES]")
    
    # T1: Strategy metrics per epoch
    for epoch_tag in ['2035_EB', '2035_BB']:
        epoch_key = epoch_tag.replace('2035_', '').lower()
        matrix_key = f'matrix_{epoch_key}'
        
        if matrix_key in data:
            try:
                path = generate_t1_strategy_metrics(
                    data[matrix_key], epoch_tag,
                    output_dir / f'grid_rdm_{epoch_tag}__strategy_metrics.csv',
                    args.overwrite
                )
                print(f"  [OK] T1 ({epoch_tag}): {path.name}")
            except Exception as e:
                print(f"  [ERROR] T1 ({epoch_tag}): {e}")
    
    # T2: Figure manifest
    try:
        path = generate_t2_figure_manifest(
            figure_paths,
            output_dir / 'grid_rdm__figure_manifest.csv',
            args.overwrite
        )
        print(f"  [OK] T2: {path.name}")
    except Exception as e:
        print(f"  [ERROR] T2: {e}")
    
    # Generate V2 figures (improved designs)
    print(f"\n[GENERATING V2 FIGURES]")
    v2_figure_paths = []
    
    # V2A: AUTO-only upgrade frequency (per pathway)
    for epoch_tag in ['2035_EB', '2035_BB']:
        epoch_key = epoch_tag.replace('2035_', '').lower()
        matrix_key = f'matrix_{epoch_key}'
        
        if matrix_key in data:
            try:
                path = plot_v2_auto_upgrade_frequency(
                    data[matrix_key],
                    epoch_tag,
                    output_dir / f'grid_rdm_{epoch_tag}__auto_upgrade_frequency_v2.png'
                )
                v2_figure_paths.append(path)
                print(f"  [OK] V2A ({epoch_tag}): {path.name}")
            except Exception as e:
                print(f"  [ERROR] V2A ({epoch_tag}): {e}")
    
    # V2A (combined): Combined EB vs BB
    if 'matrix_eb' in data and 'matrix_bb' in data:
        try:
            path = plot_v2_combined_auto_upgrade_frequency(
                data['matrix_eb'],
                data['matrix_bb'],
                output_dir / 'grid_rdm_2035_EB_vs_BB__auto_upgrade_frequency_v2.png'
            )
            v2_figure_paths.append(path)
            print(f"  [OK] V2A (combined): {path.name}")
        except Exception as e:
            print(f"  [ERROR] V2A (combined): {e}")
    
    # V2B: Strategy × upgrade heatmap
    for epoch_tag in ['2035_EB', '2035_BB']:
        epoch_key = epoch_tag.replace('2035_', '').lower()
        matrix_key = f'matrix_{epoch_key}'
        
        if matrix_key in data:
            try:
                path = plot_v2_strategy_upgrade_heatmap(
                    data[matrix_key],
                    epoch_tag,
                    output_dir / f'grid_rdm_{epoch_tag}__strategy_upgrade_heatmap_v2.png'
                )
                v2_figure_paths.append(path)
                print(f"  [OK] V2B ({epoch_tag}): {path.name}")
            except Exception as e:
                print(f"  [ERROR] V2B ({epoch_tag}): {e}")
    
    # V2C: AUTO upgrade vs headroom driver
    for epoch_tag in ['2035_EB', '2035_BB']:
        epoch_key = epoch_tag.replace('2035_', '').lower()
        matrix_key = f'matrix_{epoch_key}'
        
        if matrix_key in data:
            try:
                path = plot_v2_auto_upgrade_vs_headroom(
                    data[matrix_key],
                    epoch_tag,
                    output_dir / f'grid_rdm_{epoch_tag}__auto_upgrade_vs_headroom_v2.png'
                )
                v2_figure_paths.append(path)
                print(f"  [OK] V2C ({epoch_tag}): {path.name}")
            except Exception as e:
                print(f"  [ERROR] V2C ({epoch_tag}): {e}")
    
    # Create README for v2 figures
    readme_path = output_dir / 'README_figures_v2.txt'
    with open(readme_path, 'w') as f:
        f.write("""RDM Figures V2 - Improvements and Changes
================================================

This folder contains improved v2 versions of RDM figures with the following enhancements:

1. LABEL MAPPINGS
   - All strategy and upgrade labels have been standardized to thesis-friendly names
   - Strategy labels: "AUTO (min total cost)", "NO UPGRADE (shedding allowed)", etc.
   - Upgrade labels: "GXP upgrade opt1 (+21 MW)", "GXP upgrade opt2 (+32 MW, N-1)", etc.

2. TITLE CLEANUP
   - Removed bundle names (poc_20260105_release02) from all figure titles
   - Titles now use format: "2035_EB | Description" or "2035_BB | Description"

3. NEW V2 FIGURES

   V2A: AUTO-only upgrade frequency
   - Simple bar charts showing upgrade choice frequency under AUTO strategy only
   - More interpretable than forced-strategy stacked bars
   - Files: grid_rdm_2035_EB__auto_upgrade_frequency_v2.png
            grid_rdm_2035_BB__auto_upgrade_frequency_v2.png
            grid_rdm_2035_EB_vs_BB__auto_upgrade_frequency_v2.png (combined)

   V2B: Strategy × upgrade frequency heatmap
   - Heatmap showing percentage of futures for each strategy×upgrade combination
   - All strategies included (AUTO + forced + no-upgrade)
   - Annotated with percentages for readability
   - Files: grid_rdm_2035_EB__strategy_upgrade_heatmap_v2.png
            grid_rdm_2035_BB__strategy_upgrade_heatmap_v2.png

   V2C: AUTO upgrade vs headroom driver
   - Scatter plot showing relationship between headroom multiplier and selected upgrade
   - Points jittered slightly for better visibility
   - Files: grid_rdm_2035_EB__auto_upgrade_vs_headroom_v2.png
            grid_rdm_2035_BB__auto_upgrade_vs_headroom_v2.png

4. PRESERVATION OF ORIGINAL FIGURES
   - All original figures remain untouched
   - V2 figures use "_v2" suffix to avoid overwriting
   - Original figures can still be found without the "_v2" suffix

WHY THESE CHANGES?
- AUTO-only frequency charts are more interpretable for policy analysis
- Heatmaps provide clearer visualization of strategy×upgrade relationships
- Cleaner titles improve readability in thesis documents
- Standardized labels ensure consistency across all figures
""")
    print(f"  [OK] README: {readme_path.name}")
    
    # Final summary
    print(f"\n[SUMMARY]")
    print(f"  Output directory: {output_dir}")
    print(f"  Original figures generated: {len(figure_paths)}")
    print(f"  V2 figures generated: {len(v2_figure_paths)}")
    print(f"\n  V2 Figure files:")
    for path in sorted(v2_figure_paths):
        print(f"    {path.name}")
    
    print(f"\n[DONE]")


if __name__ == '__main__':
    main()
