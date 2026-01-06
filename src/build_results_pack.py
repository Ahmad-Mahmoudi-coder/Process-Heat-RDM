"""
Results Pack Builder

Aggregates per-epoch dispatch summaries and produces a thesis-ready results pack
with summary table and key figures.
"""

from __future__ import annotations

import sys
from pathlib import Path as PathlibPath

ROOT = PathlibPath(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Optional

from src.path_utils import repo_root, resolve_path


def load_epoch_summary(bundle_dir: PathlibPath, epoch_tag: str) -> Optional[pd.DataFrame]:
    """
    Load summary CSV for a single epoch.
    
    Looks for:
    Output/runs/<bundle>/epoch<epoch_tag>/dispatch_prop_v2/site_dispatch_<epoch_tag>_summary.csv
    
    Args:
        bundle_dir: Bundle directory (e.g., Output/runs/<bundle>)
        epoch_tag: Epoch tag (e.g., "2035_EB")
        
    Returns:
        DataFrame with summary row (TOTAL row extracted), or None if file not found
    """
    # Try proportional mode summary first
    summary_path = bundle_dir / f'epoch{epoch_tag}' / 'dispatch_prop_v2' / f'site_dispatch_{epoch_tag}_summary.csv'
    
    # If not found, try optimal mode summary
    if not summary_path.exists():
        summary_path = bundle_dir / f'epoch{epoch_tag}' / 'dispatch_prop_v2' / f'site_dispatch_{epoch_tag}_summary_opt.csv'
    
    if not summary_path.exists():
        return None
    
    try:
        df = pd.read_csv(summary_path)
        
        # Extract TOTAL row (preferred) or SYSTEM row (fallback)
        if 'unit_id' not in df.columns:
            print(f"[WARN] Summary CSV {summary_path} missing 'unit_id' column, skipping")
            return None
        
        # Look for TOTAL row first
        total_row = df[df['unit_id'] == 'TOTAL']
        if len(total_row) > 0:
            result = total_row.iloc[0:1].copy()
        else:
            # Fallback to SYSTEM row
            system_row = df[df['unit_id'] == 'SYSTEM']
            if len(system_row) > 0:
                result = system_row.iloc[0:1].copy()
            else:
                # Fallback: use first row (shouldn't happen, but be robust)
                print(f"[WARN] No TOTAL or SYSTEM row found in {summary_path}, using first row")
                result = df.iloc[0:1].copy()
        
        # Add epoch_tag column
        result['epoch_tag'] = epoch_tag
        
        return result
    except Exception as e:
        print(f"[WARN] Failed to load summary from {summary_path}: {e}")
        return None


def build_summary_table(bundle_dir: PathlibPath, epoch_tags: List[str]) -> pd.DataFrame:
    """
    Build aggregated summary table from all epoch summaries.
    
    Args:
        bundle_dir: Bundle directory
        epoch_tags: List of epoch tags to process
        
    Returns:
        DataFrame with one row per epoch_tag
    """
    all_summaries = []
    
    for epoch_tag in epoch_tags:
        summary = load_epoch_summary(bundle_dir, epoch_tag)
        if summary is not None:
            all_summaries.append(summary)
        else:
            print(f"[WARN] Summary not found for epoch {epoch_tag}, skipping")
    
    if len(all_summaries) == 0:
        raise ValueError("No summary files found for any epoch")
    
    # Concatenate all summaries
    combined = pd.concat(all_summaries, ignore_index=True)
    
    # Ensure epoch_tag is first column (after unit_id if present)
    cols = list(combined.columns)
    if 'unit_id' in cols:
        cols.remove('unit_id')
    if 'epoch_tag' in cols:
        cols.remove('epoch_tag')
    
    # Reorder: unit_id (if present), epoch_tag, then rest
    final_cols = []
    if 'unit_id' in combined.columns:
        final_cols.append('unit_id')
    final_cols.append('epoch_tag')
    final_cols.extend([c for c in cols if c not in ['unit_id', 'epoch_tag']])
    
    return combined[final_cols]


def plot_total_cost_by_epoch(summary_df: pd.DataFrame, output_path: PathlibPath) -> bool:
    """
    Plot annual_total_cost_nzd by epoch_tag.
    
    Args:
        summary_df: Summary DataFrame with epoch_tag and annual_total_cost_nzd
        output_path: Path to save figure
        
    Returns:
        True if plot created, False if column missing
    """
    if 'annual_total_cost_nzd' not in summary_df.columns:
        print(f"[WARN] Column 'annual_total_cost_nzd' not found, skipping total_cost_by_epoch.png")
        return False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    epochs = summary_df['epoch_tag'].values
    costs = summary_df['annual_total_cost_nzd'].values / 1e6  # Convert to millions
    
    # Create bar plot
    bars = ax.bar(range(len(epochs)), costs, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.1f}M',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Annual Total Cost (Million NZD)', fontsize=12)
    ax.set_title('Annual Total Cost by Epoch', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[OK] Created {output_path.name}")
    return True


def plot_unserved_by_epoch(summary_df: pd.DataFrame, output_path: PathlibPath) -> bool:
    """
    Plot annual_unserved_MWh by epoch_tag.
    
    Args:
        summary_df: Summary DataFrame with epoch_tag and annual_unserved_MWh
        output_path: Path to save figure
        
    Returns:
        True if plot created, False if column missing
    """
    if 'unserved_MWh' not in summary_df.columns:
        print(f"[WARN] Column 'unserved_MWh' not found, skipping unserved_MWh_by_epoch.png")
        return False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    epochs = summary_df['epoch_tag'].values
    unserved = summary_df['unserved_MWh'].values / 1000.0  # Convert to GWh
    
    # Create bar plot
    bars = ax.bar(range(len(epochs)), unserved, color='coral', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, unserved)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Annual Unserved Energy (GWh)', fontsize=12)
    ax.set_title('Annual Unserved Energy by Epoch', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[OK] Created {output_path.name}")
    return True


def plot_electricity_by_epoch(summary_df: pd.DataFrame, output_path: PathlibPath) -> bool:
    """
    Plot annual electricity consumption by epoch_tag.
    
    Looks for annual_electricity_MWh or annual_fuel_MWh (for EB units).
    
    Args:
        summary_df: Summary DataFrame with epoch_tag
        output_path: Path to save figure
        
    Returns:
        True if plot created, False if column missing
    """
    # Try multiple column names
    elec_col = None
    for col in ['annual_electricity_MWh', 'annual_fuel_MWh']:
        if col in summary_df.columns:
            elec_col = col
            break
    
    if elec_col is None:
        print(f"[WARN] No electricity column found (tried: annual_electricity_MWh, annual_fuel_MWh), skipping electricity_MWh_by_epoch.png")
        return False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    epochs = summary_df['epoch_tag'].values
    electricity = summary_df[elec_col].values / 1000.0  # Convert to GWh
    
    # Create bar plot
    bars = ax.bar(range(len(epochs)), electricity, color='mediumseagreen', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, electricity)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Annual Electricity (GWh)', fontsize=12)
    ax.set_title('Annual Electricity Consumption by Epoch', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[OK] Created {output_path.name}")
    return True


def plot_fuel_by_epoch(summary_df: pd.DataFrame, output_path: PathlibPath) -> bool:
    """
    Plot annual fuel consumption by epoch_tag.
    
    Looks for biomass_GJ or annual_fuel_MWh (converted to GJ).
    
    Args:
        summary_df: Summary DataFrame with epoch_tag
        output_path: Path to save figure
        
    Returns:
        True if plot created, False if column missing
    """
    # Try multiple column names
    fuel_col = None
    fuel_data = None
    
    # Prefer biomass_GJ if available
    if 'biomass_GJ' in summary_df.columns:
        fuel_col = 'biomass_GJ'
        fuel_data = summary_df[fuel_col].values / 1000.0  # Convert to TJ
        unit_label = 'Annual Biomass Fuel (TJ)'
    elif 'annual_fuel_MWh' in summary_df.columns:
        fuel_col = 'annual_fuel_MWh'
        # Convert MWh to GJ: 1 MWh = 3.6 GJ
        fuel_data = summary_df[fuel_col].values * 3.6 / 1000.0  # Convert to TJ
        unit_label = 'Annual Fuel (TJ, from MWh)'
    else:
        print(f"[WARN] No fuel column found (tried: biomass_GJ, annual_fuel_MWh), skipping fuel_GJ_by_epoch.png")
        return False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    epochs = summary_df['epoch_tag'].values
    
    # Create bar plot
    bars = ax.bar(range(len(epochs)), fuel_data, color='sienna', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, fuel_data)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(unit_label, fontsize=12)
    ax.set_title('Annual Fuel Consumption by Epoch', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[OK] Created {output_path.name}")
    return True


def generate_figures(summary_df: pd.DataFrame, figures_dir: PathlibPath) -> None:
    """
    Generate all figures for the results pack.
    
    Args:
        summary_df: Summary DataFrame
        figures_dir: Directory to save figures
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate each figure (skip with warning if column missing)
    plot_total_cost_by_epoch(summary_df, figures_dir / 'total_cost_by_epoch.png')
    plot_unserved_by_epoch(summary_df, figures_dir / 'unserved_MWh_by_epoch.png')
    plot_electricity_by_epoch(summary_df, figures_dir / 'electricity_MWh_by_epoch.png')
    plot_fuel_by_epoch(summary_df, figures_dir / 'fuel_GJ_by_epoch.png')


def main():
    """CLI entrypoint for results pack builder."""
    parser = argparse.ArgumentParser(
        description='Build thesis-ready results pack from per-epoch dispatch summaries'
    )
    parser.add_argument('--output-root', type=str, default='Output',
                       help='Output root directory (default: Output)')
    parser.add_argument('--bundle', type=str, required=True,
                       help='Bundle name (e.g., full2035_20251225_170112)')
    parser.add_argument('--epoch-tags', type=str, required=True,
                       help='Comma-separated list of epoch tags (e.g., "2020,2025,2028,2035_EB,2035_BB")')
    
    args = parser.parse_args()
    
    # Resolve paths
    ROOT = repo_root()
    output_root = PathlibPath(resolve_path(args.output_root))
    bundle_dir = output_root / 'runs' / args.bundle
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    
    # Parse epoch tags
    epoch_tags = [tag.strip() for tag in args.epoch_tags.split(',')]
    print(f"Processing {len(epoch_tags)} epochs: {', '.join(epoch_tags)}")
    
    # Build summary table
    print("\nBuilding summary table...")
    summary_df = build_summary_table(bundle_dir, epoch_tags)
    print(f"[OK] Aggregated {len(summary_df)} epoch summaries")
    
    # Create results pack directory
    results_pack_dir = bundle_dir / '_results_pack'
    results_pack_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_pack_dir / 'figures'
    
    # Write summary table
    summary_table_path = results_pack_dir / 'run_summary_table.csv'
    summary_df.to_csv(summary_table_path, index=False)
    print(f"[OK] Wrote summary table to {summary_table_path}")
    
    # Generate figures
    print("\nGenerating figures...")
    generate_figures(summary_df, figures_dir)
    
    print("\n" + "="*80)
    print("Results Pack Complete")
    print("="*80)
    print(f"Summary table: {summary_table_path}")
    print(f"Figures: {figures_dir}")
    print(f"Epochs processed: {len(summary_df)}/{len(epoch_tags)}")
    print("="*80)


if __name__ == '__main__':
    main()

