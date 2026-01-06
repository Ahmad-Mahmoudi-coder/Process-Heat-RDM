"""
Compare Strategies 2035

Compare EB vs BB dispatch strategies by reading summary CSVs and producing
a comparison table for thesis reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path as PathlibPath

ROOT = PathlibPath(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
from typing import Optional

from src.path_utils import repo_root, resolve_path


def load_summary_row(summary_path: PathlibPath, epoch_tag: str) -> Optional[pd.Series]:
    """
    Load TOTAL row from a summary CSV.
    
    Args:
        summary_path: Path to summary CSV
        epoch_tag: Epoch tag for identification
        
    Returns:
        Series with summary row, or None if not found
    """
    if not summary_path.exists():
        print(f"[WARN] Summary file not found: {summary_path}")
        return None
    
    try:
        df = pd.read_csv(summary_path)
        
        if 'unit_id' not in df.columns:
            print(f"[WARN] Summary CSV {summary_path} missing 'unit_id' column")
            return None
        
        # Look for TOTAL row first
        total_row = df[df['unit_id'] == 'TOTAL']
        if len(total_row) > 0:
            return total_row.iloc[0]
        
        # Fallback to SYSTEM row
        system_row = df[df['unit_id'] == 'SYSTEM']
        if len(system_row) > 0:
            return system_row.iloc[0]
        
        # Fallback: first row
        print(f"[WARN] No TOTAL or SYSTEM row found in {summary_path}, using first row")
        return df.iloc[0]
    except Exception as e:
        print(f"[WARN] Failed to load summary from {summary_path}: {e}")
        return None


def find_summary_path(bundle_dir: PathlibPath, epoch_tag: str) -> Optional[PathlibPath]:
    """
    Find summary CSV path for an epoch.
    
    Looks for:
    - Output/runs/<bundle>/epoch<epoch_tag>/dispatch_prop_v2/site_dispatch_<epoch_tag>_summary.csv
    - Output/runs/<bundle>/epoch<epoch_tag>/dispatch_prop_v2/site_dispatch_<epoch_tag>_summary_opt.csv
    
    Args:
        bundle_dir: Bundle directory
        epoch_tag: Epoch tag
        
    Returns:
        Path to summary CSV, or None if not found
    """
    base_path = bundle_dir / f'epoch{epoch_tag}' / 'dispatch_prop_v2'
    
    # Try proportional mode first
    summary_path = base_path / f'site_dispatch_{epoch_tag}_summary.csv'
    if summary_path.exists():
        return summary_path
    
    # Try optimal mode
    summary_path = base_path / f'site_dispatch_{epoch_tag}_summary_opt.csv'
    if summary_path.exists():
        return summary_path
    
    return None


def extract_comparison_fields(row: pd.Series, epoch_tag: str) -> dict:
    """
    Extract key comparison fields from a summary row.
    
    Args:
        row: Summary row (Series)
        epoch_tag: Epoch tag
        
    Returns:
        Dictionary with comparison fields
    """
    result = {
        'epoch_tag': epoch_tag,
        'annual_total_cost_nzd': row.get('annual_total_cost_nzd', 0.0),
        'annual_electricity_cost_nzd': row.get('annual_electricity_cost_nzd', 0.0),
        'annual_fuel_cost_nzd': row.get('annual_fuel_cost_nzd', 0.0),
        'annual_unserved_MWh': row.get('unserved_MWh', 0.0),
        'annual_unserved_cost_nzd': row.get('unserved_cost_nzd', row.get('unserved_penalty_cost_nzd', 0.0))
    }
    
    # Fill NaNs with 0.0
    for key, value in result.items():
        if pd.isna(value):
            result[key] = 0.0
        elif key != 'epoch_tag':
            result[key] = float(value)
    
    return result


def compare_strategies(bundle_dir: PathlibPath, eb_tag: str, bb_tag: str) -> pd.DataFrame:
    """
    Compare EB vs BB strategies by loading their summary CSVs.
    
    Args:
        bundle_dir: Bundle directory
        eb_tag: EB epoch tag (e.g., "2035_EB")
        bb_tag: BB epoch tag (e.g., "2035_BB")
        
    Returns:
        DataFrame with comparison (one row per strategy)
    """
    # Find summary paths
    eb_path = find_summary_path(bundle_dir, eb_tag)
    bb_path = find_summary_path(bundle_dir, bb_tag)
    
    if eb_path is None:
        raise FileNotFoundError(f"EB summary not found for {eb_tag}. Expected: {bundle_dir / f'epoch{eb_tag}' / 'dispatch_prop_v2' / f'site_dispatch_{eb_tag}_summary.csv'}")
    
    if bb_path is None:
        raise FileNotFoundError(f"BB summary not found for {bb_tag}. Expected: {bundle_dir / f'epoch{bb_tag}' / 'dispatch_prop_v2' / f'site_dispatch_{bb_tag}_summary.csv'}")
    
    # Load summary rows
    print(f"Loading EB summary from {eb_path}...")
    eb_row = load_summary_row(eb_path, eb_tag)
    if eb_row is None:
        raise ValueError(f"Failed to load EB summary row from {eb_path}")
    
    print(f"Loading BB summary from {bb_path}...")
    bb_row = load_summary_row(bb_path, bb_tag)
    if bb_row is None:
        raise ValueError(f"Failed to load BB summary row from {bb_path}")
    
    # Extract comparison fields
    eb_data = extract_comparison_fields(eb_row, eb_tag)
    bb_data = extract_comparison_fields(bb_row, bb_tag)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([eb_data, bb_data])
    
    return comparison_df


def main():
    """CLI entrypoint for strategy comparison."""
    parser = argparse.ArgumentParser(
        description='Compare EB vs BB dispatch strategies for 2035'
    )
    parser.add_argument('--output-root', type=str, default='Output',
                       help='Output root directory (default: Output)')
    parser.add_argument('--bundle', type=str, required=True,
                       help='Bundle name (e.g., full2035_20251225_170112)')
    parser.add_argument('--eb', type=str, default='2035_EB',
                       help='EB epoch tag (default: 2035_EB)')
    parser.add_argument('--bb', type=str, default='2035_BB',
                       help='BB epoch tag (default: 2035_BB)')
    
    args = parser.parse_args()
    
    # Resolve paths
    ROOT = repo_root()
    output_root = PathlibPath(resolve_path(args.output_root))
    bundle_dir = output_root / 'runs' / args.bundle
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    
    # Compare strategies
    print(f"Comparing strategies: {args.eb} vs {args.bb}")
    comparison_df = compare_strategies(bundle_dir, args.eb, args.bb)
    
    # Create results pack directory
    results_pack_dir = bundle_dir / '_results_pack'
    results_pack_dir.mkdir(parents=True, exist_ok=True)
    
    # Write comparison CSV
    comparison_path = results_pack_dir / 'compare_2035_EB_vs_BB.csv'
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"\n[OK] Comparison written to: {comparison_path}")
    print("\nComparison Summary:")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)


if __name__ == '__main__':
    main()

