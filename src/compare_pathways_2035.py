"""
Compare Pathways 2035

Compare EB vs BB dispatch pathways by reading summary CSVs and producing
a comparison table for thesis reporting.

Usage:
    python -m src.compare_pathways_2035 --bundle <bundle_id> --eb-run <relative_path> --bb-run <relative_path>
    
Default paths (if --eb-run and --bb-run not specified):
    --eb-run: epoch2035_EB/dispatch_prop_v2_capfix1
    --bb-run: epoch2035_BB/dispatch_prop_v2_capfix1
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


def load_summary_row(summary_path: PathlibPath) -> Optional[pd.Series]:
    """
    Load TOTAL row from a summary CSV.
    
    Args:
        summary_path: Path to summary CSV
        
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


def find_summary_path(bundle_dir: PathlibPath, relative_run_path: str, epoch_tag: str) -> Optional[PathlibPath]:
    """
    Find summary CSV path for a run.
    
    Args:
        bundle_dir: Bundle directory
        relative_run_path: Relative path from bundle (e.g., 'epoch2035_EB/dispatch_prop_v2_capfix1')
        epoch_tag: Epoch tag (e.g., '2035_EB')
        
    Returns:
        Path to summary CSV, or None if not found
    """
    base_path = bundle_dir / relative_run_path
    
    # Try proportional mode first
    summary_path = base_path / f'site_dispatch_{epoch_tag}_summary.csv'
    if summary_path.exists():
        return summary_path
    
    # Try optimal mode
    summary_path = base_path / f'site_dispatch_{epoch_tag}_summary_opt.csv'
    if summary_path.exists():
        return summary_path
    
    return None


def extract_all_kpis(row: pd.Series) -> dict:
    """
    Extract all KPIs from a summary row.
    
    Prefers effective electricity columns if available (for EB units when explicit fields are zero).
    
    Args:
        row: Summary row (Series)
        
    Returns:
        Dictionary with all numeric KPIs
    """
    kpis = {}
    
    # Key metrics (required)
    # For electricity metrics, prefer effective columns if available
    key_metrics = [
        'annual_total_cost_nzd',
        'avg_cost_nzd_per_MWh_heat',
        'annual_co2_tonnes',
        'annual_carbon_cost_nzd',
        'annual_fuel_cost_nzd',
        'annual_unserved_MWh',
        'annual_unserved_cost_nzd',
        'unserved_avg_cost_nzd_per_MWh',
    ]
    
    # Extract key metrics
    for metric in key_metrics:
        value = row.get(metric, None)
        if value is not None and not pd.isna(value):
            try:
                kpis[metric] = float(value)
            except (ValueError, TypeError):
                kpis[metric] = 0.0
        else:
            kpis[metric] = 0.0
    
    # Extract electricity metrics with preference for effective columns
    # Prefer effective columns if available and non-zero, otherwise fallback to regular columns
    electricity_metrics = {
        'annual_electricity_MWh': ['annual_electricity_MWh_effective', 'annual_electricity_MWh'],
        'avg_electricity_tariff_nzd_per_MWh': ['avg_electricity_tariff_nzd_per_MWh_effective', 'avg_electricity_tariff_nzd_per_MWh'],
        'annual_electricity_cost_nzd': ['annual_electricity_cost_nzd_effective', 'annual_electricity_cost_nzd'],
    }
    
    for metric_name, column_preferences in electricity_metrics.items():
        value = None
        for col in column_preferences:
            if col in row.index:
                val = row.get(col, None)
                if val is not None and not pd.isna(val):
                    try:
                        float_val = float(val)
                        if float_val > 0 or value is None:  # Prefer non-zero, or first available
                            value = float_val
                    except (ValueError, TypeError):
                        pass
        
        kpis[metric_name] = value if value is not None else 0.0
    
    # Extract any other numeric columns (additional KPIs)
    for col in row.index:
        if col not in ['unit_id'] and col not in key_metrics:
            value = row.get(col, None)
            if value is not None and not pd.isna(value):
                try:
                    float_val = float(value)
                    kpis[col] = float_val
                except (ValueError, TypeError):
                    pass  # Skip non-numeric columns
    
    return kpis


def compare_pathways(bundle_dir: PathlibPath, eb_run_path: str, bb_run_path: str, 
                     eb_tag: str = '2035_EB', bb_tag: str = '2035_BB') -> pd.DataFrame:
    """
    Compare EB vs BB pathways by loading their summary CSVs.
    
    Args:
        bundle_dir: Bundle directory
        eb_run_path: Relative path to EB run (e.g., 'epoch2035_EB/dispatch_prop_v2_capfix1')
        bb_run_path: Relative path to BB run (e.g., 'epoch2035_BB/dispatch_prop_v2_capfix1')
        eb_tag: EB epoch tag (default: "2035_EB")
        bb_tag: BB epoch tag (default: "2035_BB")
        
    Returns:
        DataFrame with comparison (pathway, metric, value, delta_BB_minus_EB, delta_EB_minus_BB)
    """
    # Find summary paths
    eb_path = find_summary_path(bundle_dir, eb_run_path, eb_tag)
    bb_path = find_summary_path(bundle_dir, bb_run_path, bb_tag)
    
    if eb_path is None:
        raise FileNotFoundError(
            f"EB summary not found. Expected: "
            f"{bundle_dir / eb_run_path / f'site_dispatch_{eb_tag}_summary.csv'}"
        )
    
    if bb_path is None:
        raise FileNotFoundError(
            f"BB summary not found. Expected: "
            f"{bundle_dir / bb_run_path / f'site_dispatch_{bb_tag}_summary.csv'}"
        )
    
    # Load summary rows
    print(f"Loading EB summary from {eb_path}...")
    eb_row = load_summary_row(eb_path)
    if eb_row is None:
        raise ValueError(f"Failed to load EB summary row from {eb_path}")
    
    print(f"Loading BB summary from {bb_path}...")
    bb_row = load_summary_row(bb_path)
    if bb_row is None:
        raise ValueError(f"Failed to load BB summary row from {bb_path}")
    
    # Extract KPIs
    eb_kpis = extract_all_kpis(eb_row)
    bb_kpis = extract_all_kpis(bb_row)
    
    # Get all unique metrics
    all_metrics = set(eb_kpis.keys()) | set(bb_kpis.keys())
    
    # Build comparison DataFrame
    comparison_data = []
    
    # Add EB row
    for metric in sorted(all_metrics):
        eb_value = eb_kpis.get(metric, 0.0)
        comparison_data.append({
            'pathway': 'EB',
            'metric': metric,
            'value': eb_value
        })
    
    # Add BB row
    for metric in sorted(all_metrics):
        bb_value = bb_kpis.get(metric, 0.0)
        comparison_data.append({
            'pathway': 'BB',
            'metric': metric,
            'value': bb_value
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add delta columns
    eb_values = comparison_df[comparison_df['pathway'] == 'EB'].set_index('metric')['value']
    bb_values = comparison_df[comparison_df['pathway'] == 'BB'].set_index('metric')['value']
    
    comparison_df['delta_BB_minus_EB'] = 0.0
    comparison_df['delta_EB_minus_BB'] = 0.0
    
    for idx, row in comparison_df.iterrows():
        metric = row['metric']
        if row['pathway'] == 'EB':
            bb_val = bb_values.get(metric, 0.0)
            eb_val = row['value']
            comparison_df.at[idx, 'delta_BB_minus_EB'] = bb_val - eb_val
            comparison_df.at[idx, 'delta_EB_minus_BB'] = eb_val - bb_val
        elif row['pathway'] == 'BB':
            eb_val = eb_values.get(metric, 0.0)
            bb_val = row['value']
            comparison_df.at[idx, 'delta_BB_minus_EB'] = bb_val - eb_val
            comparison_df.at[idx, 'delta_EB_minus_BB'] = eb_val - bb_val
    
    # Reorder columns
    comparison_df = comparison_df[['pathway', 'metric', 'value', 'delta_BB_minus_EB', 'delta_EB_minus_BB']]
    
    return comparison_df


def validate_total_rows(eb_path: PathlibPath, bb_path: PathlibPath) -> bool:
    """
    Self-test: validate that TOTAL rows exist in both summaries.
    
    Args:
        eb_path: Path to EB summary CSV
        bb_path: Path to BB summary CSV
        
    Returns:
        True if both have TOTAL rows, False otherwise
    """
    eb_row = load_summary_row(eb_path)
    bb_row = load_summary_row(bb_path)
    
    if eb_row is None:
        print(f"[TEST FAIL] EB summary missing TOTAL row: {eb_path}")
        return False
    
    if bb_row is None:
        print(f"[TEST FAIL] BB summary missing TOTAL row: {bb_path}")
        return False
    
    print("[TEST PASS] Both summaries have TOTAL rows")
    return True


def main():
    """CLI entrypoint for pathway comparison."""
    parser = argparse.ArgumentParser(
        description='Compare EB vs BB dispatch pathways for 2035'
    )
    parser.add_argument('--output-root', type=str, default='Output',
                       help='Output root directory (default: Output)')
    parser.add_argument('--bundle', type=str, required=True,
                       help='Bundle name (e.g., poc_20260105_115401)')
    parser.add_argument('--eb-run', type=str, default='epoch2035_EB/dispatch_prop_v2_capfix1',
                       help='Relative path to EB run (default: epoch2035_EB/dispatch_prop_v2_capfix1)')
    parser.add_argument('--bb-run', type=str, default='epoch2035_BB/dispatch_prop_v2_capfix1',
                       help='Relative path to BB run (default: epoch2035_BB/dispatch_prop_v2_capfix1)')
    parser.add_argument('--eb-tag', type=str, default='2035_EB',
                       help='EB epoch tag (default: 2035_EB)')
    parser.add_argument('--bb-tag', type=str, default='2035_BB',
                       help='BB epoch tag (default: 2035_BB)')
    
    args = parser.parse_args()
    
    # Resolve paths
    ROOT = repo_root()
    output_root = PathlibPath(resolve_path(args.output_root))
    bundle_dir = output_root / 'runs' / args.bundle
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    
    # Compare pathways
    print(f"Comparing pathways: {args.eb_tag} vs {args.bb_tag}")
    comparison_df = compare_pathways(bundle_dir, args.eb_run, args.bb_run, args.eb_tag, args.bb_tag)
    
    # Self-test: validate TOTAL rows
    eb_path = find_summary_path(bundle_dir, args.eb_run, args.eb_tag)
    bb_path = find_summary_path(bundle_dir, args.bb_run, args.bb_tag)
    if eb_path and bb_path:
        validate_total_rows(eb_path, bb_path)
    
    # Write comparison CSV (directly in bundle directory, not in subfolder)
    csv_path = bundle_dir / 'compare_pathways_2035_EB_vs_BB.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Comparison CSV written to: {csv_path}")
    
    # Print summary
    print("\nComparison Summary:")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)


if __name__ == '__main__':
    main()
