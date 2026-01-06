"""
Run RDM 2035

Runner module for RDM screening of 2035 EB and BB pathways using paired futures.

Usage:
    python -m src.run_rdm_2035 --bundle poc_20260105_115401 --run-id dispatch_prop_v2_capfix1 --epoch-tag 2035_EB
    python -m src.run_rdm_2035 --bundle poc_20260105_115401 --run-id dispatch_prop_v2_capfix1 --epoch-tag 2035_BB
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

from src.path_utils import repo_root, resolve_path, canonical_output_path
from src.gxp_rdm_screen import run_rdm_screen, run_rdm_matrix


def find_incremental_csv(bundle_dir: PathlibPath, epoch_tag: str, run_id: str, output_root: str = "Output") -> PathlibPath:
    """
    Find incremental electricity CSV for an epoch.
    
    Checks paths in order:
    1. Canonical: epoch<epoch_tag>/dispatch/<run_id>/signals/incremental_electricity_MW_<epoch_tag>.csv
    2. Legacy: epoch<epoch_tag>/<run_id>/signals/incremental_electricity_MW_<epoch_tag>.csv
    
    Args:
        bundle_dir: Bundle directory
        epoch_tag: Epoch tag (e.g., '2035_EB')
        run_id: Run ID (e.g., 'dispatch_prop_v2_capfix1')
        output_root: Output root directory name (default: "Output")
        
    Returns:
        Path to incremental CSV
        
    Raises:
        FileNotFoundError: If neither path exists, lists both attempted paths
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
    
    # Neither exists - raise error with both paths
    raise FileNotFoundError(
        f"Incremental electricity CSV not found for {epoch_tag}.\n"
        f"Attempted canonical path: {canonical_path}\n"
        f"Attempted legacy path: {legacy_path}"
    )


def validate_paired_futures(eb_summary_path: PathlibPath, bb_summary_path: PathlibPath) -> bool:
    """
    Self-test: validate that EB and BB use identical future_id sets (paired futures).
    
    Args:
        eb_summary_path: Path to EB RDM summary CSV
        bb_summary_path: Path to BB RDM summary CSV
        
    Returns:
        True if futures are paired, False otherwise
    """
    eb_df = pd.read_csv(eb_summary_path)
    bb_df = pd.read_csv(bb_summary_path)
    
    eb_futures = set(eb_df['future_id'].unique())
    bb_futures = set(bb_df['future_id'].unique())
    
    if eb_futures != bb_futures:
        print(f"[TEST FAIL] Future ID mismatch: EB has {len(eb_futures)} futures, BB has {len(bb_futures)} futures")
        print(f"[TEST FAIL] EB futures: {sorted(eb_futures)[:10]}...")
        print(f"[TEST FAIL] BB futures: {sorted(bb_futures)[:10]}...")
        return False
    
    print(f"[TEST PASS] Paired futures validated: {len(eb_futures)} futures match between EB and BB")
    return True


def create_comparison_summary(eb_summary_path: PathlibPath, bb_summary_path: PathlibPath, 
                              output_path: PathlibPath) -> None:
    """
    Create side-by-side comparison of EB and BB RDM summaries.
    
    Args:
        eb_summary_path: Path to EB RDM summary CSV
        bb_summary_path: Path to BB RDM summary CSV
        output_path: Path to write comparison CSV
    """
    eb_df = pd.read_csv(eb_summary_path)
    bb_df = pd.read_csv(bb_summary_path)
    
    # Validate paired futures
    validate_paired_futures(eb_summary_path, bb_summary_path)
    
    # Ensure both have same future_id set
    eb_futures = set(eb_df['future_id'].unique())
    bb_futures = set(bb_df['future_id'].unique())
    
    if eb_futures != bb_futures:
        print(f"[WARN] Future ID mismatch: EB has {len(eb_futures)} futures, BB has {len(bb_futures)} futures")
        common_futures = eb_futures & bb_futures
        print(f"[WARN] Using {len(common_futures)} common futures")
        eb_df = eb_df[eb_df['future_id'].isin(common_futures)]
        bb_df = bb_df[bb_df['future_id'].isin(common_futures)]
    
    # Merge on future_id
    comparison_df = pd.merge(
        eb_df,
        bb_df,
        on='future_id',
        suffixes=('_EB', '_BB'),
        how='inner'
    )
    
    # Reorder columns: future_id first, then key metrics side-by-side
    key_metrics = [
        'selected_upgrade_name', 'selected_capacity_MW', 'annualised_upgrade_cost_nzd',
        'annual_incremental_MWh', 'annual_shed_MWh', 'shed_fraction',
        'annual_shed_cost_nzd', 'total_cost_nzd', 'n_hours_binding', 'unserved_peak_MW'
    ]
    
    cols = ['future_id']
    for metric in key_metrics:
        eb_col = f'{metric}_EB'
        bb_col = f'{metric}_BB'
        if eb_col in comparison_df.columns and bb_col in comparison_df.columns:
            cols.extend([eb_col, bb_col])
    
    # Add any remaining columns
    for col in comparison_df.columns:
        if col not in cols:
            cols.append(col)
    
    comparison_df = comparison_df[[col for col in cols if col in comparison_df.columns]]
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    print(f"[OK] Comparison summary written to: {output_path}")


def self_test_paths(bundle_dir: PathlibPath, run_id: str, output_root: str = "Output") -> None:
    """
    Self-test function to print resolved paths for EB and BB when files exist.
    
    Args:
        bundle_dir: Bundle directory
        run_id: Run ID (e.g., 'dispatch_prop_v2_capfix1')
        output_root: Output root directory name (default: "Output")
    """
    print("=" * 60)
    print("Self-test: Incremental electricity CSV path resolution")
    print("=" * 60)
    
    for epoch_tag in ['2035_EB', '2035_BB']:
        print(f"\nTesting {epoch_tag}:")
        try:
            resolved_path = find_incremental_csv(bundle_dir, epoch_tag, run_id, output_root)
            print(f"  [OK] Resolved path: {resolved_path}")
            print(f"  [OK] File exists: {resolved_path.exists()}")
            if resolved_path.exists():
                file_size = resolved_path.stat().st_size
                print(f"  [OK] File size: {file_size:,} bytes")
        except FileNotFoundError as e:
            print(f"  [FAIL] {e}")
    
    print("\n" + "=" * 60)


def main():
    """CLI entrypoint for RDM 2035 runner."""
    parser = argparse.ArgumentParser(
        description='Run RDM screening for 2035 EB or BB pathway'
    )
    parser.add_argument('--output-root', type=str, default='Output',
                       help='Output root directory (default: Output)')
    parser.add_argument('--bundle', type=str, required=True,
                       help='Bundle name (e.g., poc_20260105_115401)')
    parser.add_argument('--run-id', type=str, required=True,
                       help='Run ID (e.g., dispatch_prop_v2_capfix1)')
    parser.add_argument('--epoch-tag', type=str, required=True,
                       help='Epoch tag (2035_EB or 2035_BB)')
    parser.add_argument('--futures-csv', type=str, default='Input/rdm/futures_2035.csv',
                       help='Path to futures CSV (default: Input/rdm/futures_2035.csv)')
    parser.add_argument('--upgrade-toml', type=str, default='Input/configs/grid_upgrades_southland_edendale.toml',
                       help='Path to upgrade menu TOML (default: Input/configs/grid_upgrades_southland_edendale.toml)')
    parser.add_argument('--headroom-csv', type=str, default=None,
                       help='Optional path to headroom CSV (if missing, generates PoC headroom)')
    parser.add_argument('--create-comparison', action='store_true',
                       help='After running, create comparison CSV if both EB and BB summaries exist')
    parser.add_argument('--self-test', action='store_true',
                       help='Run self-test to print resolved paths for EB and BB, then exit')
    
    args = parser.parse_args()
    
    # Resolve paths
    ROOT = repo_root()
    output_root = PathlibPath(resolve_path(args.output_root))
    bundle_dir = output_root / 'runs' / args.bundle
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    
    # Run self-test if requested
    if args.self_test:
        self_test_paths(bundle_dir, args.run_id, args.output_root)
        return
    
    # Validate epoch tag
    if args.epoch_tag not in ['2035_EB', '2035_BB']:
        print(f"[WARN] Epoch tag '{args.epoch_tag}' is not 2035_EB or 2035_BB, proceeding anyway...")
    
    # Find incremental electricity CSV
    try:
        incremental_csv = find_incremental_csv(bundle_dir, args.epoch_tag, args.run_id, args.output_root)
        print(f"[OK] Found incremental electricity CSV: {incremental_csv}")
    except FileNotFoundError as e:
        raise  # Re-raise with the detailed error message
    
    # Resolve other paths
    futures_csv = resolve_path(args.futures_csv)
    upgrade_toml = resolve_path(args.upgrade_toml)
    headroom_csv = resolve_path(args.headroom_csv) if args.headroom_csv else None
    
    # Run RDM screening
    print(f"[RDM] Running RDM screening for {args.epoch_tag}...")
    
    results_df = run_rdm_screen(
        incremental_csv,
        futures_csv,
        upgrade_toml,
        headroom_csv,
        args.epoch_tag,
        strategy_id='S_AUTO',
        strategy_label='Auto-select upgrade (min cost)'
    )
    
    # Write output
    rdm_dir = bundle_dir / 'rdm'
    rdm_dir.mkdir(parents=True, exist_ok=True)
    output_csv = rdm_dir / f'rdm_summary_{args.epoch_tag}.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\n[OK] RDM summary written to: {output_csv}")
    
    # Generate RDM matrix (all strategies for all futures)
    print(f"\n[RDM Matrix] Generating matrix for {args.epoch_tag}...")
    matrix_df = run_rdm_matrix(
        incremental_csv,
        futures_csv,
        upgrade_toml,
        headroom_csv,
        args.epoch_tag
    )
    
    matrix_csv = rdm_dir / f'rdm_matrix_{args.epoch_tag}.csv'
    matrix_df.to_csv(matrix_csv, index=False)
    print(f"[OK] RDM matrix written to: {matrix_csv}")
    
    # Create comparison if requested and both summaries exist
    if args.create_comparison:
        eb_summary = rdm_dir / 'rdm_summary_2035_EB.csv'
        bb_summary = rdm_dir / 'rdm_summary_2035_BB.csv'
        
        if eb_summary.exists() and bb_summary.exists():
            comparison_path = rdm_dir / 'rdm_compare_2035_EB_vs_BB.csv'
            create_comparison_summary(eb_summary, bb_summary, comparison_path)
        else:
            print(f"[INFO] Comparison not created: missing summaries (EB: {eb_summary.exists()}, BB: {bb_summary.exists()})")
    
    # Self-test: validate paired futures if both summaries exist
    eb_summary = rdm_dir / 'rdm_summary_2035_EB.csv'
    bb_summary = rdm_dir / 'rdm_summary_2035_BB.csv'
    if eb_summary.exists() and bb_summary.exists():
        validate_paired_futures(eb_summary, bb_summary)


if __name__ == '__main__':
    main()

