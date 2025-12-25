"""
Lightweight audit script to verify maintenance windows are correctly applied in dispatch outputs.

Usage:
    python scripts/audit_maintenance.py --epoch 2020 --run "<path-to-run-folder>" --mode opt
"""

# Bootstrap: allow `python .\scripts\script.py` (adds repo root to sys.path)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import argparse
from src.path_utils import repo_root
from src.maintenance_utils import load_maintenance_windows
from src.time_utils import parse_any_timestamp


def audit_maintenance(epoch: int, run_path: Path, mode: str = 'opt') -> bool:
    """
    Audit maintenance windows in dispatch outputs.
    
    Args:
        epoch: Epoch year
        run_path: Path to run folder (e.g., Output/runs/<run_id>)
        mode: Dispatch mode ('opt' for optimal_subset/lp, 'prop' for proportional)
        
    Returns:
        True if all checks pass, False otherwise
    """
    # Determine dispatch CSV filename
    if mode == 'opt':
        dispatch_csv = run_path / f'site_dispatch_{epoch}_long_costed_opt.csv'
    elif mode == 'prop':
        dispatch_csv = run_path / f'site_dispatch_{epoch}_long.csv'
    else:
        print(f"[ERROR] Unknown mode: {mode}. Use 'opt' or 'prop'")
        return False
    
    if not dispatch_csv.exists():
        print(f"[ERROR] Dispatch CSV not found: {dispatch_csv}")
        return False
    
    # Load dispatch output
    print(f"Loading dispatch output from {dispatch_csv}...")
    dispatch_df = pd.read_csv(dispatch_csv)
    
    # Validate required columns
    required_cols = ['timestamp_utc', 'unit_id', 'heat_MW']
    missing = [col for col in required_cols if col not in dispatch_df.columns]
    if missing:
        print(f"[ERROR] Dispatch CSV missing required columns: {missing}")
        return False
    
    # Parse timestamps
    dispatch_df['timestamp_utc'] = parse_any_timestamp(dispatch_df['timestamp_utc'])
    if dispatch_df['timestamp_utc'].dt.tz is None:
        dispatch_df['timestamp_utc'] = dispatch_df['timestamp_utc'].dt.tz_localize('UTC')
    else:
        dispatch_df['timestamp_utc'] = dispatch_df['timestamp_utc'].dt.tz_convert('UTC')
    
    # Load maintenance windows
    repo_root_path = repo_root()
    maint_df = load_maintenance_windows(repo_root_path, epoch, variant=None)
    
    if len(maint_df) == 0:
        print(f"[INFO] No maintenance windows found for epoch {epoch}")
        return True
    
    # Parse maintenance timestamps
    maint_df['start_timestamp_utc'] = parse_any_timestamp(maint_df['start_timestamp_utc'])
    maint_df['end_timestamp_utc'] = parse_any_timestamp(maint_df['end_timestamp_utc'])
    if maint_df['start_timestamp_utc'].dt.tz is None:
        maint_df['start_timestamp_utc'] = maint_df['start_timestamp_utc'].dt.tz_localize('UTC')
    else:
        maint_df['start_timestamp_utc'] = maint_df['start_timestamp_utc'].dt.tz_convert('UTC')
    if maint_df['end_timestamp_utc'].dt.tz is None:
        maint_df['end_timestamp_utc'] = maint_df['end_timestamp_utc'].dt.tz_localize('UTC')
    else:
        maint_df['end_timestamp_utc'] = maint_df['end_timestamp_utc'].dt.tz_convert('UTC')
    
    print(f"\nAuditing {len(maint_df)} maintenance windows...")
    print("=" * 80)
    
    all_passed = True
    tolerance = 1e-6
    
    for idx, window in maint_df.iterrows():
        unit_id = window['unit_id']
        start = window['start_timestamp_utc']
        end = window['end_timestamp_utc']
        availability = float(window['availability'])
        
        # Find dispatch rows in this window
        mask = (dispatch_df['timestamp_utc'] >= start) & (dispatch_df['timestamp_utc'] < end) & (dispatch_df['unit_id'] == unit_id)
        window_dispatch = dispatch_df[mask]
        
        n_rows = len(window_dispatch)
        max_heat_MW = window_dispatch['heat_MW'].max() if n_rows > 0 else 0.0
        sum_heat_MWh = (window_dispatch['heat_MW'] * 1.0).sum() if n_rows > 0 else 0.0  # Assuming 1h timesteps
        
        # Check: if availability == 0, heat_MW must be 0
        if availability == 0.0:
            if max_heat_MW > tolerance:
                print(f"\n[FAIL] Window {idx+1}: {unit_id} [{start} to {end})")
                print(f"  Availability: {availability} (fully offline)")
                print(f"  Max heat_MW: {max_heat_MW:.6f} (expected: 0)")
                print(f"  Rows matched: {n_rows}")
                print(f"  Sum heat_MWh: {sum_heat_MWh:.6f}")
                all_passed = False
            else:
                print(f"[PASS] Window {idx+1}: {unit_id} [{start} to {end}) - availability=0, max_heat={max_heat_MW:.6f}")
        else:
            # For availability > 0, just report
            print(f"[INFO] Window {idx+1}: {unit_id} [{start} to {end}) - availability={availability}, max_heat={max_heat_MW:.6f}, rows={n_rows}")
    
    print("=" * 80)
    
    if all_passed:
        print("\n[OK] All maintenance windows passed audit")
        return True
    else:
        print("\n[FAIL] Some maintenance windows failed audit")
        return False


def main():
    parser = argparse.ArgumentParser(description='Audit maintenance windows in dispatch outputs')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch year (e.g., 2020)')
    parser.add_argument('--run', type=str, required=True, help='Path to run folder (e.g., Output/runs/<run_id>)')
    parser.add_argument('--mode', type=str, default='opt', choices=['opt', 'prop'],
                       help='Dispatch mode: opt (optimal_subset/lp) or prop (proportional)')
    
    args = parser.parse_args()
    
    run_path = Path(args.run).resolve()
    if not run_path.exists():
        print(f"[ERROR] Run folder not found: {run_path}")
        sys.exit(1)
    
    success = audit_maintenance(args.epoch, run_path, args.mode)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

