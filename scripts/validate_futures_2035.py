"""
Validate Grid RDM Futures (2035)

Fast sanity check for futures CSV:
- Row count matches expected (100 or 200)
- future_id is 0..n-1, unique
- Anchors 0-20 match expected values exactly
- Column ranges within bounds
- U_voll only in {5000, 10000, 15000, 20000}
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np

from src.path_utils import repo_root, resolve_path

# Expected anchor futures (future_id 0-20)
EXPECTED_ANCHORS = [
    [0, 0.85, 1.0, 1.0, 10000.0, 1.25],
    [1, 0.90, 0.95, 0.90, 10000.0, 1.30],
    [2, 0.95, 1.05, 1.10, 15000.0, 1.20],
    [3, 0.80, 1.10, 1.20, 15000.0, 1.40],
    [4, 0.90, 1.0, 1.0, 20000.0, 1.25],
    [5, 0.85, 1.05, 1.15, 10000.0, 1.35],
    [6, 0.95, 0.95, 0.95, 15000.0, 1.15],
    [7, 0.88, 1.08, 1.05, 15000.0, 1.28],
    [8, 0.92, 0.98, 1.08, 10000.0, 1.22],
    [9, 0.87, 1.02, 0.92, 20000.0, 1.32],
    [10, 0.93, 1.03, 1.12, 15000.0, 1.18],
    [11, 0.86, 0.97, 1.03, 10000.0, 1.27],
    [12, 0.91, 1.06, 0.88, 15000.0, 1.23],
    [13, 0.89, 1.01, 1.18, 20000.0, 1.29],
    [14, 0.94, 0.99, 1.06, 10000.0, 1.21],
    [15, 0.88, 1.04, 0.97, 15000.0, 1.26],
    [16, 0.92, 1.07, 1.14, 15000.0, 1.24],
    [17, 0.86, 0.96, 1.01, 20000.0, 1.31],
    [18, 0.90, 1.09, 1.07, 10000.0, 1.19],
    [19, 0.95, 1.0, 0.94, 15000.0, 1.33],
    [20, 0.87, 1.02, 1.11, 15000.0, 1.17],
]

N_ANCHORS = len(EXPECTED_ANCHORS)

# Expected bounds
BOUNDS = {
    'U_headroom_mult': (0.75, 1.00),
    'U_inc_mult': (0.85, 1.15),
    'U_upgrade_capex_mult': (0.80, 1.50),
    'U_consents_uplift': (1.00, 1.60),
}

VALID_VOLL = {5000, 10000, 15000, 20000}


def validate_futures(csv_path: Path, expected_n: int = None) -> bool:
    """
    Validate futures CSV.
    
    Args:
        csv_path: Path to futures CSV
        expected_n: Expected number of futures (if None, checks for 100 or 200)
        
    Returns:
        True if valid, False otherwise
    """
    print(f"[VALIDATE] Checking {csv_path}...")
    
    if not csv_path.exists():
        print(f"  [FAIL] File not found: {csv_path}")
        return False
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [FAIL] Failed to read CSV: {e}")
        return False
    
    # Check row count
    n_rows = len(df)
    if expected_n is None:
        if n_rows not in [100, 200]:
            print(f"  [FAIL] Expected 100 or 200 rows, got {n_rows}")
            return False
        expected_n = n_rows
    else:
        if n_rows != expected_n:
            print(f"  [FAIL] Expected {expected_n} rows, got {n_rows}")
            return False
    
    print(f"  [OK] Row count: {n_rows}")
    
    # Check required columns
    required_cols = ['future_id', 'U_headroom_mult', 'U_inc_mult', 'U_upgrade_capex_mult', 'U_voll', 'U_consents_uplift']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"  [FAIL] Missing columns: {missing}")
        return False
    
    print(f"  [OK] Required columns present")
    
    # Check future_id
    if df['future_id'].dtype != 'int64':
        print(f"  [FAIL] future_id must be integer type")
        return False
    
    future_ids = df['future_id'].values
    expected_ids = set(range(expected_n))
    actual_ids = set(future_ids)
    
    if actual_ids != expected_ids:
        missing_ids = expected_ids - actual_ids
        extra_ids = actual_ids - expected_ids
        if missing_ids:
            print(f"  [FAIL] Missing future_ids: {sorted(missing_ids)}")
        if extra_ids:
            print(f"  [FAIL] Extra future_ids: {sorted(extra_ids)}")
        return False
    
    if len(future_ids) != len(set(future_ids)):
        print(f"  [FAIL] future_id contains duplicates")
        return False
    
    print(f"  [OK] future_id: 0..{expected_n-1}, unique")
    
    # Check anchors (0-20) match exactly
    anchors_df = df[df['future_id'] < N_ANCHORS].sort_values('future_id')
    
    for i, anchor_row in anchors_df.iterrows():
        fid = int(anchor_row['future_id'])
        expected = EXPECTED_ANCHORS[fid]
        
        # Check each column (with tolerance for floating point)
        if abs(anchor_row['U_headroom_mult'] - expected[1]) > 1e-10:
            print(f"  [FAIL] Anchor {fid}: U_headroom_mult mismatch: got {anchor_row['U_headroom_mult']}, expected {expected[1]}")
            return False
        if abs(anchor_row['U_inc_mult'] - expected[2]) > 1e-10:
            print(f"  [FAIL] Anchor {fid}: U_inc_mult mismatch: got {anchor_row['U_inc_mult']}, expected {expected[2]}")
            return False
        if abs(anchor_row['U_upgrade_capex_mult'] - expected[3]) > 1e-10:
            print(f"  [FAIL] Anchor {fid}: U_upgrade_capex_mult mismatch: got {anchor_row['U_upgrade_capex_mult']}, expected {expected[3]}")
            return False
        if abs(anchor_row['U_voll'] - expected[4]) > 1e-10:
            print(f"  [FAIL] Anchor {fid}: U_voll mismatch: got {anchor_row['U_voll']}, expected {expected[4]}")
            return False
        if abs(anchor_row['U_consents_uplift'] - expected[5]) > 1e-10:
            print(f"  [FAIL] Anchor {fid}: U_consents_uplift mismatch: got {anchor_row['U_consents_uplift']}, expected {expected[5]}")
            return False
    
    print(f"  [OK] Anchors 0-{N_ANCHORS-1} match exactly")
    
    # Check column ranges
    for col, (min_val, max_val) in BOUNDS.items():
        vals = df[col].values
        if vals.min() < min_val or vals.max() > max_val:
            print(f"  [FAIL] {col} out of bounds: min={vals.min():.3f}, max={vals.max():.3f}, expected [{min_val}, {max_val}]")
            return False
    
    print(f"  [OK] Column ranges within bounds")
    
    # Check U_voll values
    voll_values = set(df['U_voll'].values)
    invalid_voll = voll_values - VALID_VOLL
    if invalid_voll:
        print(f"  [FAIL] Invalid U_voll values: {sorted(invalid_voll)}")
        print(f"  [INFO] Valid values: {sorted(VALID_VOLL)}")
        return False
    
    print(f"  [OK] U_voll only in {sorted(VALID_VOLL)}")
    
    print(f"\n[OK] All validation checks passed for {n_rows} futures")
    return True


def main():
    """CLI entrypoint for futures validation."""
    parser = argparse.ArgumentParser(
        description='Validate grid RDM futures (2035) CSV'
    )
    parser.add_argument('--csv', type=str, default='Input/rdm/futures_2035.csv',
                       help='Path to futures CSV (default: Input/rdm/futures_2035.csv)')
    parser.add_argument('--n', type=int, default=None,
                       help='Expected number of futures (if None, checks for 100 or 200)')
    
    args = parser.parse_args()
    
    csv_path = Path(resolve_path(args.csv))
    
    success = validate_futures(csv_path, expected_n=args.n)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

