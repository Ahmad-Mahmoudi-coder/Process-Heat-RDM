"""
Generate Grid RDM Futures (2035)

Expands Input/rdm/futures_2035.csv from 21 to 100 or 200 futures.
Preserves the first 21 futures exactly (bit-for-bit identical anchors).

Uses Latin Hypercube Sampling (LHS) for continuous uncertainties and
weighted discrete sampling for U_voll.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import shutil

from src.path_utils import repo_root, resolve_path


# ANCHOR FUTURES (future_id 0-20) - must remain bit-for-bit identical
ANCHORS = [
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

N_ANCHORS = len(ANCHORS)


def latin_hypercube_sample(n_samples: int, n_dims: int, bounds: list, seed: int = 42) -> np.ndarray:
    """
    Generate Latin Hypercube Sample (LHS) for continuous variables.
    
    Args:
        n_samples: Number of samples
        n_dims: Number of dimensions
        bounds: List of (min, max) tuples for each dimension
        seed: Random seed
        
    Returns:
        Array of shape (n_samples, n_dims)
    """
    np.random.seed(seed)
    
    # Create LHS grid
    samples = np.zeros((n_samples, n_dims))
    
    for dim in range(n_dims):
        # Create stratified samples
        lower, upper = bounds[dim]
        step = (upper - lower) / n_samples
        samples[:, dim] = np.linspace(lower + step/2, upper - step/2, n_samples)
        
        # Randomly permute
        np.random.shuffle(samples[:, dim])
    
    return samples


def sample_triangular(n_samples: int, a: float, mode: float, b: float, seed: int) -> np.ndarray:
    """
    Sample from triangular distribution.
    
    Args:
        n_samples: Number of samples
        a: Minimum
        mode: Mode (peak)
        b: Maximum
        seed: Random seed
        
    Returns:
        Array of samples
    """
    np.random.seed(seed)
    return np.random.triangular(a, mode, b, size=n_samples)


def sample_discrete_weighted(n_samples: int, values: list, probs: list, seed: int) -> np.ndarray:
    """
    Sample from discrete distribution with weights.
    
    Args:
        n_samples: Number of samples
        values: List of possible values
        probs: List of probabilities (must sum to 1.0)
        seed: Random seed
        
    Returns:
        Array of samples
    """
    np.random.seed(seed)
    return np.random.choice(values, size=n_samples, p=probs)


def generate_futures(n_total: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate futures table with anchors preserved.
    
    Args:
        n_total: Total number of futures (must be >= N_ANCHORS)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with futures
    """
    if n_total < N_ANCHORS:
        raise ValueError(f"n_total ({n_total}) must be >= {N_ANCHORS} (number of anchors)")
    
    n_new = n_total - N_ANCHORS
    
    # Start with anchors (do not modify)
    futures_list = []
    for anchor in ANCHORS:
        futures_list.append({
            'future_id': int(anchor[0]),
            'U_headroom_mult': float(anchor[1]),
            'U_inc_mult': float(anchor[2]),
            'U_upgrade_capex_mult': float(anchor[3]),
            'U_voll': float(anchor[4]),
            'U_consents_uplift': float(anchor[5])
        })
    
    # Generate new futures (21..n_total-1)
    # Use LHS for continuous variables
    continuous_bounds = [
        (0.75, 1.00),   # U_headroom_mult: triangular a=0.75, mode=0.90, b=1.00
        (0.85, 1.15),   # U_inc_mult: triangular a=0.85, mode=1.00, b=1.15
        (0.80, 1.50),   # U_upgrade_capex_mult: triangular a=0.80, mode=1.00, b=1.50
        (1.00, 1.60),   # U_consents_uplift: triangular a=1.00, mode=1.20, b=1.60
    ]
    
    # Generate LHS samples for continuous variables (uniform [0,1] quantiles)
    # Then map to triangular distributions using inverse CDF
    lhs_samples = latin_hypercube_sample(n_new, len(continuous_bounds), [(0.0, 1.0)] * len(continuous_bounds), seed=seed)
    
    def triangular_inverse_cdf(u_vals, a, mode, b):
        """
        Inverse CDF of triangular distribution (vectorized).
        
        Maps uniform quantiles u [0,1] to triangular values.
        """
        u_vals = np.clip(u_vals, 0.0, 1.0)  # Ensure u is in [0,1]
        threshold = (mode - a) / (b - a)
        result = np.zeros_like(u_vals)
        
        # Left side of triangle
        mask_left = u_vals < threshold
        result[mask_left] = a + np.sqrt(u_vals[mask_left] * (b - a) * (mode - a))
        
        # Right side of triangle
        mask_right = ~mask_left
        result[mask_right] = b - np.sqrt((1 - u_vals[mask_right]) * (b - a) * (b - mode))
        
        return result
    
    # Map LHS quantiles to triangular distributions
    headroom_samples = triangular_inverse_cdf(lhs_samples[:, 0], 0.75, 0.90, 1.00)
    inc_samples = triangular_inverse_cdf(lhs_samples[:, 1], 0.85, 1.00, 1.15)
    capex_samples = triangular_inverse_cdf(lhs_samples[:, 2], 0.80, 1.00, 1.50)
    consents_samples = triangular_inverse_cdf(lhs_samples[:, 3], 1.00, 1.20, 1.60)
    
    # Generate discrete U_voll samples
    voll_values = [5000, 10000, 15000, 20000]
    voll_probs = [0.10, 0.40, 0.35, 0.15]
    voll_samples = sample_discrete_weighted(n_new, voll_values, voll_probs, seed=seed + 5)
    
    # Round continuous variables to 3 decimals (new futures only)
    headroom_samples = np.round(headroom_samples, 3)
    inc_samples = np.round(inc_samples, 3)
    capex_samples = np.round(capex_samples, 3)
    consents_samples = np.round(consents_samples, 3)
    
    # Create new futures
    for i in range(n_new):
        future_id = N_ANCHORS + i
        futures_list.append({
            'future_id': future_id,
            'U_headroom_mult': float(headroom_samples[i]),
            'U_inc_mult': float(inc_samples[i]),
            'U_upgrade_capex_mult': float(capex_samples[i]),
            'U_voll': float(voll_samples[i]),
            'U_consents_uplift': float(consents_samples[i])
        })
    
    # Create DataFrame
    df = pd.DataFrame(futures_list)
    
    # Ensure column order
    df = df[['future_id', 'U_headroom_mult', 'U_inc_mult', 'U_upgrade_capex_mult', 'U_voll', 'U_consents_uplift']]
    
    return df


def backup_file(file_path: Path) -> Path:
    """
    Create timestamped backup of file.
    
    Args:
        file_path: Path to file to backup
        
    Returns:
        Path to backup file
    """
    if not file_path.exists():
        return None
    
    archive_dir = file_path.parent / '_archive'
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = archive_dir / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    backup_path = backup_dir / file_path.name
    shutil.copy2(file_path, backup_path)
    
    return backup_path


def main():
    """CLI entrypoint for futures generation."""
    parser = argparse.ArgumentParser(
        description='Generate grid RDM futures (2035) with preserved anchors'
    )
    parser.add_argument('--n', type=int, required=True,
                       help='Total number of futures (must be >= 21)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--out', type=str, default='Input/rdm/futures_2035.csv',
                       help='Output path (default: Input/rdm/futures_2035.csv)')
    
    args = parser.parse_args()
    
    if args.n < N_ANCHORS:
        raise ValueError(f"--n must be >= {N_ANCHORS} (number of anchor futures)")
    
    # Resolve output path
    out_path = Path(resolve_path(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing file if it exists
    if out_path.exists():
        backup_path = backup_file(out_path)
        print(f"[BACKUP] Backed up existing file to: {backup_path}")
    
    # Generate futures
    print(f"[GENERATE] Generating {args.n} futures (preserving {N_ANCHORS} anchors)...")
    df = generate_futures(args.n, seed=args.seed)
    
    # Write to CSV
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(df)} futures to: {out_path}")
    
    # Print summary
    print("\n[SUMMARY]")
    print(f"  Total futures: {len(df)}")
    print(f"  Anchors (0-{N_ANCHORS-1}): preserved exactly")
    print(f"  New futures ({N_ANCHORS}-{len(df)-1}): {len(df) - N_ANCHORS} generated")
    print("\n  Column statistics (all futures):")
    for col in ['U_headroom_mult', 'U_inc_mult', 'U_upgrade_capex_mult', 'U_voll', 'U_consents_uplift']:
        vals = df[col].values
        print(f"    {col}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")
    
    print(f"\n[OK] Generation complete (seed={args.seed})")


if __name__ == '__main__':
    main()

