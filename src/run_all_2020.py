"""
DEPRECATED: This script is a backward-compatibility wrapper.
Use src.run_all with --epoch 2020 instead.

This script will be moved to scripts/legacy/ in a future update.

Backward-compatible wrapper for 2020 epoch pipeline.
This is a thin wrapper that calls run_all.py with --epoch 2020.
For new code, use: python -m src.run_all --epoch 2020
"""

# Bootstrap: allow `python .\src\script.py` (adds repo root to sys.path)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess
import argparse

from src.path_utils import repo_root


def main():
    """Main entrypoint - thin wrapper that calls run_all.py with --epoch 2020."""
    parser = argparse.ArgumentParser(
        description='Run 2020 DemandPack and dispatch pipeline (backward-compatible wrapper)'
    )
    parser.add_argument('--run-id', type=str, default=None,
                       help='Optional run ID override (default: auto-generated YYYYMMDD_HHMMSS)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean Output/latest/ before running (optional)')
    parser.add_argument('--archive', action='store_true',
                       help='Archive Output/latest/ to Output/_archive/<run_id>/ before running (optional)')
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Input directory (default: repo_root/Input)')
    parser.add_argument('--demandpack-config', type=str, default=None,
                       help='Path to demandpack config TOML (default: <input>/configs/demandpack_2020.toml)')
    parser.add_argument('--signals-config', type=str, default=None,
                       help='Path to signals config TOML (default: <input>/signals/signals_config.toml)')
    parser.add_argument('--utilities-csv', type=str, default=None,
                       help='Path to site utilities CSV (default: auto-discover from <input>/site/utilities/)')
    args = parser.parse_args()
    
    # Build command to call run_all.py with --epoch 2020
    ROOT = repo_root()
    cmd = [sys.executable, '-m', 'src.run_all', '--epoch', '2020']
    
    # Pass through all arguments
    if args.run_id:
        cmd.extend(['--run-id', args.run_id])
    if args.clean:
        cmd.append('--clean')
    if args.archive:
        cmd.append('--archive')
    if args.input_dir:
        cmd.extend(['--input-dir', args.input_dir])
    if args.demandpack_config:
        cmd.extend(['--demandpack-config', args.demandpack_config])
    if args.signals_config:
        cmd.extend(['--signals-config', args.signals_config])
    if args.utilities_csv:
        cmd.extend(['--utilities-csv', args.utilities_csv])
    
    # Execute
    result = subprocess.run(cmd, cwd=str(ROOT))
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()

