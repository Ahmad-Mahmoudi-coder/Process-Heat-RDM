"""
Run-from-scratch convenience script for 2020 DemandPack and dispatch.

Non-destructive by default: outputs go to Output/runs/<run_id>/ and Output/latest/

Regenerates:
1. DemandPack hourly demand profile
2. All DemandPack diagnostic figures
3. Optimal subset dispatch with plots
"""

# Bootstrap: allow `python .\src\script.py` (adds repo root to sys.path)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import subprocess
import argparse

from src.path_utils import repo_root, resolve_path, input_root
from src.run_utils import (
    generate_run_id, setup_run_dir, update_latest_symlink,
    archive_latest, clean_latest
)


def run_command(cmd, description, output_dir: Path, cwd: Path = None):
    """Run a command and handle errors."""
    print(f"{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    if cwd is None:
        cwd = repo_root()
    
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(cwd))
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n[OK] {description} completed\n")


def find_demandpack_configs(configs_dir: Path) -> list[Path]:
    """Find all demandpack*.toml files in the configs directory."""
    if not configs_dir.exists():
        return []
    return sorted(configs_dir.glob('demandpack*.toml'))


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(
        description='Run 2020 DemandPack and dispatch pipeline (non-destructive)'
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
                       help='Path to demandpack config TOML (default: auto-discover from <input>/configs)')
    parser.add_argument('--signals-config', type=str, default=None,
                       help='Path to signals config TOML (default: <input>/signals/signals_config.toml)')
    parser.add_argument('--utilities-csv', type=str, default=None,
                       help='Path to site utilities CSV (default: auto-discover from <input>/site/utilities/)')
    args = parser.parse_args()
    
    # Define root directories
    ROOT = repo_root()
    if args.input_dir:
        INPUT_DIR = resolve_path(args.input_dir)
    else:
        INPUT_DIR = input_root()
    MODULES_DIR = ROOT / "modules"
    OUTPUT_DIR = ROOT / "Output"
    
    # Validation: check INPUT_DIR exists and contains expected structure
    print("="*60)
    print("2020 DemandPack and Dispatch - Full Pipeline")
    print("="*60)
    print(f"Repository root: {ROOT}")
    print(f"Input directory: {INPUT_DIR}")
    print()
    
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        sys.exit(1)
    
    # Check Input/configs exists
    configs_dir = INPUT_DIR / 'configs'
    if not configs_dir.exists():
        print(f"[ERROR] Configs directory not found: {configs_dir}")
        sys.exit(1)
    
    # Check Input/signals/signals_config.toml exists (or use provided path)
    if args.signals_config:
        signals_config_path = resolve_path(args.signals_config, base=INPUT_DIR)
    else:
        signals_config_path = INPUT_DIR / 'signals' / 'signals_config.toml'
    
    if not signals_config_path.exists():
        print(f"[ERROR] Signals config not found: {signals_config_path}")
        sys.exit(1)
    
    # Find demandpack configs
    if args.demandpack_config:
        demandpack_config_path = resolve_path(args.demandpack_config, base=INPUT_DIR)
        if not demandpack_config_path.exists():
            print(f"[ERROR] DemandPack config not found: {demandpack_config_path}")
            sys.exit(1)
    else:
        # Auto-discover from configs directory
        configs = find_demandpack_configs(configs_dir)
        if not configs:
            print(f"[ERROR] No demandpack*.toml files found in {configs_dir}")
            sys.exit(1)
        elif len(configs) == 1:
            demandpack_config_path = configs[0]
            print(f"[OK] Using demandpack config: {demandpack_config_path}")
        else:
            # Multiple configs found - list them and exit
            print(f"[ERROR] Multiple demandpack configs found in {configs_dir}:")
            print("Please specify one using --demandpack-config")
            print()
            for i, cfg in enumerate(configs, 1):
                print(f"  {i}. {cfg.name} ({cfg})")
            print()
            print(f"Example: --demandpack-config {configs[0].relative_to(ROOT)}")
            sys.exit(1)
    
    print(f"[OK] Input directory validated: {INPUT_DIR}")
    print(f"[OK] DemandPack config: {demandpack_config_path}")
    print(f"[OK] Signals config: {signals_config_path}")
    print()
    
    # Generate or use provided run ID
    run_id = args.run_id if args.run_id else generate_run_id()
    print(f"Run ID: {run_id}")
    
    # Handle archive/clean options
    if args.archive:
        archive_latest(run_id)
    elif args.clean:
        clean_latest()
    
    # Setup run directory
    run_dir = setup_run_dir(run_id)
    print(f"Output directory: {run_dir}")
    print()
    
    # Step 1: Read total site capacity from utilities
    # Use provided path or auto-discover (same logic as site_dispatch_2020)
    import pandas as pd
    if args.utilities_csv:
        util_csv = resolve_path(args.utilities_csv, base=INPUT_DIR)
    else:
        # Auto-discover: try default location first
        util_csv = INPUT_DIR / 'site' / 'utilities' / 'site_utilities_2020.csv'
        if not util_csv.exists():
            # Try searching for *utilities*2020*.csv
            utilities_dir = INPUT_DIR / 'site' / 'utilities'
            if utilities_dir.exists():
                matches = sorted(utilities_dir.glob('*utilities*2020*.csv'))
                if len(matches) == 1:
                    util_csv = matches[0]
                elif len(matches) > 1:
                    print(f"[ERROR] Multiple utilities CSV files found in {utilities_dir}:")
                    print("Please specify one using --utilities-csv")
                    print()
                    for i, match in enumerate(matches, 1):
                        print(f"  {i}. {match.name} ({match})")
                    print()
                    print(f"Example: --utilities-csv {matches[0].relative_to(ROOT)}")
                    sys.exit(1)
    
    if not util_csv.exists():
        print(f"[ERROR] Site utilities CSV not found: {util_csv}")
        print("Please specify the utilities CSV using --utilities-csv")
        sys.exit(1)
    
    util_df = pd.read_csv(util_csv)
    total_capacity_MW = util_df['max_heat_MW'].sum()
    print(f"Total site capacity: {total_capacity_MW:.2f} MW")
    print()
    
    # Step 2: Generate DemandPack with peak cap
    demand_csv = str(run_dir / 'hourly_heat_demand_2020.csv')
    config_path_str = str(demandpack_config_path)
    run_command(
        [sys.executable, '-m', 'src.generate_demandpack', '--config', config_path_str,
         '--cap-peak-mw', str(total_capacity_MW), '--output-dir', str(run_dir)],
        "Step 1: Generate DemandPack hourly demand profile",
        run_dir,
        cwd=ROOT
    )
    
    # Step 3: Generate DemandPack diagnostic figures
    figures_dir = run_dir / 'Figures'
    run_command(
        [sys.executable, '-m', 'src.plot_demandpack_diagnostics', 
         '--full-diagnostics', 
         '--config', config_path_str,
         '--data', demand_csv,
         '--output-dir', str(run_dir)],
        "Step 2: Generate DemandPack diagnostic figures",
        run_dir,
        cwd=ROOT
    )
    
    # Step 4: Run optimal subset dispatch with plots (reserve penalties OFF by default)
    dispatch_cmd = [
        sys.executable, '-m', 'src.site_dispatch_2020',
        '--mode', 'optimal_subset',
        '--plot',
        '--no-load-cost-nzd-per-h', '50.0',
        '--demand-csv', demand_csv,
        '--output-dir', str(run_dir)
    ]
    
    # Pass through utilities-csv if provided
    if args.utilities_csv:
        dispatch_cmd.extend(['--utilities-csv', args.utilities_csv])
    
    # Pass through demandpack-config if available (for utilities path lookup)
    if args.demandpack_config:
        dispatch_cmd.extend(['--demandpack-config', str(demandpack_config_path)])
    
    run_command(
        dispatch_cmd,
        "Step 3: Compute optimal subset dispatch and generate plots",
        run_dir,
        cwd=ROOT
    )
    
    # Step 5: Run regional electricity PoC (if GXP file exists)
    gxp_csv_path = MODULES_DIR / 'edendale_gxp' / 'outputs_latest' / 'gxp_hourly_2020.csv'
    incremental_path = run_dir / 'site_electricity_incremental_2020.csv'
    
    if gxp_csv_path.exists():
        regional_output = str(run_dir / 'regional_electricity_signals_2020.csv')
        cmd = [sys.executable, '-m', 'src.regional_electricity_poc',
               '--epoch', '2020',
               '--gxp-csv', str(gxp_csv_path),
               '--out', regional_output]
        
        # Add incremental if it exists, otherwise will default to zero series
        if incremental_path.exists():
            cmd.extend(['--incremental-csv', str(incremental_path)])
        
        run_command(
            cmd,
            "Step 4: Compute regional electricity signals (GXP capacity PoC)",
            run_dir,
            cwd=ROOT
        )
    else:
        print(f"[SKIP] Regional electricity PoC: {gxp_csv_path} not found")
    
    # Verify all required figures exist
    demandpack_plots = [
        'heat_2020_timeseries.png',
        'heat_2020_daily_envelope.png',
        'heat_2020_hourly_means_by_season.png',
        'heat_2020_monthly_totals.png',
        'heat_2020_LDC.png',
        'heat_2020_weekday_profiles_Feb.png',
        'heat_2020_weekday_profiles_Jun.png',
        'heat_2020_load_histogram.png'
    ]
    optimal_plots = [
        'heat_2020_unit_stack_opt.png',
        'heat_2020_units_online_opt.png',
        'heat_2020_unit_utilisation_duration_opt.png'
    ]
    all_required = demandpack_plots + optimal_plots
    missing = [p for p in all_required if not (figures_dir / p).exists()]
    if missing:
        print(f"\n[ERROR] Missing required figures ({len(missing)}/{len(all_required)}):")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)
    
    # Update Output/latest/ to point to this run
    print("\nUpdating Output/latest/...")
    update_latest_symlink(run_id)
    
    print("="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print(f"\nRun ID: {run_id}")
    print(f"Output directory: {run_dir}")
    print("\nGenerated outputs:")
    print("  CSVs:")
    for csv_file in sorted(run_dir.glob('*.csv')):
        print(f"    - {csv_file.name}")
    
    print("\n  Figures:")
    for png_file in sorted(figures_dir.glob('*.png')):
        print(f"    - {png_file.name}")
    print(f"\nOutput/latest/ has been updated to mirror this run.")
    print()


if __name__ == '__main__':
    main()

