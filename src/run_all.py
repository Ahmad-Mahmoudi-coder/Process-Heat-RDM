"""
Epoch-aware pipeline runner for DemandPack and dispatch.

All outputs go to Output/runs/<run_id>/ only (no latest directory)

Regenerates:
1. DemandPack hourly demand profile
2. All DemandPack diagnostic figures
3. Optimal subset dispatch with plots
4. Regional electricity signals (if GXP data available)
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
import re
import os

from src.path_utils import repo_root, resolve_path, input_root, resolve_cfg_path
# run_utils functions replaced by output_paths module


def run_command(cmd, description, output_dir: Path, cwd: Path = None, allow_failure: bool = False):
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command to run
        description: Description of the command
        output_dir: Output directory
        cwd: Working directory (default: repo_root)
        allow_failure: If True, return False on failure instead of exiting
        
    Returns:
        True if successful, False if failed (only if allow_failure=True)
        
    Raises:
        SystemExit: If command fails and allow_failure=False
    """
    print(f"{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    if cwd is None:
        cwd = repo_root()
    
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(cwd))
    
    if result.returncode != 0:
        if allow_failure:
            print(f"\n[WARN] Command failed with exit code {result.returncode}")
            return False
        else:
            print(f"\n[ERROR] Command failed with exit code {result.returncode}")
            sys.exit(1)
    
    print(f"\n[OK] {description} completed\n")
    return True


def find_demandpack_configs(configs_dir: Path) -> list[Path]:
    """Find all demandpack*.toml files in the configs directory."""
    if not configs_dir.exists():
        return []
    return sorted(configs_dir.glob('demandpack*.toml'))


def infer_epoch_from_config_name(config_path: Path) -> int:
    """Infer epoch from config filename (e.g., demandpack_2020.toml -> 2020)."""
    name = config_path.stem  # e.g., "demandpack_2020"
    match = re.search(r'(\d{4})', name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer epoch from config filename: {config_path.name}")


def resolve_utilities_csv_path(path_str: str, repo_root: Path, input_dir: Path) -> Path:
    """
    Resolve utilities CSV path with specific fallback order.
    
    Resolution order:
    1. If absolute: use it
    2. Else if exists relative to repo root: use it
    3. Else if exists relative to Input root: use it
    4. Else raise error
    
    Args:
        path_str: Path string provided by user
        repo_root: Repository root directory
        input_dir: Input directory (repo_root / "Input")
        
    Returns:
        Resolved absolute Path
        
    Raises:
        SystemExit: If path cannot be resolved
    """
    from src.path_utils import resolve_path
    
    path = Path(path_str)
    
    # Expand user and environment variables
    path_str_expanded = os.path.expanduser(str(path))
    path_str_expanded = os.path.expandvars(path_str_expanded)
    path = Path(path_str_expanded)
    
    # If absolute, use it directly
    if path.is_absolute():
        resolved = path.resolve()
        if resolved.exists():
            return resolved
        print(f"[ERROR] Utilities CSV not found: {resolved}")
        print("  (Path is absolute but does not exist)")
        sys.exit(1)
    
    # Try 1: relative to repo root
    candidate1 = (repo_root / path).resolve()
    if candidate1.exists():
        return candidate1
    
    # Try 2: relative to Input root
    candidate2 = (input_dir / path).resolve()
    if candidate2.exists():
        return candidate2
    
    # None found - show attempted paths
    print(f"[ERROR] Utilities CSV not found: {path_str}")
    print("  Attempted paths:")
    print(f"    1. {candidate1}")
    print(f"    2. {candidate2}")
    print()
    print("  Please provide an absolute path, or a path relative to:")
    print(f"    - Repository root: {repo_root}")
    print(f"    - Input directory: {input_dir}")
    sys.exit(1)


def find_utilities_csv(INPUT_DIR: Path, epoch: int, demandpack_config_path: Path = None, variant: str = None) -> Path:
    """
    Find utilities CSV with auto-discovery logic.
    
    Resolution order:
    1. From demandpack config (if provided and contains utilities_csv)
    2. <input_dir>/site/utilities/site_utilities_<epoch>.csv
    3. Search <input_dir>/site/utilities/ for *utilities*<epoch>*.csv
    
    Returns:
        Path to utilities CSV
        
    Raises:
        SystemExit: If not found or multiple matches
    """
    attempted_paths = []
    
    # Try 1: From demandpack config
    if demandpack_config_path:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None
        
        if tomllib:
            try:
                with open(demandpack_config_path, 'rb') as f:
                    demandpack_config = tomllib.load(f)
                
                utilities_path = None
                if 'general' in demandpack_config and 'utilities_csv' in demandpack_config['general']:
                    utilities_path = demandpack_config['general']['utilities_csv']
                elif 'site' in demandpack_config and 'utilities_csv' in demandpack_config['site']:
                    utilities_path = demandpack_config['site']['utilities_csv']
                
                if utilities_path:
                    candidate = resolve_cfg_path(demandpack_config_path, utilities_path)
                    attempted_paths.append(f"  a) From demandpack config: {candidate}")
                    if candidate.exists():
                        return candidate
            except Exception:
                pass
    
    # Try 2: Default location (with variant if provided)
    if variant:
        candidate = INPUT_DIR / 'site' / 'utilities' / f'site_utilities_{epoch}_{variant}.csv'
    else:
        candidate = INPUT_DIR / 'site' / 'utilities' / f'site_utilities_{epoch}.csv'
    attempted_paths.append(f"  b) Default location: {candidate}")
    if candidate.exists():
        return candidate
    
    # Try 3: Search for *utilities*<epoch>*.csv
    utilities_dir = INPUT_DIR / 'site' / 'utilities'
    if utilities_dir.exists():
        pattern = f'*utilities*{epoch}*.csv'
        matches = sorted(utilities_dir.glob(pattern))
        attempted_paths.append(f"  c) Searched in: {utilities_dir} (pattern: {pattern})")
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # If variant is provided, try to match it
            if variant:
                variant_pattern = f'*utilities*{epoch}_{variant}.csv'
                variant_matches = sorted(utilities_dir.glob(variant_pattern))
                if len(variant_matches) == 1:
                    return variant_matches[0]
                elif len(variant_matches) > 1:
                    # Multiple matches for variant - should not happen, but use first
                    return variant_matches[0]
                else:
                    # Variant not found - error with expected path
                    expected_path = utilities_dir / f'site_utilities_{epoch}_{variant}.csv'
                    print(f"[ERROR] Utilities CSV not found for variant {variant}: {expected_path}")
                    print(f"Available files for epoch {epoch}:")
                    for i, match in enumerate(matches, 1):
                        print(f"  {i}. {match.name}")
                    raise FileNotFoundError(f"Utilities CSV not found for variant {variant}")
            else:
                # No variant specified, but multiple found - error
                print(f"[ERROR] Multiple utilities CSV files found for epoch {epoch} in {utilities_dir}:")
                print("Please specify one using --utilities-csv or use --variants-2035 for batch mode")
                print()
                for i, match in enumerate(matches, 1):
                    print(f"  {i}. {match.name} ({match})")
                print()
                print(f"Example: --utilities-csv {matches[0].relative_to(repo_root())}")
                raise ValueError(f"Multiple utilities CSV files found for epoch {epoch}")
    
    # Not found
    print(f"[ERROR] Utilities CSV file not found for epoch {epoch}. Attempted paths:")
    for path_str in attempted_paths:
        print(path_str)
    print()
    print("Please specify the utilities CSV using --utilities-csv")
    print(f"Example: --utilities-csv {INPUT_DIR / 'site' / 'utilities' / f'site_utilities_{epoch}.csv'}")
    raise FileNotFoundError(f"Utilities CSV not found for epoch {epoch}")


def run_single_epoch(epoch: int, batch_run_id: str, args, INPUT_DIR: Path, ROOT: Path, 
                     MODULES_DIR: Path, OUTPUT_DIR: Path, variant: str = None):
    """
    Run pipeline for a single epoch.
    
    Args:
        epoch: Epoch year
        batch_run_id: Base batch run ID (will be extended with epoch/variant)
        args: Parsed arguments
        INPUT_DIR: Input directory
        ROOT: Repository root
        MODULES_DIR: Modules directory
        OUTPUT_DIR: Output directory
        variant: Optional variant string (e.g., "EB", "BB" for 2035)
        
    Returns:
        Tuple of (derived_run_id, success_dict) where success_dict has keys:
        - 'demandpack': bool
        - 'plot': bool
        - 'dispatch': bool
        - 'regional': bool (or None if skipped)
    """
    # Initialize success dict with all False (will be updated as steps succeed)
    success_dict = {
        'demandpack': False,
        'plot': False,
        'dispatch': False,
        'regional': None  # None means skipped, True/False means attempted
    }
    
    # Create derived run ID
    if variant:
        derived_run_id = f"{batch_run_id}_epoch{epoch}_{variant}"
    else:
        derived_run_id = f"{batch_run_id}_epoch{epoch}"
    
    # Resolve demandpack config
    configs_dir = INPUT_DIR / 'configs'
    if args.demandpack_config:
        demandpack_config_path = resolve_path(args.demandpack_config, base=INPUT_DIR)
        if not demandpack_config_path.exists():
            print(f"[ERROR] DemandPack config not found: {demandpack_config_path}")
            return derived_run_id, success_dict
    else:
        demandpack_config_path = configs_dir / f'demandpack_{epoch}.toml'
        if not demandpack_config_path.exists():
            print(f"[ERROR] DemandPack config not found: {demandpack_config_path}")
            print("Please specify using --demandpack-config")
            return derived_run_id, success_dict
    
    # Resolve signals config
    if args.signals_config:
        signals_config_path = resolve_path(args.signals_config, base=INPUT_DIR)
    else:
        signals_config_path = INPUT_DIR / 'signals' / 'signals_config.toml'
    
    if not signals_config_path.exists():
        print(f"[ERROR] Signals config not found: {signals_config_path}")
        return derived_run_id, success_dict
    
    # Resolve utilities CSV (with variant support for 2035)
    if args.utilities_csv:
        utilities_csv_path = resolve_utilities_csv_path(args.utilities_csv, ROOT, INPUT_DIR)
        if utilities_csv_path is None or not utilities_csv_path.exists():
            print(f"[ERROR] Utilities CSV not found: {args.utilities_csv}")
            return derived_run_id, success_dict
    else:
        if epoch == 2035 and variant:
            # Use variant-specific utilities CSV
            utilities_csv_path = INPUT_DIR / 'site' / 'utilities' / f'site_utilities_2035_{variant}.csv'
            if not utilities_csv_path.exists():
                print(f"[ERROR] Utilities CSV not found for variant {variant}: {utilities_csv_path}")
                print(f"Expected path: {utilities_csv_path}")
                return derived_run_id, success_dict
            print(f"[OK] Using utilities CSV for variant {variant}: {utilities_csv_path}")
        else:
            try:
                utilities_csv_path = find_utilities_csv(INPUT_DIR, epoch, demandpack_config_path, variant=variant)
                print(f"[OK] Auto-discovered utilities CSV: {utilities_csv_path}")
            except (FileNotFoundError, ValueError) as e:
                # find_utilities_csv raises exceptions on failure - catch and return
                print(f"[ERROR] {e}")
                return derived_run_id, success_dict
    
    # Resolve GXP CSV
    if args.gxp_csv:
        gxp_csv_path = resolve_path(args.gxp_csv)
    else:
        gxp_csv_path = MODULES_DIR / 'edendale_gxp' / 'outputs_latest' / f'gxp_hourly_{epoch}.csv'
    
    print(f"[OK] Input directory validated: {INPUT_DIR}")
    print(f"[OK] DemandPack config: {demandpack_config_path}")
    print(f"[OK] Signals config: {signals_config_path}")
    print(f"[OK] Utilities CSV: {utilities_csv_path}")
    print()
    
    # Resolve output paths using new system
    from src.output_paths import resolve_run_paths
    
    output_paths = resolve_run_paths(
        output_root=OUTPUT_DIR,
        epoch=epoch,
        config_path=demandpack_config_path,
        run_id=derived_run_id
    )
    
    run_id = output_paths['run_id']
    run_dir = output_paths['run_dir']
    
    print(f"Run ID: {run_id}")
    print(f"Output directory: {run_dir}")
    print()
    
    # Step 1: Read total site capacity from utilities
    import pandas as pd
    from src.site_dispatch_2020 import load_utilities
    
    # Load all utilities (raw) to compute total capacity
    util_df_raw = pd.read_csv(utilities_csv_path)
    capacity_total = util_df_raw['max_heat_MW'].sum()
    
    # Load dispatchable utilities only (filtered by status)
    util_df = load_utilities(str(utilities_csv_path), epoch=epoch)
    capacity_dispatchable = util_df['max_heat_MW'].sum()
    
    print(f"Site capacity (raw): {capacity_total:.2f} MW")
    print(f"Site capacity (dispatchable): {capacity_dispatchable:.2f} MW")
    print(f"Using dispatchable capacity for cap-peak-mw: {capacity_dispatchable:.2f} MW")
    print()
    
    # Step 2: Generate DemandPack with peak cap
    demand_csv = str(output_paths['run_demandpack_dir'] / f'hourly_heat_demand_{epoch}.csv')
    config_path_str = str(demandpack_config_path)
    
    demandpack_success = run_command(
        [sys.executable, '-m', 'src.generate_demandpack', '--config', config_path_str,
         '--epoch', str(epoch),
         '--cap-peak-mw', str(capacity_dispatchable),
         '--output-root', str(OUTPUT_DIR),
         '--run-id', run_id],
        f"Step 1: Generate DemandPack hourly demand profile (epoch {epoch}{' ' + variant if variant else ''})",
        run_dir,
        cwd=ROOT,
        allow_failure=True
    )
    
    success_dict['demandpack'] = demandpack_success
    
    if not demandpack_success:
        print(f"[WARN] DemandPack generation failed for epoch {epoch} (run_id={run_id})")
        print("  Continuing to next step...")
    
    # Step 3: Generate DemandPack diagnostic figures (with resilience)
    plot_cmd = [
        sys.executable, '-m', 'src.plot_demandpack_diagnostics', 
        '--full-diagnostics', 
        '--config', config_path_str,
        '--epoch', str(epoch),
        '--output-root', str(OUTPUT_DIR),
        '--run-id', run_id
    ]
    
    plot_success = False
    result = subprocess.run(plot_cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode == 2 and ("unrecognized arguments" in result.stderr or "unrecognized arguments" in result.stdout):
        # Fallback to legacy arguments
        print("[WARN] Plot script doesn't support new args, using legacy --output-dir")
        plot_cmd_legacy = [
            sys.executable, '-m', 'src.plot_demandpack_diagnostics', 
            '--full-diagnostics', 
            '--config', config_path_str,
            '--epoch', str(epoch),
            '--output-dir', str(output_paths['run_figures_dir']),
            '--data', demand_csv
        ]
        result_legacy = subprocess.run(plot_cmd_legacy, capture_output=True, text=True, cwd=str(ROOT))
        if result_legacy.returncode == 0:
            print(result_legacy.stdout)
            if result_legacy.stderr:
                print(result_legacy.stderr)
            plot_success = True
        else:
            print(f"[WARN] Plot step failed for epoch {epoch} (run_id={run_id})")
            print(result_legacy.stdout)
            print(result_legacy.stderr)
            print("  Continuing to next step...")
    elif result.returncode == 0:
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        plot_success = True
    else:
        print(f"[WARN] Plot step failed for epoch {epoch} (run_id={run_id})")
        print(result.stdout)
        print(result.stderr)
        print("  Continuing to next step...")
    
    success_dict['plot'] = plot_success
    
    # Step 4: Run optimal subset dispatch with plots
    dispatch_cmd = [
        sys.executable, '-m', 'src.site_dispatch_2020',
        '--mode', 'optimal_subset',
        '--epoch', str(epoch),
        '--plot',
        '--no-load-cost-nzd-per-h', '50.0',
        '--demand-csv', demand_csv,
        '--utilities-csv', str(utilities_csv_path),
        '--output-root', str(OUTPUT_DIR),
        '--run-id', run_id
    ]
    
    # Pass through demandpack-config
    dispatch_cmd.extend(['--demandpack-config', str(demandpack_config_path)])
    
    dispatch_success = run_command(
        dispatch_cmd,
        f"Step 3: Compute optimal subset dispatch and generate plots (epoch {epoch}{' ' + variant if variant else ''})",
        run_dir,
        cwd=ROOT,
        allow_failure=True
    )
    
    success_dict['dispatch'] = dispatch_success
    
    if not dispatch_success:
        print(f"[WARN] Dispatch step failed for epoch {epoch} (run_id={run_id})")
        print("  Continuing to next step...")
    
    # Step 5: Run regional electricity PoC (if GXP file exists)
    incremental_path = run_dir / f'site_electricity_incremental_{epoch}.csv'
    regional_success = None
    
    if gxp_csv_path.exists():
        regional_output = str(run_dir / f'regional_electricity_signals_{epoch}.csv')
        cmd = [sys.executable, '-m', 'src.regional_electricity_poc',
               '--epoch', str(epoch),
               '--gxp-csv', str(gxp_csv_path),
               '--out', regional_output]
        
        if incremental_path.exists():
            cmd.extend(['--incremental-csv', str(incremental_path)])
        
        regional_success = run_command(
            cmd,
            f"Step 4: Compute regional electricity signals (GXP capacity PoC, epoch {epoch}{' ' + variant if variant else ''})",
            run_dir,
            cwd=ROOT,
            allow_failure=True
        )
        
        success_dict['regional'] = regional_success
        
        if not regional_success:
            print(f"[WARN] Regional electricity PoC failed for epoch {epoch} (run_id={run_id})")
            print("  Continuing...")
    else:
        print(f"[SKIP] Regional electricity PoC: {gxp_csv_path} not found")
        # regional_success remains None (skipped)
    
    # Outputs are written directly to run_dir (no copying to latest)
    
    print(f"[OK] Epoch {epoch}{' ' + variant if variant else ''} completed")
    print()
    
    return derived_run_id, success_dict


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(
        description='Run epoch-aware DemandPack and dispatch pipeline (non-destructive)'
    )
    parser.add_argument('--epoch', type=int, default=None,
                       help='Single epoch year (e.g., 2020, 2025, 2028, 2035). Required if --epochs not provided.')
    parser.add_argument('--epochs', type=str, default=None,
                       help='Comma-separated list of epochs (e.g., "2020,2025,2028,2035"). If provided, --epoch becomes optional.')
    parser.add_argument('--variants-2035', type=str, default=None,
                       help='Comma-separated list of 2035 variants (e.g., "EB,BB"). Defaults to "EB,BB" if multiple utilities found.')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Base run ID (default: auto-generated). For multi-epoch, extended with _epoch{epoch} or _epoch2035_{variant}')
    parser.add_argument('--clean', action='store_true',
                       help='Clean Output/latest/ before running (optional)')
    parser.add_argument('--archive', action='store_true',
                       help='Archive Output/latest/ to Output/_archive/ before running batch (optional, archives once per batch)')
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Input directory (default: repo_root/Input)')
    parser.add_argument('--demandpack-config', type=str, default=None,
                       help='Path to demandpack config TOML (default: <input>/configs/demandpack_<epoch>.toml)')
    parser.add_argument('--signals-config', type=str, default=None,
                       help='Path to signals config TOML (default: <input>/signals/signals_config.toml)')
    parser.add_argument('--utilities-csv', type=str, default=None,
                       help='Path to site utilities CSV (default: auto-discover from <input>/site/utilities/)')
    parser.add_argument('--gxp-csv', type=str, default=None,
                       help='Path to GXP hourly CSV (default: modules/edendale_gxp/outputs_latest/gxp_hourly_<epoch>.csv)')
    args = parser.parse_args()
    
    # Parse epochs
    epochs = []
    if args.epochs:
        epochs = [int(e.strip()) for e in args.epochs.split(',')]
    elif args.epoch:
        epochs = [args.epoch]
    else:
        print("[ERROR] Either --epoch or --epochs must be provided")
        sys.exit(1)
    
    # Parse 2035 variants
    variants_2035 = []
    if args.variants_2035:
        variants_2035 = [v.strip() for v in args.variants_2035.split(',')]
    elif 2035 in epochs:
        # Check if multiple 2035 utilities exist
        INPUT_DIR = input_root() if not args.input_dir else resolve_path(args.input_dir)
        utilities_dir = INPUT_DIR / 'site' / 'utilities'
        if utilities_dir.exists():
            matches = sorted(utilities_dir.glob('*utilities*2035*.csv'))
            if len(matches) > 1:
                # Default to both EB and BB if found
                variants_2035 = ['EB', 'BB']
    
    # Define root directories
    ROOT = repo_root()
    if args.input_dir:
        INPUT_DIR = resolve_path(args.input_dir)
    else:
        INPUT_DIR = input_root()
    MODULES_DIR = ROOT / "modules"
    OUTPUT_DIR = ROOT / "Output"
    
    # Generate base batch run ID
    from src.output_paths import generate_run_id
    if args.run_id:
        batch_run_id = args.run_id
    else:
        # Generate base ID (will be extended per epoch)
        if len(epochs) > 1:
            batch_run_id = generate_run_id(epochs[0], None).replace(f"_epoch{epochs[0]}", "_all_epochs")
        else:
            batch_run_id = generate_run_id(epochs[0], None)
    
    print("="*60)
    if len(epochs) > 1:
        print(f"Multi-epoch DemandPack and Dispatch - Full Pipeline")
        print(f"Epochs: {epochs}")
        if variants_2035:
            print(f"2035 variants: {variants_2035}")
    else:
        print(f"Epoch {epochs[0]} DemandPack and Dispatch - Full Pipeline")
    print("="*60)
    print(f"Repository root: {ROOT}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Base batch run ID: {batch_run_id}")
    print()
    
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        sys.exit(1)
    
    # Check Input/configs exists
    configs_dir = INPUT_DIR / 'configs'
    if not configs_dir.exists():
        print(f"[ERROR] Configs directory not found: {configs_dir}")
        sys.exit(1)
    
    # Archive previous runs once at the beginning if --archive is set
    if args.archive:
        from src.output_paths import archive_previous_runs
        archive_previous_runs(OUTPUT_DIR)
    
    # Write batch metadata
    from src.output_paths import write_batch_metadata
    write_batch_metadata(OUTPUT_DIR, batch_run_id, epochs, variants_2035)
    
    # Run each epoch
    all_run_ids = []
    all_successes = []
    for epoch in epochs:
        if epoch == 2035 and variants_2035:
            # Run each variant
            for variant in variants_2035:
                run_id, success = run_single_epoch(epoch, batch_run_id, args, INPUT_DIR, ROOT, MODULES_DIR, OUTPUT_DIR, variant=variant)
                all_run_ids.append(run_id)
                all_successes.append((run_id, epoch, variant, success))
        else:
            # Run once for this epoch
            run_id, success = run_single_epoch(epoch, batch_run_id, args, INPUT_DIR, ROOT, MODULES_DIR, OUTPUT_DIR)
            all_run_ids.append(run_id)
            all_successes.append((run_id, epoch, None, success))
    
    # Final summary with success status
    print("="*60)
    print("Pipeline completed!")
    print("="*60)
    print(f"\nBase batch run ID: {batch_run_id}")
    print(f"All run IDs: {', '.join(all_run_ids)}")
    print(f"Latest directory: {OUTPUT_DIR / 'latest'}")
    
    # Check for failures
    failures = []
    for run_id, epoch, variant, success in all_successes:
        epoch_label = f"{epoch}{'_' + variant if variant else ''}"
        if not success.get('demandpack', True):
            failures.append(f"  - {epoch_label} (run_id={run_id}): DemandPack generation failed")
        if not success.get('plot', True):
            failures.append(f"  - {epoch_label} (run_id={run_id}): Plot step failed")
        if not success.get('dispatch', True):
            failures.append(f"  - {epoch_label} (run_id={run_id}): Dispatch step failed")
        if success.get('regional') is False:
            failures.append(f"  - {epoch_label} (run_id={run_id}): Regional electricity PoC failed")
    
    if failures:
        print("\n[WARN] Some steps failed:")
        for failure in failures:
            print(failure)
        print("\nGenerated outputs (partial):")
    else:
        print("\n[OK] All steps completed successfully!")
        print("\nGenerated outputs:")
    
    print("  All outputs are in:")
    for run_id in all_run_ids:
        run_dir = OUTPUT_DIR / 'runs' / run_id
        print(f"    - {run_dir}")
    print()
    
    # Exit with non-zero if any failures
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
