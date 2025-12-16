"""
Run-from-scratch convenience script for 2020 DemandPack and dispatch.

Deletes existing outputs and regenerates:
1. DemandPack hourly demand profile
2. All DemandPack diagnostic figures
3. Optimal subset dispatch with plots
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import subprocess
import sys
from pathlib import Path
import shutil


def delete_outputs():
    """Delete existing CSV and PNG outputs, keeping folder structure."""
    output_dir = Path('Output')
    figures_dir = output_dir / 'Figures'
    
    print("Cleaning existing outputs...")
    
    # Delete CSV files in Output/ (but keep the directory)
    if output_dir.exists():
        for csv_file in output_dir.glob('*.csv'):
            csv_file.unlink()
            print(f"  Deleted {csv_file}")
    
    # Delete PNG files in Output/Figures/ (but keep the directory)
    if figures_dir.exists():
        for png_file in figures_dir.glob('*.png'):
            png_file.unlink()
            print(f"  Deleted {png_file}")
    
    print("[OK] Outputs cleaned\n")


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n[OK] {description} completed\n")


def main():
    """Main entrypoint."""
    print("="*60)
    print("2020 DemandPack and Dispatch - Full Pipeline")
    print("="*60)
    print()
    
    # Step 1: Clean outputs
    delete_outputs()
    
    # Step 2: Read total site capacity from utilities
    import pandas as pd
    util_df = pd.read_csv('Input/site_utilities_2020.csv')
    total_capacity_MW = util_df['max_heat_MW'].sum()
    print(f"Total site capacity: {total_capacity_MW:.2f} MW")
    
    # Step 3: Generate DemandPack with peak cap
    run_command(
        ['python', '-m', 'src.generate_demandpack', '--config', 'Input/demandpack_config.toml',
         '--cap-peak-mw', str(total_capacity_MW)],
        "Step 1: Generate DemandPack hourly demand profile"
    )
    
    # Step 4: Generate DemandPack diagnostic figures
    run_command(
        ['python', '-m', 'src.plot_demandpack_diagnostics', 
         '--full-diagnostics', 
         '--config', 'Input/demandpack_config.toml',
         '--data', 'Output/hourly_heat_demand_2020.csv'],
        "Step 2: Generate DemandPack diagnostic figures"
    )
    
    # Step 5: Run optimal subset dispatch with plots (reserve penalties OFF by default)
    run_command(
        ['python', '-m', 'src.site_dispatch_2020',
         '--mode', 'optimal_subset',
         '--plot',
         '--no-load-cost-nzd-per-h', '50.0'],
        "Step 3: Compute optimal subset dispatch and generate plots"
    )
    
    # Step 6: Run regional electricity PoC (if input files exist)
    baseline_path = Path('Input/gxp_baseline_import_hourly_2020.csv')
    capacity_path = Path('Input/gxp_headroom_hourly_2020.csv')
    incremental_path = Path('Output/site_electricity_incremental_2020.csv')
    tariff_path = Path('Input/gxp_tariff_hourly_2020.csv')
    
    if all(p.exists() for p in [baseline_path, capacity_path, tariff_path]):
        if incremental_path.exists():
            run_command(
                ['python', '-m', 'src.regional_electricity_poc',
                 '--epoch', '2020',
                 '--baseline', str(baseline_path),
                 '--capacity', str(capacity_path),
                 '--incremental', str(incremental_path),
                 '--tariff', str(tariff_path),
                 '--out', 'Output/regional_electricity_signals_2020.csv'],
                "Step 4: Compute regional electricity signals (GXP capacity PoC)"
            )
        else:
            print(f"[SKIP] Regional electricity PoC: {incremental_path} not found")
    else:
        print("[SKIP] Regional electricity PoC: Required input files not found")
    
    # Verify all required figures exist
    figures_dir = Path('Output/Figures')
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
    
    print("="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print("\nGenerated outputs:")
    print("  CSVs in Output/:")
    output_dir = Path('Output')
    for csv_file in sorted(output_dir.glob('*.csv')):
        print(f"    - {csv_file.name}")
    
    print("\n  Figures in Output/Figures/:")
    for png_file in sorted(figures_dir.glob('*.png')):
        print(f"    - {png_file.name}")
    print()


if __name__ == '__main__':
    main()

