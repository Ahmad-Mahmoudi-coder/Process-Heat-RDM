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
    
    # Step 2: Generate DemandPack
    run_command(
        ['python', '-m', 'src.generate_demandpack', '--config', 'Input/demandpack_config.toml'],
        "Step 1: Generate DemandPack hourly demand profile"
    )
    
    # Step 3: Generate DemandPack diagnostic figures
    run_command(
        ['python', '-m', 'src.plot_demandpack_diagnostics', 
         '--full-diagnostics', 
         '--config', 'Input/demandpack_config.toml',
         '--data', 'Output/hourly_heat_demand_2020.csv'],
        "Step 2: Generate DemandPack diagnostic figures"
    )
    
    # Step 4: Run optimal subset dispatch with plots
    run_command(
        ['python', '-m', 'src.site_dispatch_2020',
         '--mode', 'optimal_subset',
         '--plot',
         '--reserve-frac', '0.15',
         '--reserve-penalty-nzd-per-MWh', '2000',
         '--no-load-cost-nzd-per-h', '50.0'],
        "Step 3: Compute optimal subset dispatch and generate plots"
    )
    
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

