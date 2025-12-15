"""
Runner script for 2020 optimal dispatch pipeline.

Cleans Output/ (except Backup/), regenerates demand if needed, and runs optimal dispatch.
"""

import subprocess
import sys
from pathlib import Path
import shutil

def main():
    """Run the full optimal dispatch pipeline."""
    project_root = Path(__file__).parent.parent
    
    # Change to project root
    import os
    os.chdir(project_root)
    
    print("="*60)
    print("2020 Optimal Dispatch Pipeline")
    print("="*60)
    
    # Clean Output/ except Backup/
    output_dir = Path('Output')
    backup_dir = output_dir / 'Backup'
    
    print("\nCleaning Output/ directory...")
    if output_dir.exists():
        # Preserve Backup/ if it exists
        backup_temp = None
        if backup_dir.exists():
            backup_temp = output_dir.parent / 'Backup_temp'
            shutil.move(str(backup_dir), str(backup_temp))
        
        # Remove Output/
        shutil.rmtree(str(output_dir))
        output_dir.mkdir()
        
        # Restore Backup/
        if backup_temp and backup_temp.exists():
            shutil.move(str(backup_temp), str(backup_dir))
    
    # Check if demand CSV exists, generate if needed
    demand_csv = output_dir / 'hourly_heat_demand_2020.csv'
    if not demand_csv.exists():
        print("\nGenerating demand profile...")
        result = subprocess.run([
            sys.executable, '-m', 'src.generate_demandpack',
            '--config', 'Input/demandpack_config.toml'
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error generating demand: {result.stderr}")
            return 1
        print("Demand profile generated.")
    else:
        print("\nUsing existing demand profile.")
    
    # Run optimal dispatch
    print("\nRunning optimal subset dispatch (weekly blocks)...")
    result = subprocess.run([
        sys.executable, '-m', 'src.site_dispatch_2020',
        '--mode', 'optimal_subset',
        '--commitment-block-hours', '168',
        '--plot'
    ], capture_output=False)
    
    if result.returncode != 0:
        print("\nError running dispatch.")
        return 1
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print("\nOutput files:")
    print("  - Output/site_dispatch_2020_long_costed_opt.csv")
    print("  - Output/site_dispatch_2020_wide_opt.csv")
    print("  - Output/site_dispatch_2020_summary_opt.csv")
    print("\nFigures:")
    print("  - Output/Figures/heat_2020_unit_stack_opt.png")
    print("  - Output/Figures/heat_2020_units_online_opt.png")
    print("  - Output/Figures/heat_2020_unit_utilisation_duration_opt.png")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

