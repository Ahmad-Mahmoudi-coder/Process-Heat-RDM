"""
Run management utilities for non-destructive output handling.

Provides run directory structure:
- Output/runs/<run_id>/ where run_id = YYYYMMDD_HHMMSS
- Output/latest/ mirrors the most recent run
"""

import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

from src.path_utils import repo_root


def generate_run_id() -> str:
    """
    Generate a run ID in format YYYYMMDD_HHMMSS.
    
    Returns:
        Run ID string, e.g. '20240115_143022'
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_run_dir(run_id: Optional[str] = None) -> Path:
    """
    Get the run directory path for a given run_id.
    
    If run_id is None, generates a new one.
    
    Args:
        run_id: Optional run ID. If None, generates a new one.
        
    Returns:
        Path to the run directory
    """
    if run_id is None:
        run_id = generate_run_id()
    
    ROOT = repo_root()
    OUTPUT_DIR = ROOT / 'Output'
    return OUTPUT_DIR / 'runs' / run_id


def setup_run_dir(run_id: Optional[str] = None) -> Path:
    """
    Create and return the run directory for a new run.
    
    Args:
        run_id: Optional run ID. If None, generates a new one.
        
    Returns:
        Path to the created run directory
    """
    run_dir = get_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Figures subdirectory
    (run_dir / 'Figures').mkdir(parents=True, exist_ok=True)
    
    return run_dir


def update_latest_symlink(run_id: str) -> None:
    """
    Update Output/latest/ to point to the most recent run.
    
    On Windows, creates a text file latest_run.txt with the run_id,
    and copies files to Output/latest/ for compatibility.
    
    Args:
        run_id: Run ID to set as latest
    """
    ROOT = repo_root()
    OUTPUT_DIR = ROOT / 'Output'
    latest_dir = OUTPUT_DIR / 'latest'
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    # Create text file with run_id
    latest_run_file = OUTPUT_DIR / 'latest_run.txt'
    with open(latest_run_file, 'w') as f:
        f.write(run_id)
    
    # Copy files from run directory to latest (for backward compatibility)
    run_dir = get_run_dir(run_id)
    if run_dir.exists():
        # Copy all files and directories
        copy_directory_safe(run_dir, latest_dir)


def copy_directory_safe(src: Path, dst: Path) -> None:
    """
    Copy directory contents safely, handling Windows file locking.
    
    If a file cannot be moved due to WinError 32 (file in use),
    prints a clear message and skips that file.
    
    Args:
        src: Source directory
        dst: Destination directory
    """
    if not src.exists():
        return
    
    # Copy files
    for file_path in src.glob('*'):
        if file_path.is_file():
            dst_file = dst / file_path.name
            try:
                # Remove destination if it exists
                if dst_file.exists():
                    try:
                        dst_file.unlink()
                    except PermissionError:
                        # File is locked, skip it
                        print(f"[SKIP] Cannot overwrite {dst_file.name} (file is locked/in use)")
                        continue
                
                shutil.copy2(file_path, dst_file)
            except (OSError, PermissionError) as e:
                # Handle Windows file locking (WinError 32)
                if hasattr(e, 'winerror') and e.winerror == 32:
                    print(f"[SKIP] Cannot copy {file_path.name} to {dst_file.name} (file is locked/in use)")
                elif isinstance(e, PermissionError):
                    print(f"[SKIP] Cannot copy {file_path.name} to {dst_file.name} (permission denied)")
                else:
                    print(f"[SKIP] Cannot copy {file_path.name} to {dst_file.name}: {e}")
        elif file_path.is_dir():
            # Recursively copy subdirectories
            dst_subdir = dst / file_path.name
            dst_subdir.mkdir(parents=True, exist_ok=True)
            copy_directory_safe(file_path, dst_subdir)


def archive_latest(run_id: Optional[str] = None) -> None:
    """
    Move Output/latest/ into Output/_archive/<run_id>/ before regenerating.
    
    Args:
        run_id: Optional run ID for archive folder. If None, uses current timestamp.
    """
    if run_id is None:
        run_id = generate_run_id()
    
    ROOT = repo_root()
    OUTPUT_DIR = ROOT / 'Output'
    latest_dir = OUTPUT_DIR / 'latest'
    archive_dir = OUTPUT_DIR / '_archive' / run_id
    
    if not latest_dir.exists():
        print(f"[SKIP] No Output/latest/ to archive")
        return
    
    # Check if latest_dir has any files
    if not any(latest_dir.iterdir()):
        print(f"[SKIP] Output/latest/ is empty, nothing to archive")
        return
    
    print(f"Archiving Output/latest/ to {archive_dir}...")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Move files safely
    moved_count = 0
    skipped_count = 0
    
    for item in latest_dir.iterdir():
        archive_item = archive_dir / item.name
        try:
            if item.is_file():
                if archive_item.exists():
                    archive_item.unlink()
                shutil.move(str(item), str(archive_item))
                moved_count += 1
            elif item.is_dir():
                if archive_item.exists():
                    shutil.rmtree(archive_item)
                shutil.move(str(item), str(archive_item))
                moved_count += 1
        except (OSError, PermissionError) as e:
            # Handle Windows file locking
            if hasattr(e, 'winerror') and e.winerror == 32:
                print(f"[SKIP] Cannot move {item.name} (file is locked/in use)")
                skipped_count += 1
            elif isinstance(e, PermissionError):
                print(f"[SKIP] Cannot move {item.name} (permission denied)")
                skipped_count += 1
            else:
                print(f"[SKIP] Cannot move {item.name}: {e}")
                skipped_count += 1
    
    if moved_count > 0:
        print(f"[OK] Archived {moved_count} items to {archive_dir}")
    if skipped_count > 0:
        print(f"[WARNING] Skipped {skipped_count} items due to file locking")


def clean_latest() -> None:
    """
    Delete Output/latest/ directory contents (but keep the directory).
    
    Handles Windows file locking gracefully.
    """
    ROOT = repo_root()
    OUTPUT_DIR = ROOT / 'Output'
    latest_dir = OUTPUT_DIR / 'latest'
    
    if not latest_dir.exists():
        print("[SKIP] Output/latest/ does not exist")
        return
    
    print("Cleaning Output/latest/...")
    deleted_count = 0
    skipped_count = 0
    
    for item in latest_dir.iterdir():
        try:
            if item.is_file():
                item.unlink()
                deleted_count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                deleted_count += 1
        except (OSError, PermissionError) as e:
            # Handle Windows file locking
            if hasattr(e, 'winerror') and e.winerror == 32:
                print(f"[SKIP] Cannot delete {item.name} (file is locked/in use)")
                skipped_count += 1
            elif isinstance(e, PermissionError):
                print(f"[SKIP] Cannot delete {item.name} (permission denied)")
                skipped_count += 1
            else:
                print(f"[SKIP] Cannot delete {item.name}: {e}")
                skipped_count += 1
    
    if deleted_count > 0:
        print(f"[OK] Deleted {deleted_count} items from Output/latest/")
    if skipped_count > 0:
        print(f"[WARNING] Skipped {skipped_count} items due to file locking")


def get_latest_run_id() -> Optional[str]:
    """
    Get the run ID of the most recent run from latest_run.txt.
    
    Returns:
        Run ID string, or None if not found
    """
    ROOT = repo_root()
    OUTPUT_DIR = ROOT / 'Output'
    latest_run_file = OUTPUT_DIR / 'latest_run.txt'
    if latest_run_file.exists():
        with open(latest_run_file, 'r') as f:
            return f.read().strip()
    return None




