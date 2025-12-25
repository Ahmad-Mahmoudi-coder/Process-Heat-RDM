"""
Output path management for standardized run directories.

Provides functions to resolve consistent output paths for runs and archives.
All outputs go to Output/runs/<run_id>/ only (no latest directory).
"""

from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import json
import hashlib
import shutil


def generate_run_id(epoch: int, config_path: Optional[Path] = None) -> str:
    """
    Generate a run ID with timestamp, epoch, and config name.
    
    Format: YYYYMMDD_HHMMSS_epoch{epoch}_{config_stem}
    
    Args:
        epoch: Epoch year
        config_path: Optional config file path (stem used in run_id)
        
    Returns:
        Run ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_stem = ""
    if config_path:
        config_stem = config_path.stem
    else:
        config_stem = "default"
    
    run_id = f"{timestamp}_epoch{epoch}"
    if config_stem:
        run_id += f"_{config_stem}"
    
    return run_id


def resolve_run_paths(
    output_root: Path,
    epoch: int,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None
) -> Dict[str, Path]:
    """
    Resolve all output paths for a run.
    
    Creates directory structure:
    - Output/runs/<run_id>/
    - Output/runs/<run_id>/demandpack/
    - Output/runs/<run_id>/figures/
    - Output/runs/<run_id>/meta/
    - Output/_archive/
    
    Args:
        output_root: Root output directory (default: repo_root / "Output")
        epoch: Epoch year
        config_path: Optional config file path (used for run_id generation)
        run_id: Optional run ID override (if None, auto-generated)
        
    Returns:
        Dictionary with keys:
        - run_dir: Output/runs/<run_id>/
        - run_demandpack_dir: run_dir/demandpack
        - run_figures_dir: run_dir/figures
        - run_meta_dir: run_dir/meta
        - archive_dir: Output/_archive
        - run_id: The run ID used
    """
    if run_id is None:
        run_id = generate_run_id(epoch, config_path)
    
    # Create all directories
    run_dir = output_root / "runs" / run_id
    run_demandpack_dir = run_dir / "demandpack"
    run_figures_dir = run_dir / "figures"
    run_meta_dir = run_dir / "meta"
    
    archive_dir = output_root / "_archive"
    
    # Create directories
    run_demandpack_dir.mkdir(parents=True, exist_ok=True)
    run_figures_dir.mkdir(parents=True, exist_ok=True)
    run_meta_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "run_dir": run_dir,
        "run_demandpack_dir": run_demandpack_dir,
        "run_figures_dir": run_figures_dir,
        "run_meta_dir": run_meta_dir,
        "archive_dir": archive_dir,
        "run_id": run_id,
    }


def read_batch_metadata(output_root: Path) -> Optional[Dict]:
    """
    Read batch metadata from Output/runs/_batch.json.
    
    Args:
        output_root: Root output directory
        
    Returns:
        Dictionary with batch metadata, or None if file doesn't exist
    """
    batch_metadata_path = output_root / "runs" / "_batch.json"
    if not batch_metadata_path.exists():
        return None
    
    try:
        with open(batch_metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read batch metadata: {e}")
        return None


def write_batch_metadata(
    output_root: Path,
    batch_run_id: str,
    epochs: List[int],
    variants_2035: Optional[List[str]] = None
) -> None:
    """
    Write batch metadata to Output/runs/_batch.json.
    
    Args:
        output_root: Root output directory
        batch_run_id: Base batch run ID
        epochs: List of epochs in this batch
        variants_2035: Optional list of 2035 variants (e.g., ["EB", "BB"])
    """
    batch_metadata_path = output_root / "runs" / "_batch.json"
    batch_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "batch_run_id": batch_run_id,
        "created_at": datetime.now().isoformat(),
        "epochs": epochs,
        "variants_2035": variants_2035 or []
    }
    
    with open(batch_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Wrote batch metadata: {batch_metadata_path.relative_to(output_root)}")


def archive_previous_runs(output_root: Path) -> None:
    """
    Archive previous runs by moving all existing run folders from Output/runs/ to Output/_archive/.
    
    Reads batch metadata to name the archive folder. If no metadata exists, uses "unknown".
    
    Args:
        output_root: Root output directory
    """
    runs_dir = output_root / "runs"
    archive_dir = output_root / "_archive"
    
    if not runs_dir.exists():
        return
    
    # Find all run folders (exclude _batch.json and other metadata files)
    run_folders = [
        item for item in runs_dir.iterdir()
        if item.is_dir() and not item.name.startswith('_')
    ]
    
    if not run_folders:
        return
    
    # Read previous batch metadata to get batch ID
    prev_batch_metadata = read_batch_metadata(output_root)
    prev_batch_id = "unknown"
    if prev_batch_metadata and "batch_run_id" in prev_batch_metadata:
        prev_batch_id = prev_batch_metadata["batch_run_id"]
    
    # Create archive directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"runs_{timestamp}_{prev_batch_id}"
    archive_target = archive_dir / archive_name
    
    # Ensure archive directory exists
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # If archive target already exists, add a counter
    if archive_target.exists():
        counter = 1
        while archive_target.exists():
            archive_target = archive_dir / f"runs_{timestamp}_{prev_batch_id}_{counter}"
            counter += 1
    
    print(f"[INFO] Archiving {len(run_folders)} previous run folder(s) to {archive_target.relative_to(output_root)}")
    
    # Move all run folders to archive
    archive_target.mkdir(parents=True, exist_ok=True)
    for run_folder in run_folders:
        dest = archive_target / run_folder.name
        if dest.exists():
            # If destination exists, add a suffix
            counter = 1
            while dest.exists():
                dest = archive_target / f"{run_folder.name}_{counter}"
                counter += 1
        shutil.move(str(run_folder), str(dest))
        print(f"  Moved: {run_folder.name} -> {archive_target.name}/{dest.name}")
    
    # Also move batch metadata if it exists
    batch_metadata_path = runs_dir / "_batch.json"
    if batch_metadata_path.exists():
        dest_metadata = archive_target / "_batch.json"
        shutil.move(str(batch_metadata_path), str(dest_metadata))
        print(f"  Moved: _batch.json -> {archive_target.name}/_batch.json")


def create_run_manifest(
    output_paths: Dict[str, Path],
    epoch: int,
    config_path: Optional[Path],
    output_root: Path,
    factor_files: Optional[Dict[str, Path]] = None
) -> None:
    """
    Create a run manifest JSON file in run_meta_dir.
    
    Args:
        output_paths: Dictionary from resolve_run_paths()
        epoch: Epoch year
        config_path: Config file path (if used)
        output_root: Output root directory
        factor_files: Optional dict of {name: path} for factor CSV files to hash
    """
    manifest = {
        "epoch": epoch,
        "run_id": output_paths["run_id"],
        "created_at": datetime.now().isoformat(),
        "config_path": str(config_path) if config_path else None,
        "output_root": str(output_root),
    }
    
    # Add factor file hashes if provided
    if factor_files:
        manifest["factor_files"] = {}
        for name, path in factor_files.items():
            if path and path.exists():
                try:
                    # Compute SHA256 hash
                    with open(path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    manifest["factor_files"][name] = {
                        "path": str(path),
                        "hash": file_hash
                    }
                except Exception as e:
                    manifest["factor_files"][name] = {
                        "path": str(path),
                        "error": str(e)
                    }
    
    # Write manifest
    manifest_path = output_paths["run_meta_dir"] / "run_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[OK] Created run manifest: {manifest_path.relative_to(output_paths['run_dir'].parents[2])}")


