"""
Path utilities for robust cross-platform path handling.

Provides functions to resolve paths relative to the repository root,
regardless of the current working directory.
"""

from pathlib import Path
import os
from typing import Union


def repo_root() -> Path:
    """
    Returns the repository root by anchoring on this file's location.
    
    Assumes src/ is directly under the repo root.
    
    Returns:
        Path to the repository root directory
    """
    # This file is in src/, so go up one level to get repo root
    this_file = Path(__file__).resolve()
    root = this_file.parent.parent
    return root


def input_root() -> Path:
    """
    Returns the Input directory root (repo_root / "Input").
    
    Returns:
        Path to the Input directory
    """
    return repo_root() / "Input"


def resolve_path(p: Union[str, Path], base: Path = None) -> Path:
    """
    Resolve a path robustly, handling absolute paths, relative paths, and environment variables.
    
    - Expands user (~) and environment variables
    - If absolute, returns as-is
    - If relative, resolves relative to `base` (or repo_root if base is None)
    
    Args:
        p: Path string or Path object to resolve
        base: Base directory for relative paths (default: repo_root)
        
    Returns:
        Resolved absolute Path
    """
    if base is None:
        base = repo_root()
    
    # Convert to Path if string
    path = Path(p) if isinstance(p, str) else p
    
    # Expand user and environment variables
    path_str = str(path)
    path_str = os.path.expanduser(path_str)
    path_str = os.path.expandvars(path_str)
    path = Path(path_str)
    
    # If absolute, return as-is
    if path.is_absolute():
        return path.resolve()
    
    # Otherwise resolve relative to base
    return (base / path).resolve()


def resolve_cfg_path(cfg_path: Union[str, Path], p: Union[str, Path]) -> Path:
    """
    Resolve a path from a config file, with multiple fallback strategies.
    
    Resolution order:
    1. If p is absolute, return as-is
    2. Resolve relative to config file's directory
    3. If not found, resolve relative to Input root (repo_root/Input)
    4. If not found, resolve relative to repo root
    5. If none exist, return the "best guess" path (Input root) for error reporting
    
    This supports the new Input folder structure while maintaining backward compatibility.
    
    Args:
        cfg_path: Path to the config file (used to determine base directory) - can be str or Path
        p: Path string or Path object from config
        
    Returns:
        Resolved absolute Path (may not exist if all fallbacks fail)
    """
    # Normalize cfg_path to Path
    cfg_path = Path(cfg_path).resolve() if isinstance(cfg_path, str) else cfg_path.resolve()
    
    # Convert to Path if string
    path = Path(p) if isinstance(p, str) else p
    
    # Expand user and environment variables
    path_str = str(path)
    path_str = os.path.expanduser(path_str)
    path_str = os.path.expandvars(path_str)
    path = Path(path_str)
    
    # If absolute, return as-is
    if path.is_absolute():
        return path.resolve()
    
    # Try 1: resolve relative to config file's directory
    cfg_dir = cfg_path.parent.resolve()
    candidate1 = (cfg_dir / path).resolve()
    if candidate1.exists():
        return candidate1
    
    # Try 2: resolve relative to Input root
    input_dir = input_root()
    candidate2 = (input_dir / path).resolve()
    if candidate2.exists():
        return candidate2
    
    # Try 3: resolve relative to repo root
    root = repo_root()
    candidate3 = (root / path).resolve()
    if candidate3.exists():
        return candidate3
    
    # None exist: return the "best guess" (Input root) for error reporting
    return candidate2


def canonical_output_path(bundle: str, epoch_tag: str = None, layer: str = None, 
                          runid: str = None, filename: str = None, 
                          output_root: str = "Output") -> Path:
    """
    Generate canonical output paths following the standard folder structure.
    
    Structure: Output/runs/<bundle>/epoch<epoch_tag>/<layer>/<runid>/<filename>
    
    Args:
        bundle: Bundle name (e.g., "poc_20260105_115401")
        epoch_tag: Epoch tag (e.g., "2020", "2035_EB") - optional for bundle-level paths
        layer: Layer name ("demandpack", "dispatch", "regional_electricity") - optional
        runid: Run ID (e.g., "dispatch_prop_v2_capfix1") - optional
        filename: Filename (e.g., "site_dispatch_2035_EB_summary.csv") - optional
        output_root: Output root directory (default: "Output")
        
    Returns:
        Path object for the canonical location
        
    Examples:
        >>> canonical_output_path("poc_20260105_115401", "2035_EB", "dispatch", "dispatch_prop_v2_capfix1", "site_dispatch_2035_EB_summary.csv")
        Path("Output/runs/poc_20260105_115401/epoch2035_EB/dispatch/dispatch_prop_v2_capfix1/site_dispatch_2035_EB_summary.csv")
        
        >>> canonical_output_path("poc_20260105_115401", filename="kpi_table_capfix1.csv")
        Path("Output/runs/poc_20260105_115401/kpi_table_capfix1.csv")
    """
    root = repo_root()
    base_path = root / output_root / "runs" / bundle
    
    if epoch_tag:
        base_path = base_path / f"epoch{epoch_tag}"
    
    if layer:
        base_path = base_path / layer
    
    if runid:
        base_path = base_path / runid
    
    if filename:
        return base_path / filename
    
    return base_path


def ensure_canonical_path(source_path: Path, canonical_path: Path) -> None:
    """
    Ensure a file exists at the canonical path, creating a copy if needed.
    
    This is a compatibility shim to support existing code that writes to non-canonical
    locations. It copies the file to the canonical location if it doesn't already exist.
    
    Args:
        source_path: Path where file currently exists (may be non-canonical)
        canonical_path: Canonical path where file should exist
    """
    if canonical_path.exists():
        return  # Already exists at canonical location
    
    if source_path.exists():
        # Create parent directory if needed
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy file
        import shutil
        shutil.copy2(source_path, canonical_path)
        print(f"[COMPAT] Copied {source_path} to canonical location: {canonical_path}")
    else:
        # Source doesn't exist, create empty file at canonical location for structure
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.touch()
        print(f"[COMPAT] Created placeholder at canonical location: {canonical_path} (source not found: {source_path})")