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


