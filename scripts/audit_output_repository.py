"""Audit Output/ directory for repository inclusion. Read-only scan."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "Output"
MANIFEST_PATH = OUTPUT_ROOT / "output_file_manifest.csv"
INVENTORY_JSON = REPO_ROOT / "docs" / "_output_audit_data.json"

LFS_THRESHOLD = 50 * 1024 * 1024  # 50 MiB
GITHUB_LFS_MAX = 2 * 1024 * 1024 * 1024  # 2 GiB per file (GitHub default)

LFS_EXTENSIONS = {
    ".parquet", ".feather", ".h5", ".hdf5", ".xlsx", ".xlsm",
    ".zip", ".7z", ".tif", ".tiff",
}

SENSITIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"password",
        r"passwd",
        r"\bsecret\b",
        r"\btoken\b",
        r"api[_-]?key",
        r"apikey",
        r"credential",
        r"private[_-]?key",
        r"client[_-]?secret",
        r"connection[_-]?string",
        r"confidential",
        r"restricted",
        r"proprietary",
        r"commercial-in-confidence",
    ]
]

SECRET_FORMAT_PATTERNS = [
    (re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"), "private_key_block"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_access_key_id"),
    (re.compile(r"ghp_[A-Za-z0-9]{36,}"), "github_pat"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "openai_api_key"),
]

TEXT_EXTENSIONS = {
    ".csv", ".json", ".toml", ".md", ".txt", ".log", ".jsonl", ".yaml", ".yml",
    ".xml", ".html", ".htm", ".ini", ".cfg", ".ps1", ".py", ".bat", ".sh",
}


def classify_storage(size_bytes: int, ext: str) -> str:
    if size_bytes >= LFS_THRESHOLD:
        return "git-lfs"
    if ext in LFS_EXTENSIONS and size_bytes >= 1 * 1024 * 1024:
        return "git-lfs"
    return "git"


def top_level_area(rel_path: str) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    return parts[0] if parts else "root"


def run_or_bundle(rel_path: str) -> str:
    norm = rel_path.replace("\\", "/")
    if norm.startswith("runs/"):
        parts = norm.split("/")
        return parts[1] if len(parts) > 1 else "runs"
    if "_archive" in norm:
        return "_archive"
    return top_level_area(rel_path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_files() -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    empty_dirs: list[str] = []

    for dirpath, dirnames, filenames in os.walk(OUTPUT_ROOT):
        rel_dir = Path(dirpath).relative_to(OUTPUT_ROOT)
        if not dirnames and not filenames:
            empty_dirs.append(str(rel_dir).replace("\\", "/") if str(rel_dir) != "." else "")

        for fname in filenames:
            if fname == "output_file_manifest.csv":
                continue
            full = Path(dirpath) / fname
            rel = str(full.relative_to(OUTPUT_ROOT)).replace("\\", "/")
            st = full.stat()
            ext = full.suffix.lower()
            storage = classify_storage(st.st_size, ext)
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            records.append({
                "relative_path": rel,
                "file_size_bytes": st.st_size,
                "sha256": sha256_file(full),
                "extension": ext or "(no ext)",
                "top_level_output_area": top_level_area(rel),
                "run_or_bundle": run_or_bundle(rel),
                "modified_time_local": mtime,
                "storage_method": storage,
                "notes": "",
            })

    return records, empty_dirs


def sensitive_audit(records: list[dict]) -> dict:
    flagged_paths: list[dict] = []
    text_scanned = 0

    for rec in records:
        rel = rec["relative_path"]
        ext = rec["extension"]
        path = OUTPUT_ROOT / rel.replace("/", os.sep)

        if ext in {".xlsx", ".xlsm", ".docx", ".pdf", ".zip", ".png", ".jpg", ".jpeg"}:
            continue

        if ext not in TEXT_EXTENSIONS and ext != "(no ext)":
            continue

        try:
            raw = path.read_bytes()
        except OSError:
            continue

        if b"\x00" in raw[:8192]:
            continue

        text_scanned += 1
        try:
            content = raw.decode("utf-8", errors="ignore")
        except Exception:
            continue

        lower_name = rel.lower()
        for pat in SENSITIVE_PATTERNS:
            if pat.search(lower_name) or pat.search(content):
                flagged_paths.append({
                    "path": rel,
                    "reason": f"pattern:{pat.pattern}",
                    "severity": "review",
                })
                break

        for pat, kind in SECRET_FORMAT_PATTERNS:
            if pat.search(content):
                flagged_paths.append({
                    "path": rel,
                    "reason": kind,
                    "severity": "blocked",
                })
                break

    blocked = [f for f in flagged_paths if f["severity"] == "blocked"]
    review = [f for f in flagged_paths if f["severity"] == "review"]

    if blocked:
        status = "blocked"
    elif review:
        status = "requires_review"
    else:
        status = "clean"

    return {
        "status": status,
        "text_files_scanned": text_scanned,
        "blocked": blocked,
        "review": review,
    }


def write_manifest(records: list[dict]) -> None:
    fieldnames = [
        "relative_path", "file_size_bytes", "sha256", "extension",
        "top_level_output_area", "run_or_bundle", "modified_time_local",
        "storage_method", "notes",
    ]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: r["relative_path"]))


def main() -> int:
    if not OUTPUT_ROOT.exists():
        print("ERROR: Output directory not found", file=sys.stderr)
        return 1

    records, empty_dirs = scan_files()
    sensitive = sensitive_audit(records)

    total_size = sum(r["file_size_bytes"] for r in records)
    git_records = [r for r in records if r["storage_method"] == "git"]
    lfs_records = [r for r in records if r["storage_method"] == "git-lfs"]
    git_size = sum(r["file_size_bytes"] for r in git_records)
    lfs_size = sum(r["file_size_bytes"] for r in lfs_records)

    ext_counts = Counter(r["extension"] for r in records)
    largest = sorted(records, key=lambda r: -r["file_size_bytes"])[:50]

    runs = []
    runs_dir = OUTPUT_ROOT / "runs"
    if runs_dir.exists():
        runs = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])

    archive_folders = sorted({
        str(p.relative_to(OUTPUT_ROOT)).replace("\\", "/")
        for p in OUTPUT_ROOT.rglob("*")
        if p.is_dir() and ("archive" in p.name.lower() or p.name.startswith("_archive"))
    })

    oversized_lfs = [r for r in records if r["file_size_bytes"] > GITHUB_LFS_MAX]

    audit_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(records),
        "total_size_bytes": total_size,
        "git_file_count": len(git_records),
        "git_size_bytes": git_size,
        "lfs_file_count": len(lfs_records),
        "lfs_size_bytes": lfs_size,
        "empty_directories": empty_dirs,
        "empty_directory_count": len(empty_dirs),
        "extensions": dict(ext_counts.most_common()),
        "largest_files": largest,
        "historical_runs": runs,
        "archive_folders": archive_folders,
        "sensitive_audit": sensitive,
        "oversized_lfs_files": [r["relative_path"] for r in oversized_lfs],
        "files_ge_50mib": [r["relative_path"] for r in records if r["file_size_bytes"] >= LFS_THRESHOLD],
        "files_ge_100mib": [r["relative_path"] for r in records if r["file_size_bytes"] >= 100 * 1024 * 1024],
    }

    INVENTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with INVENTORY_JSON.open("w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    write_manifest(records)

    print(json.dumps({
        "total_files": len(records),
        "total_size_gib": round(total_size / 1024**3, 3),
        "git_files": len(git_records),
        "lfs_files": len(lfs_records),
        "empty_dirs": len(empty_dirs),
        "sensitive_status": sensitive["status"],
        "manifest": str(MANIFEST_PATH),
        "audit_json": str(INVENTORY_JSON),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
