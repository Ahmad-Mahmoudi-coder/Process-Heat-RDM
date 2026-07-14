# Output Repository Inventory

**Generated:** 2026-07-14  
**Repository:** [Process-Heat-RDM](https://github.com/Ahmad-Mahmoudi-coder/Process-Heat-RDM.git)  
**Pre-upload commit (source tree):** `568e6cc2d592189e2c2d105137a4e3baeefe628e`  
**Branch:** `main`

## Purpose

This document records the addition of the **complete local `Output/` directory** to the Git repository so that cloning on another machine restores the full history of generated model results — including historical runs, archives, thesis packs, RDM outputs, dispatch outputs, figures, tables, manifests, and intermediate artefacts.

The previous curated-output policy (restricting tracking to a single PoC bundle) is **not** applied. All existing content below `Output/` is included without deletion, renaming, regeneration, compression, or deduplication.

## Summary

| Metric | Value |
|--------|------:|
| Total files | 2,057 (including `output_file_manifest.csv`) |
| Total size | 1.08 GiB (1,160,498,132 bytes) |
| Ordinary Git files | 2,056 |
| Ordinary Git payload | ~1.08 GiB (1,155,750,344 bytes) |
| Git LFS files | 1 |
| Git LFS payload | 4.5 MiB (4,747,788 bytes) |
| Empty directories (not representable in Git) | 176 |
| Sensitive-data audit | **Clean** |
| Files ≥ 50 MiB | 0 |
| Files ≥ 100 MiB | 0 |
| Files ≥ 1 GiB | 0 |
| Files exceeding GitHub LFS per-file limit (2 GiB) | 0 |

## Top-level Output areas

| Area | File count |
|------|----------:|
| `_archive/` | 1,223 |
| `New folder/` | 429 |
| `runs/` | 404 |
| `output_file_manifest.csv` | 1 |

## Historical run folders (`Output/runs/`)

Nine run/bundle directories:

1. `20260105_214004_epoch2020_demandpack_2020`
2. `20260105_214020_epoch2025_demandpack_2025`
3. `20260105_214021_epoch2028_demandpack_2028`
4. `20260105_214022_epoch2035_demandpack_2035`
5. `20260106_085721_epoch2020_demandpack_2020`
6. `20260106_085749_epoch2025_demandpack_2025`
7. `20260106_085754_epoch2028_demandpack_2028`
8. `20260106_085759_epoch2035_demandpack_2035`
9. **`poc_20260105_release02`** — primary frozen PoC bundle

Oldest historical run: `20260105_214004_epoch2020_demandpack_2020`  
Newest run: `poc_20260105_release02`

## Archive folders

Archive-related directories under `Output/`:

- `_archive/` (root archive tree)
- `New folder/_archive`
- `New folder/runs/_archive`
- `runs/poc_20260105_release02/_archive_overlays`

The `_archive/` tree contains historical PoC bundles (`poc_20260105_115401`, `poc_20260105_release01`), timestamped run snapshots, and intermediate experiment outputs.

## Primary frozen PoC bundle

**Path:** `Output/runs/poc_20260105_release02`

Contains epoch dispatch outputs, regional signals, RDM results, thesis pack, tables, and overlay archives. This bundle remains the canonical frozen release but is tracked alongside all other Output history.

## File types present

| Extension | Count |
|-----------|------:|
| `.png` | 1,115 |
| `.csv` | 777 |
| `.json` | 139 |
| `.md` | 6 |
| `.pdf` | 4 |
| `.zip` | 3 |
| `.log` | 3 |
| `.toml` | 3 |
| `.txt` | 2 |
| `.jsonl` | 2 |
| `.bak` | 1 |
| `.docx` | 1 |

## Largest files (top 10)

| Size (MiB) | Relative path |
|------------:|---------------|
| 6.69 | `_archive/New folder (2)/poc_20260105_115401/epoch2020/dispatch_prop_v2_fix4/site_dispatch_2020_long.csv` |
| 6.69 | `_archive/New folder (2)/poc_20260105_115401/epoch2020/dispatch_prop_v2_fix4/site_dispatch_2020_long_costed.csv` |
| 6.58 | `_archive/New folder (2)/poc_20260105_115401/epoch2025/dispatch_prop_v2_capfix1/site_dispatch_2025_long.csv` |
| 6.58 | `_archive/New folder (2)/poc_20260105_115401/epoch2025/dispatch_prop_v2_capfix1/site_dispatch_2025_long_costed.csv` |
| 6.56 | `_archive/New folder (2)/poc_20260105_115401/epoch2025/dispatch_prop_v2_fix4/site_dispatch_2025_long.csv` |
| 6.56 | `_archive/New folder (2)/poc_20260105_115401/epoch2025/dispatch_prop_v2_fix4/site_dispatch_2025_long_costed.csv` |
| 6.45 | `_archive/New folder (2)/poc_20260105_115401/epoch2028/dispatch_prop_v2_fix4/site_dispatch_2028_long.csv` |
| 6.45 | `_archive/New folder (2)/poc_20260105_115401/epoch2028/dispatch_prop_v2_fix4/site_dispatch_2028_long_costed.csv` |
| 6.44 | `_archive/New folder (2)/poc_20260105_115401/epoch2035_BB/dispatch_prop_v2_capfix1/site_dispatch_2035_BB_long.csv` |
| 6.44 | `_archive/New folder (2)/poc_20260105_115401/epoch2035_BB/dispatch_prop_v2_capfix1/site_dispatch_2035_BB_long_costed.csv` |

No file exceeds 10 MiB. All large dispatch CSV files remain in ordinary Git.

## Storage method assignment

### Ordinary Git (`storage_method: git`)

- All Markdown, TOML, JSON, YAML manifests
- All CSV files (including multi-megabyte dispatch long tables)
- PNG figures (all under 50 MiB)
- PDF, DOCX, LOG, TXT, JSONL, BAK files
- Small ZIP archives under 1 MiB

### Git LFS (`storage_method: git-lfs`)

| File | Size | Reason |
|------|-----:|--------|
| `runs/poc_20260105_release02/rdm/figures/New folder/grid_rdm__figure_manifest.zip` | 4.5 MiB | `Output/**/*.zip` LFS rule (extension policy for binary archives) |

`.gitattributes` also defines LFS tracking for future large Parquet, Feather, HDF5, XLSX, XLSM, ZIP, 7Z, and TIFF files under `Output/`.

## Empty directories

Git cannot preserve 176 empty directories. No `.gitkeep` files were added because these paths are regenerated by the model pipeline and placeholder files could interfere with run logic.

Representative empty paths:

- `runs/poc_20260105_release02/epoch2020/dispatch/dispatch_prop_v2_capfix1/demandpack`
- `runs/poc_20260105_release02/epoch2020/dispatch/dispatch_prop_v2_capfix1/meta`
- `_archive/runs_20251225_104059_20251225_103851_epoch2020_default/.../signals`
- Multiple `figures/`, `demandpack/`, and `meta/` placeholders across historical runs

Full list available in the audit scan (`docs/_output_audit_data.json`, local audit artefact).

## Sensitive-data audit

**Result: Clean**

Scanned text-readable files under `Output/` for credential indicators (`password`, `token`, `api_key`, `private_key`, `connection_string`, `confidential`, etc.) and common secret formats (PEM private keys, AWS/GitHub/OpenAI key patterns). No blocked or review-required findings.

No credentials, API keys, access tokens, or private keys were found. Outputs contain model results, KPI tables, dispatch CSVs, and generated figures only.

## GitHub technical limits

| Check | Status |
|-------|--------|
| Individual file > 100 MiB (ordinary Git forbidden) | None |
| Individual file > 2 GiB (GitHub hard limit) | None |
| Total push payload ~1.08 GiB vs 2 GiB single-push guidance | **Within limit** — single commit/push feasible |
| Git LFS objects | 1 file, 4.5 MiB |

## Files GitHub cannot accept

None identified. All 2,057 files are within GitHub and Git LFS size limits.

## Manifest

Machine-readable inventory: `Output/output_file_manifest.csv`

Columns: `relative_path`, `file_size_bytes`, `sha256`, `extension`, `top_level_output_area`, `run_or_bundle`, `modified_time_local`, `storage_method`, `notes`

## Snapshot commit

This inventory documents the Output snapshot added in commit:

**`Add complete historical Output directory`**

(Update this line with the actual commit hash after push.)

## Clone instructions (second laptop)

```powershell
git clone https://github.com/Ahmad-Mahmoudi-coder/Process-Heat-RDM.git
cd Process-Heat-RDM
git lfs install
git lfs pull
```

### Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r modules/edendale_gxp/requirements.txt
```

The repository uses a Python virtual environment and the `modules/edendale_gxp/requirements.txt` dependency file. No model rerun is required to access historical outputs after clone.
