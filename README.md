# Edendale DemandPack PoC

This repository contains the synthetic hourly heat demand generator and PoC site model for the Edendale site (site_id 11_031) as part of PhD research on industrial electrification and dispatch modelling.

The DemandPack generator produces synthetic hourly heat demand profiles for 2020 using annual targets, seasonal factors, weekday patterns, and hourly profiles. The baseline implementation supports minimal but defensible PoC demand generation to support later site dispatch and electrification epochs.

## PoC Quickstart

**Canonical bundle example:** `poc_20260105_115401`

### Running Dispatch for All Epochs

```bash
# Run dispatch for 2025, 2028, 2035_EB, and 2035_BB
python -m src.site_dispatch_2020 --epoch 2025 --mode proportional --demand-csv Output/runs/poc_20260105_115401/epoch2025/demandpack/demandpack/hourly_heat_demand_2025.csv --output-root Output --run-id poc_20260105_115401/epoch2025/dispatch_prop_v2_capfix1

python -m src.site_dispatch_2020 --epoch 2028 --mode proportional --demand-csv Output/runs/poc_20260105_115401/epoch2028/demandpack/demandpack/hourly_heat_demand_2028.csv --output-root Output --run-id poc_20260105_115401/epoch2028/dispatch_prop_v2_capfix1

python -m src.site_dispatch_2020 --epoch 2035_EB --mode proportional --demand-csv Output/runs/poc_20260105_115401/epoch2035_EB/demandpack/demandpack/hourly_heat_demand_2035.csv --output-root Output --run-id poc_20260105_115401/epoch2035_EB/dispatch_prop_v2_capfix1

python -m src.site_dispatch_2020 --epoch 2035_BB --mode proportional --demand-csv Output/runs/poc_20260105_115401/epoch2035_BB/demandpack/demandpack/hourly_heat_demand_2035.csv --output-root Output --run-id poc_20260105_115401/epoch2035_BB/dispatch_prop_v2_capfix1
```

### Generating KPI Table

```bash
# KPI table is produced by the KPI export command (if available)
# Output location: Output/runs/poc_20260105_115401/kpi_table_capfix1.csv
# Note: This file is kept as-is and not modified by the PoC pipeline
```

### Running Pathway Comparator

```bash
# Compare EB vs BB pathways (using default paths)
python -m src.compare_pathways_2035 --bundle poc_20260105_115401 --output-root Output

# Or with explicit run paths
python -m src.compare_pathways_2035 --bundle poc_20260105_115401 --eb-run epoch2035_EB/dispatch_prop_v2_capfix1 --bb-run epoch2035_BB/dispatch_prop_v2_capfix1 --output-root Output
```

This creates:
- `Output/runs/poc_20260105_115401/compare_pathways_2035_EB_vs_BB.csv`

### Running RDM Screening

```bash
# Run RDM screening for EB pathway
python -m src.run_rdm_2035 --bundle poc_20260105_115401 --run-id dispatch_prop_v2_capfix1 --epoch-tag 2035_EB --output-root Output

# Run RDM screening for BB pathway
python -m src.run_rdm_2035 --bundle poc_20260105_115401 --run-id dispatch_prop_v2_capfix1 --epoch-tag 2035_BB --output-root Output

# Create comparison (after both runs)
python -m src.run_rdm_2035 --bundle poc_20260105_115401 --run-id dispatch_prop_v2_capfix1 --epoch-tag 2035_EB --output-root Output --create-comparison
```

This creates:
- `Output/runs/poc_20260105_115401/rdm/rdm_summary_2035_EB.csv`
- `Output/runs/poc_20260105_115401/rdm/rdm_summary_2035_BB.csv`
- `Output/runs/poc_20260105_115401/rdm/rdm_compare_2035_EB_vs_BB.csv` (if `--create-comparison` is used)

**Note:** Both EB and BB RDM summaries use the **same paired futures** from `Input/rdm/futures_2035.csv` to ensure fair comparison.

### Smoke Tests

```bash
# Validate that TOTAL rows exist in dispatch summaries
python -m src.compare_pathways_2035 --bundle poc_20260105_115401 --output-root Output

# Validate that EB and BB RDM summaries use paired futures (runs automatically after both summaries exist)
python -m src.run_rdm_2035 --bundle poc_20260105_115401 --run-id dispatch_prop_v2_capfix1 --epoch-tag 2035_EB --output-root Output --create-comparison
```

## 2035_BB Pathway Assumptions

The 2035_BB (Biomass Boiler) pathway uses the following assumptions:

- **Biomass fuel cost basis:** Defined in `Input/site/utilities/site_utilities_2035_BB.csv` (fuel_cost_nzd_per_GJ column)
- **Biomass CO2 factor:** Defined in utilities CSV (co2_tonnes_per_GJ column). Biomass is typically treated as carbon-neutral (0 tonnes CO2/GJ) for reporting, but may have upstream emissions factored in.
- **ETS price:** Used for carbon cost calculation. Defined in dispatch configuration or signals config.

**Reporting vs Constraints:**
- **Physically constrained:** Heat demand, unit capacities, maintenance windows, fuel availability
- **Reporting-only:** CO2 emissions and carbon costs (used for comparison but do not affect dispatch quantities)

## Running Scripts

**Recommended usage** (from repository root):

### Epoch-Aware Pipeline (New)

The `run_all.py` script supports multiple epochs (2020, 2025, 2028, 2035):

```bash
# Run pipeline for specific epoch (auto-discovers configs)
python -m src.run_all --epoch 2020
python -m src.run_all --epoch 2025
python -m src.run_all --epoch 2028
python -m src.run_all --epoch 2035

# With explicit config selection
python -m src.run_all --epoch 2020 --demandpack-config Input/configs/demandpack_2020.toml

# With explicit utilities CSV (required for 2035 if multiple variants exist)
python -m src.run_all --epoch 2035 --utilities-csv site/utilities/site_utilities_2035_EB.csv
# Or with Input/ prefix:
python -m src.run_all --epoch 2035 --utilities-csv Input/site/utilities/site_utilities_2035_EB.csv

# With clean/archive options
python -m src.run_all --epoch 2020 --clean --archive

# Run all epochs and variants (recommended for full pipeline)
python -m src.run_all --epochs "2020,2025,2028,2035" --variants-2035 "EB,BB" --archive
```

### Backward-Compatible 2020 Pipeline

The `run_all_2020.py` script is a thin wrapper for backward compatibility:

```bash
# Equivalent to: python -m src.run_all --epoch 2020
python -m src.run_all_2020
```

### Individual Scripts

```bash
# Generate DemandPack (epoch inferred from config name)
python -m src.generate_demandpack --config Input/configs/demandpack_2020.toml

# With explicit epoch
python -m src.generate_demandpack --config Input/configs/demandpack_2025.toml --epoch 2025

# Site dispatch (epoch-aware)
python -m src.site_dispatch_2020 --mode optimal_subset --epoch 2025
```

Scripts can also be run directly as files:

```bash
python .\src\run_all.py --epoch 2020
python .\src\run_all_2020.py
python .\src\generate_demandpack.py
```

Both methods work from any working directory - paths resolve relative to the repository root automatically.

## Input Folder Structure

The repository uses a structured Input folder:

```
Input/
  configs/          # DemandPack configuration files (demandpack_*.toml)
  factors/          # Factor files (daily_factors.csv, seasonal_factors.csv, weekday_factors.csv)
  signals/          # Signals configuration (signals_config.toml, epochs_register.csv, etc.)
  site/             # Site-related CSVs (site_annual.csv, utilities/, etc.)
    utilities/      # Site utilities CSV files (site_utilities_<epoch>.csv, site_utilities_2035_EB.csv, etc.)
    maintenance/   # Planned maintenance windows (optional, one file per epoch/variant)
  uncertainty/      # Uncertainty scenarios
  weather/          # Weather data files
```

### Maintenance Windows (Optional)

Planned maintenance outages can be specified via CSV files in `Input/site/maintenance/`:

- `maintenance_windows_2020.csv`
- `maintenance_windows_2025.csv`
- `maintenance_windows_2028.csv`
- `maintenance_windows_2035_EB.csv`
- `maintenance_windows_2035_BB.csv`

**Schema:**
```csv
unit_id,start_timestamp_utc,end_timestamp_utc,availability
CB1,2020-06-01T00:00:00Z,2020-06-07T23:59:59Z,0
CB2,2020-12-15T00:00:00Z,2020-12-20T23:59:59Z,0
```

**Column Definitions:**
- `unit_id`: Unit identifier (must match utilities CSV)
- `start_timestamp_utc`: Start of maintenance window (ISO-8601 UTC with Z suffix)
- `end_timestamp_utc`: End of maintenance window (ISO-8601 UTC with Z suffix, exclusive)
- `availability`: 0 = unit unavailable (outage), 1 = unit available

**Auto-Discovery:**
The pipeline automatically discovers maintenance files using robust path resolution:
- Uses absolute paths with `.is_file()` check (not just `.exists()`)
- Falls back to case-insensitive directory scan if exact match not found
- For non-variant epochs (2020/2025/2028): `Input/site/maintenance/maintenance_windows_{epoch}.csv`
- For 2035 variants (EB/BB):
  1. First tries: `maintenance_windows_2035_{variant}.csv`
  2. Falls back to: `maintenance_windows_2035.csv`
- Logs `[OK] Using maintenance windows: <ABS_PATH>` when found
- Logs `[INFO] Maintenance windows missing: <ABS_PATH>` when not found
- If a maintenance file is missing, all units are assumed available (no outages)

**Dispatch Model Enforcement:**
The dispatch model enforces: `heat_MW[u,t] <= max_heat_MW[u] * availability_multiplier[u,t]` where `availability_multiplier` is 0.0 during outages, 1.0 otherwise.

### Grid Upgrade Options (Regional Module)

The regional electricity module (`src/regional_electricity_poc.py`) supports discrete grid capacity upgrade options to minimize total cost (upgrade cost + shed cost).

**Configuration File:** `Input/signals/grid_upgrades.toml`

**Schema:**
```toml
[[upgrades]]
capacity_MW = 0
annual_cost_nzd = 0.0
# Default: no upgrade (baseline)

[[upgrades]]
capacity_MW = 10
annual_cost_nzd = 50000.0
# Example: 10 MW upgrade at $50k/year

[[upgrades]]
capacity_MW = 20
annual_cost_nzd = 90000.0
# Example: 20 MW upgrade at $90k/year
```

**Usage:**
```bash
# Regional module with upgrade options (auto-discovers grid_upgrades.toml)
python -m src.regional_electricity_poc --epoch 2025 \
  --gxp-csv modules/edendale_gxp/outputs_latest/gxp_hourly_2025.csv \
  --incremental-csv Output/runs/<run_id>/signals/incremental_electricity_MW_2025.csv \
  --upgrades-config Input/signals/grid_upgrades.toml \
  --voll 10000.0

# Or use default path (Input/signals/grid_upgrades.toml)
python -m src.regional_electricity_poc --epoch 2025 \
  --gxp-csv modules/edendale_gxp/outputs_latest/gxp_hourly_2025.csv \
  --incremental-csv Output/runs/<run_id>/signals/incremental_electricity_MW_2025.csv
```

**Output Columns:**
- `upgrade_selected_MW`: Selected upgrade capacity (MW)
- `upgrade_annual_cost_nzd`: Annual cost of selected upgrade (NZD/year)
- `headroom_effective_MW`: Effective headroom after upgrade (base + upgrade)
- `shed_MW`: Shed demand (should be ~0 if upgrade selected)
- `shed_MWh`: Shed energy (should be ~0 if upgrade selected)

**Decision Logic:**
For each hour, the module chooses the upgrade option that minimizes:
```
total_cost = upgrade_hourly_cost + shed_MWh * VOLL
```
where `upgrade_hourly_cost = annual_cost_nzd / 8760` (prorated to hourly).

If no upgrade options are configured or the file is missing, the module defaults to no upgrade (baseline behavior).

## Robust Decision Making (RDM) Experiments

The RDM module (`src/run_rdm.py`) evaluates grid upgrade strategies across uncertain futures using Latin Hypercube Sampling (LHS) for continuous uncertainties and uniform sampling for discrete uncertainties.

### Running RDM Experiments

```bash
# Basic run with experiment config
python -m src.run_rdm --config Input/configs/rdm_experiment_2035_EB.toml --bundle TEST_run --n-futures 100

# With clean output directory
python -m src.run_rdm --config Input/configs/rdm_experiment_2035_EB.toml --bundle TEST_run --n-futures 100 --clean

# Dry run (creates output directory and metadata, then exits)
python -m src.run_rdm --config Input/configs/rdm_experiment_2035_EB.toml --bundle TEST_run --dry-run
```

### Output Files

RDM experiments generate outputs in `Output/runs/<BUNDLE>/epoch<epoch_tag>/dispatch_prop_v2/rdm/<experiment_name>/`:

- `config_used.toml`: Copy of experiment configuration
- `run_metadata.json`: Run metadata (config paths, timestamps)
- `futures.csv`: Sampled uncertainty futures (one row per future)
- `rdm_summary_<epoch_tag>.csv`: Results for all future-strategy combinations
- `robust_summary.json`: Aggregated statistics (quantiles, regret, satisficing)
- `run_ledger.jsonl`: Minimal provenance per run
- `figures/`: Generated plots
  - `regret_cdf.png`: Step ECDF of regret by strategy (rescaled to billions NZD)
  - `total_cost_boxplot.png`: Boxplot of total cost by strategy (rescaled to billions NZD)
  - `total_cost_boxplot_excl_none.png`: Boxplot excluding S0_NONE strategy
  - `shed_hist.png`: Histogram of annual shed energy by strategy
  - `shed_fraction_hist.png`: Histogram of shed fraction by strategy

### Key Metrics

**Regret:**
- Computed as difference between strategy cost and minimum cost in each future
- Regret is computed against ALL strategies (including AUTO) for reference
- Best strategies (minimax regret, max satisficing) exclude `auto_select` strategies by default
- AUTO strategies are treated as ex-post optimum benchmarks (not implementable strategies)

**Satisficing:**
- Whether strategy meets constraints:
  - `annual_shed_MWh <= shed_MWh_max` (default: 0.0)
  - `max_exceed_MW <= max_exceed_MW_max` (default: 0.01)
  - `shed_fraction <= shed_fraction_max` (optional, if specified in config)

**Derived Metrics:**
- `annual_incremental_MWh`: Total incremental electricity demand (after uncertainty transforms)
- `shed_fraction`: `annual_shed_MWh / annual_incremental_MWh` (fraction of demand that cannot be served)
- `p95_exceed_MW`: 95th percentile of exceedance across timesteps

**Quantiles:**
- p05/p50/p95 of total_cost and annual_shed_MWh by strategy

### Configuration

Experiment configs are TOML files with sections:
- `[experiment]`: name, epoch_tag, n_futures, seed, sampling, dt_h
- `[paths]`: headroom_csv, incremental_csv, upgrade_menu_toml, etc.
- `[[strategies]]`: strategy_id, mode (force|auto_select), force_upgrade_name (optional), label
- `[[uncertainties]]`: u_id, kind, target, bounds/options
- `[metrics]`: thresholds and reporting options
- `[outputs]`: output_root, save flags

**Optional metrics fields in `[metrics]`:**
- `shed_MWh_max` (default: 0.0): Maximum annual shed energy in MWh
- `max_exceed_MW_max` (default: 0.01): Maximum exceedance in MW
- `shed_fraction_max` (optional): Maximum shed fraction (0-1)
- `exclude_auto_from_ranking` (default: true): Exclude auto_select strategies from robust ranking

**Optional path fields in `[paths]`:**
- `signals_config_toml` (optional): Path to signals config (not required for deterministic core)
- `signals_epoch_key` (optional): Epoch key for signals lookup (not required)

### Config File Selection

When multiple demandpack configs exist in `Input/configs/`, scripts will prompt you to specify one:

```bash
# Auto-discover (works if only one config exists)
python -m src.run_all_2020

# Explicitly specify a config
python -m src.run_all_2020 --demandpack-config Input/configs/demandpack_2020.toml
```

If multiple configs are found and none is specified, you'll see a numbered list:

```
[ERROR] Multiple demandpack configs found in Input/configs:
Please specify one using --demandpack-config

  1. demandpack_2020.toml (Input/configs/demandpack_2020.toml)
  2. demandpack_2025.toml (Input/configs/demandpack_2025.toml)
  3. demandpack_2028.toml (Input/configs/demandpack_2028.toml)

Example: --demandpack-config Input/configs/demandpack_2020.toml
```

### Path Resolution

All file paths in TOML configs are resolved in this order:
1. Relative to the config file's directory
2. Relative to `Input/` root
3. Relative to repository root

This ensures configs work regardless of their location within the Input folder structure.

## Generating DemandPack Figures

To generate all diagnostic figures, run from the project root:

```bash
# Epoch inferred from data filename or config name
python -m src.plot_demandpack_diagnostics --full-diagnostics --data Output/hourly_heat_demand_2025.csv

# With explicit epoch
python -m src.plot_demandpack_diagnostics --full-diagnostics --data Output/hourly_heat_demand_2025.csv --epoch 2025
```

This will create all figures in `Output/Figures/` (epoch-tagged):
- `heat_<epoch>_timeseries.png` - Annual hourly time series
- `heat_<epoch>_daily_envelope.png` - Daily min/max envelope
- `heat_<epoch>_hourly_means_by_season.png` - Average hourly profile by season
- `heat_<epoch>_monthly_totals.png` - Monthly totals bar chart
- `heat_<epoch>_LDC.png` - Load duration curve
- `heat_<epoch>_weekday_profiles_Feb.png` - Weekday profiles for February
- `heat_<epoch>_weekday_profiles_Jun.png` - Weekday profiles for June
- `heat_<epoch>_load_histogram.png` - Distribution of hourly load

For quick plotting of just the core timeseries and envelope:

```bash
python -m src.plot_demandpack_diagnostics --data Output/hourly_heat_demand_2025.csv
```

## Computing Site Dispatch

The dispatch module supports three modes and auto-discovers the utilities CSV file:

### Utilities CSV Auto-Discovery

The dispatch module automatically finds the utilities CSV in this order:
1. From demandpack config (if `--demandpack-config` is provided and config contains `utilities_csv` field)
2. `<input_dir>/site/utilities/site_utilities_<epoch>.csv`
3. Search `<input_dir>/site/utilities/` for `*utilities*<epoch>*.csv` (if exactly one match, use it)

**For epoch 2035 with multiple variants** (e.g., `site_utilities_2035_EB.csv` and `site_utilities_2035_BB.csv`):

If multiple utilities files are found for the same epoch, you'll see a numbered list and must specify one:

```bash
# This will show an error with numbered list for 2035 variants
python -m src.run_all --epoch 2035

# Explicitly specify utilities CSV (required for 2035)
# Preferred: relative to repo root (without Input/ prefix)
python -m src.run_all --epoch 2035 --utilities-csv site/utilities/site_utilities_2035_EB.csv

# Also supported: with Input/ prefix
python -m src.run_all --epoch 2035 --utilities-csv Input/site/utilities/site_utilities_2035_EB.csv

# Or specify in demandpack config's utilities_csv field
```

**Path Resolution for `--utilities-csv`:**
Paths are resolved in this order:
1. If absolute: use it directly
2. Else if exists relative to repo root: use it
3. Else if exists relative to Input root: use it
4. Else raise error with attempted paths

Example error message when multiple 2035 utilities exist:
```
[ERROR] Multiple utilities CSV files found for epoch 2035 in Input/site/utilities/:
Please specify one using --utilities-csv

  1. site_utilities_2035_EB.csv (Input/site/utilities/site_utilities_2035_EB.csv)
  2. site_utilities_2035_BB.csv (Input/site/utilities/site_utilities_2035_BB.csv)

Example: --utilities-csv site/utilities/site_utilities_2035_EB.csv
```

### Proportional Mode (Default)

Allocates demand proportionally by unit capacity:

```bash
python -m src.site_dispatch_2020 --mode proportional --epoch 2020
```

Outputs (epoch-tagged):
- `Output/site_dispatch_<epoch>_long.csv` - Long-form dispatch data
- `Output/site_dispatch_<epoch>_wide.csv` - Wide-form dispatch data
- `Output/site_dispatch_<epoch>_long_costed.csv` - Costed long-form data
- `Output/site_dispatch_<epoch>_summary.csv` - Annual summary by unit

### Optimal Subset Mode

Uses unit-commitment-lite logic with optimal subset selection:

```bash
# Weekly commitment blocks (168 hours)
python -m src.site_dispatch_2020 --mode optimal_subset --epoch 2025 --commitment-block-hours 168 --plot

# Daily commitment blocks (24 hours)
python -m src.site_dispatch_2020 --mode optimal_subset --epoch 2028 --commitment-block-hours 24 --plot
```

Outputs (epoch-tagged):
- `Output/site_dispatch_<epoch>_long_costed_opt.csv` - Long-form with all cost components
- `Output/site_dispatch_<epoch>_wide_opt.csv` - Wide-form dispatch
- `Output/site_dispatch_<epoch>_summary_opt.csv` - Annual summary
- `Output/Figures/heat_<epoch>_unit_stack_opt.png` - Stacked dispatch plot
- `Output/Figures/heat_<epoch>_units_online_opt.png` - Units online over time
- `Output/Figures/heat_<epoch>_unit_utilisation_duration_opt.png` - Utilisation duration curves

### Status Column Selection

The dispatch module automatically selects the status column based on epoch:
- Prefers `status_<epoch>` (e.g., `status_2025`, `status_2035`)
- Falls back to `status` if `status_<epoch>` not found
- Prints a warning when using the fallback

This allows utilities CSVs to use epoch-specific status columns (e.g., `status_2035_EB` vs `status_2035_BB`) or a generic `status` column.

## Thesis-Ready Pipeline

The PoC pipeline is organized into layers that can be run independently or all together:

### Layer-by-Layer Runner

Use `scripts/run_poc_layers.ps1` to run the complete pipeline:

```powershell
# Run all layers for all epochs
.\scripts\run_poc_layers.ps1 -Bundle poc_20260105_115401 -RunId dispatch_prop_v2_capfix1 -Layers all

# Run specific layers
.\scripts\run_poc_layers.ps1 -Bundle poc_20260105_115401 -RunId dispatch_prop_v2_capfix1 -Layers "demandpack,dispatch,compare,rdm"

# Run for specific epochs
.\scripts\run_poc_layers.ps1 -Bundle poc_20260105_115401 -Epochs "2020,2025,2035_EB,2035_BB" -Layers dispatch
```

**Available Layers:**
- `demandpack`: Generate demand profiles for all epochs
- `dispatch`: Run site dispatch for all epochs (proportional mode with plots)
- `kpi`: Generate/export KPI table (manual step, script provides info)
- `compare`: Compare 2035_EB vs 2035_BB pathways
- `rdm`: Run RDM screening for 2035_EB and 2035_BB (with paired futures)
- `thesis_pack`: Curate tables and figures into thesis pack folder
- `all`: Run all layers in sequence

### Expected Outputs

After running the pipeline, the following structure is created:

```
Output/runs/<bundle>/
├── epoch<epoch_tag>/
│   ├── demandpack/demandpack/hourly_heat_demand_<epoch>.csv
│   └── dispatch/<runid>/
│       ├── site_dispatch_<epoch_tag>_*.csv (long, long_costed, wide, summary)
│       ├── signals/incremental_electricity_MW_<epoch_tag>.csv (for 2035 epochs)
│       ├── signals_used/tariff_nzd_per_MWh_<epoch_tag>.csv
│       └── figures/heat_<epoch_tag>_unit_stack.png
├── kpi_table_<runid>.csv
├── compare_pathways_2035_EB_vs_BB.csv
├── rdm/
│   ├── rdm_summary_2035_EB.csv
│   ├── rdm_summary_2035_BB.csv
│   └── rdm_compare_2035_EB_vs_BB.csv
└── thesis_pack/
    ├── tables/ (curated CSVs)
    ├── figures/ (curated PNGs)
    └── pointers.md (mapping to source paths)
```

**Complete artefact catalogue:** See [docs/THESIS_ARTEFACTS.md](docs/THESIS_ARTEFACTS.md)  
**Figure listing:** See [docs/THESIS_FIGURES.md](docs/THESIS_FIGURES.md)  
**Decision model:** See [docs/THESIS_DECISION_MODEL.md](docs/THESIS_DECISION_MODEL.md)

### Release Procedure

**1. Clean bundle (if needed):**
```powershell
# Remove existing outputs (optional, for clean run)
Remove-Item -Recurse -Force "Output\runs\<bundle>"
```

**2. Run all layers:**
```powershell
.\scripts\run_poc_layers.ps1 -Bundle <bundle> -RunId dispatch_prop_v2_capfix1 -Layers all
```

**3. Run smoke test:**
```powershell
.\scripts\smoke_test_release.ps1 -Bundle <bundle> -RunId dispatch_prop_v2_capfix1
```

**4. Commit and push:**
```powershell
git add Output/runs/<bundle>
git commit -m "Add PoC results for bundle <bundle>"
git push
```

**Smoke Test Gates:**
- All required artefacts exist
- TOTAL and UNSERVED rows exist in summaries
- Penalties and unserved are zero (release requirement)
- 2035_BB has biomass costs and emissions > 0
- Paired futures validated (EB and BB use identical future_id sets)

### Example: Full Run

```powershell
# Set variables
$bundle = "poc_20260105_115401"
$runId = "dispatch_prop_v2_capfix1"

# Run all layers
.\scripts\run_poc_layers.ps1 -Bundle $bundle -RunId $runId -Layers all

# Validate release gates
.\scripts\smoke_test_release.ps1 -Bundle $bundle -RunId $runId

# If smoke test passes, commit
git add "Output/runs/$bundle"
git commit -m "Add PoC results for bundle $bundle"
git push
```

## Documentation

- **[PoC Pipeline](docs/POC_PIPELINE.md)**: Folder conventions, key artefacts, acceptance checklist
- **[Thesis Decision Model](docs/THESIS_DECISION_MODEL.md)**: Decision structure, uncertainty factors, evaluation metrics, one-pass coupling
- **[Thesis Artefacts](docs/THESIS_ARTEFACTS.md)**: Complete list of output files with schemas
- **[Thesis Figures](docs/THESIS_FIGURES.md)**: Figure listing with generation commands and suggested captions

## Reproducibility

All development is tracked with git commits and tags. See `docs/git_workflow.md` for version control practices and workflow guidance.

