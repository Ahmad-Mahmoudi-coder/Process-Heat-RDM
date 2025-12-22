# Edendale DemandPack PoC

This repository contains the synthetic hourly heat demand generator and PoC site model for the Edendale site (site_id 11_031) as part of PhD research on industrial electrification and dispatch modelling.

The DemandPack generator produces synthetic hourly heat demand profiles for 2020 using annual targets, seasonal factors, weekday patterns, and hourly profiles. The baseline implementation supports minimal but defensible PoC demand generation to support later site dispatch and electrification epochs.

## Running Scripts

**Recommended usage** (from repository root):

```bash
# Full pipeline (auto-discovers configs)
python -m src.run_all_2020

# With explicit config selection
python -m src.run_all_2020 --demandpack-config Input/configs/demandpack_2020.toml

# With explicit utilities CSV
python -m src.run_all_2020 --utilities-csv Input/site/utilities/site_utilities_2020.csv

# Generate DemandPack only (auto-discovers config)
python -m src.generate_demandpack

# With explicit config
python -m src.generate_demandpack --config Input/configs/demandpack_2020.toml

# Site dispatch
python -m src.site_dispatch_2020 --mode optimal_subset
```

Scripts can also be run directly as files:

```bash
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
  uncertainty/      # Uncertainty scenarios
  weather/          # Weather data files
```

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

## Generating DemandPack 2020 Figures

To generate all diagnostic figures for the 2020 DemandPack, run from the project root:

```bash
python -m src.plot_demandpack_diagnostics --full-diagnostics
```

This will create all figures in `Output/Figures/`:
- `heat_2020_timeseries.png` - Annual hourly time series
- `heat_2020_daily_envelope.png` - Daily min/max envelope
- `heat_2020_hourly_means_by_season.png` - Average hourly profile by season
- `heat_2020_monthly_totals.png` - Monthly totals bar chart
- `heat_2020_LDC.png` - Load duration curve
- `heat_2020_weekday_profiles_Feb.png` - Weekday profiles for February
- `heat_2020_weekday_profiles_Jun.png` - Weekday profiles for June
- `heat_2020_load_histogram.png` - Distribution of hourly load

For quick plotting of just the core timeseries and envelope (saved to `Output/`):

```bash
python -m src.plot_demandpack_diagnostics
```

## Computing 2020 Site Dispatch

The dispatch module supports two modes and auto-discovers the utilities CSV file:

### Utilities CSV Auto-Discovery

The dispatch module automatically finds the utilities CSV in this order:
1. From demandpack config (if `--demandpack-config` is provided and config contains `utilities_csv` field)
2. `<input_dir>/site/utilities/site_utilities_2020.csv`
3. Search `<input_dir>/site/utilities/` for `*utilities*2020*.csv` (if exactly one match, use it)

If multiple utilities files are found, you'll see a numbered list and must specify one:

```bash
# Auto-discover (works if only one utilities file exists)
python -m src.site_dispatch_2020 --mode optimal_subset

# Explicitly specify utilities CSV
python -m src.site_dispatch_2020 --mode optimal_subset --utilities-csv Input/site/utilities/site_utilities_2020.csv
```

### Proportional Mode (Default)

Allocates demand proportionally by unit capacity:

```bash
python -m src.site_dispatch_2020 --mode proportional
```

Outputs:
- `Output/site_dispatch_2020_long.csv` - Long-form dispatch data
- `Output/site_dispatch_2020_wide.csv` - Wide-form dispatch data
- `Output/site_dispatch_2020_long_costed.csv` - Costed long-form data
- `Output/site_dispatch_2020_summary.csv` - Annual summary by unit

### Optimal Subset Mode

Uses unit-commitment-lite logic with optimal subset selection:

```bash
# Weekly commitment blocks (168 hours)
python -m src.site_dispatch_2020 --mode optimal_subset --commitment-block-hours 168 --plot

# Daily commitment blocks (24 hours)
python -m src.site_dispatch_2020 --mode optimal_subset --commitment-block-hours 24 --plot
```

Outputs:
- `Output/site_dispatch_2020_long_costed_opt.csv` - Long-form with all cost components
- `Output/site_dispatch_2020_wide_opt.csv` - Wide-form dispatch
- `Output/site_dispatch_2020_summary_opt.csv` - Annual summary
- `Output/Figures/heat_2020_unit_stack_opt.png` - Stacked dispatch plot
- `Output/Figures/heat_2020_units_online_opt.png` - Units online over time
- `Output/Figures/heat_2020_unit_utilisation_duration_opt.png` - Utilisation duration curves

### Full Pipeline

Run the complete optimal dispatch pipeline:

```bash
python scripts/run_2020_dispatch_optimal.py
```

## Reproducibility

All development is tracked with git commits and tags. See `docs/git_workflow.md` for version control practices and workflow guidance.

