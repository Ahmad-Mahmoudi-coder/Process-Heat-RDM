# Edendale DemandPack PoC

This repository contains the synthetic hourly heat demand generator and PoC site model for the Edendale site (site_id 11_031) as part of PhD research on industrial electrification and dispatch modelling.

The DemandPack generator produces synthetic hourly heat demand profiles for 2020 using annual targets, seasonal factors, weekday patterns, and hourly profiles. The baseline implementation supports minimal but defensible PoC demand generation to support later site dispatch and electrification epochs.

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

The dispatch module supports two modes:

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

