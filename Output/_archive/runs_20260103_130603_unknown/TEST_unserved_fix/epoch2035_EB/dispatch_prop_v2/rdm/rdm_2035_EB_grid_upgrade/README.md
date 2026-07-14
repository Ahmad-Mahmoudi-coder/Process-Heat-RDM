# RDM Experiment: rdm_2035_EB_grid_upgrade

**Epoch**: 2035_EB  
**Futures**: 5  
**Strategies**: 6

## Files

- `config_used.toml`: Copy of experiment configuration
- `run_metadata.json`: Run metadata (config paths, timestamps)
- `futures.csv`: Sampled uncertainty futures (one row per future)
- `rdm_summary.csv`: Results for all future-strategy combinations
- `robust_summary.json`: Aggregated statistics (quantiles, regret, satisficing)
- `run_ledger.jsonl`: Minimal provenance per run
- `figures/`: Generated plots (regret CDF, cost boxplot, shed histogram)

## Key Metrics

- **Regret**: Difference between strategy cost and minimum cost in each future
- **Satisficing**: Whether strategy meets constraints (shed_MWh <= threshold, max_exceed_MW <= threshold)
- **Quantiles**: p05/p50/p95 of total_cost and annual_shed_MWh by strategy

## PoC Assumptions

- Upgrades available instantly upon selection (lead_time_months retained for reporting only)
- Annualised costs computed using CRF from assumptions in upgrade menu TOML
