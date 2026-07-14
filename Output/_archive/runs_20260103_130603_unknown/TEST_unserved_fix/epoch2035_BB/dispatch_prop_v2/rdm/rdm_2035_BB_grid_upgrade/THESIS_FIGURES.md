# Thesis-ready figures for rdm_2035_BB_grid_upgrade (2035_BB)

## Preamble

- **Benchmark strategy**: S_AUTO (ex-post optimum; not deployable ex-ante)
- **Regret baseline for ranking**: best_policy
- **Regret baseline for reporting**: benchmark
- **Satisficing thresholds**:
  - annual_shed_MWh_max = 0.100 MWh
  - shed_fraction_max = 0.0000
  - max_exceed_MW_max = 0.01 MW

## Figures to cite (required)

**A) regret_quantiles.png** (missing)
Robust tail-regret comparison across policy strategies (benchmark framing if benchmark exists).

**B) regret_ecdf_policy.png** (missing)
Distribution of policy regret vs best-policy baseline, highlighting robustness across futures.

**C) regret_ecdf_vs_benchmark.png** (missing)
Policy regret relative to AUTO ex-post benchmark to show value-of-information gap / benchmark framing.

**D) [failure_rate_bar.png](figures/failure_rate_bar.png)**
Failure/satisficing rates under configured thresholds (reliability framing).

**E) exceed_quantiles.png** (missing)
Exceedance risk summary (p50/p95/p99/max exceed_MW) linking to headroom constraint and satisficing.

**F) [shed_hist.png](figures/shed_hist.png)**
Curtailment/shedding severity distribution across strategies (operational feasibility).

## Notes for captioning

- **Benchmark definition**: AUTO = ex-post optimum (not deployable ex-ante)
- **Regret baselines**: Policy regret computed vs best-policy baseline for ranking; benchmark regret computed vs S_AUTO ex-post optimum for reporting.
- **Satisficing constraints**: Strategy passes if annual_shed_MWh ≤ 0.100 MWh, shed_fraction ≤ 0.0000, max_exceed_MW ≤ 0.01 MW.