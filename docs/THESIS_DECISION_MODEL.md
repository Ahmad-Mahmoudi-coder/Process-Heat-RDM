# Thesis Decision Model Documentation

## Decision Context

**Primary Decision:** 2035 pathway choice (EB vs BB)
- **EB (Electrified Boiler):** High electrification pathway using electric boilers
- **BB (Biomass Boiler):** Biomass boiler pathway with lower electrification

**Site Pathway Decision:** The choice between EB and BB represents a strategic decision at the site level, affecting:
- Electricity consumption patterns
- Fuel costs and carbon emissions
- Grid connection requirements
- Operational flexibility

## Uncertainty Factors

Uncertainty factors are defined in `Input/rdm/futures_2035.csv` and are **shared (paired) across EB and BB pathways** to ensure fair comparison.

### Uncertainty Schema

**File:** `Input/rdm/futures_2035.csv`

**Columns:**
- `future_id`: Unique future identifier (integer)
- `U_headroom_mult`: Headroom multiplier (typically 0.70-1.00)
  - Captures uncertainty in regional grid headroom availability
  - Applied to base headroom time-series
- `U_inc_mult`: Incremental demand multiplier (typically 0.85-1.15)
  - Captures uncertainty in electrification intensity / demand growth
  - Applied to incremental electricity demand from site dispatch
- `U_upgrade_capex_mult`: Upgrade capex multiplier (typically 0.80-1.50)
  - Captures uncertainty in grid upgrade capital costs
  - Applied to all upgrade options (except "none")
- `U_voll`: Value of lost load (NZD/MWh, discrete: 5000, 10000, 15000, 20000)
  - Captures uncertainty in economic value of unserved demand
  - Used to compute shed costs
- `U_consents_uplift`: Consents uplift multiplier (typically 1.00-1.60)
  - Captures uncertainty in consenting costs for grid upgrades
  - Applied to upgrade capex before annualisation

### Deep Uncertainty

**Uncertain Parameters and Ranges:**

| Parameter | Range/Options | Description |
|-----------|---------------|-------------|
| `U_headroom_mult` | 0.70 - 1.00 | Headroom multiplier (tighter headroom due to regional competition, outages, forecast error) |
| `U_inc_mult` | 0.85 - 1.15 | Incremental demand multiplier (electrification intensity / demand growth uncertainty) |
| `U_upgrade_capex_mult` | 0.80 - 1.50 | Upgrade capex multiplier (screening-level capex uncertainty) |
| `U_voll` | 5000, 10000, 15000, 20000 NZD/MWh | Value of lost load (discrete options) |
| `U_consents_uplift` | 1.00 - 1.60 | Consents uplift multiplier (consenting cost uncertainty) |

**Paired Futures Requirement:**

**Critical:** EB and BB pathways must use **identical `future_id` sets** from the same `futures_2035.csv` file. This ensures:
- Fair comparison across pathways
- Same uncertainty realizations for both pathways
- Consistent evaluation framework
- Robustness metrics are directly comparable

The `futures_2035.csv` file contains a fixed set of futures (typically 20-100 futures) that are applied to both EB and BB pathways. This pairing ensures that any differences in outcomes are due to pathway characteristics, not different uncertainty realizations.

## Outputs and Evaluation Metrics

### Site Layer (Dispatch)

**Source:** `Output/runs/<bundle>/epoch<epoch_tag>/dispatch/<runid>/site_dispatch_<epoch_tag>_summary.csv`

**Key Metrics:**

**Cost Metrics:**
- `annual_total_cost_nzd`: Total annual cost (NZD) - sum of all cost components
- `annual_fuel_cost_nzd`: Annual fuel cost (NZD) - cost of delivered fuels (biomass, coal, gas)
- `annual_electricity_cost_nzd`: Annual electricity cost (NZD) - cost of electricity consumption
- `avg_cost_nzd_per_MWh_heat`: Average cost per MWh heat (NZD/MWh) - total cost divided by heat output
- `annual_carbon_cost_nzd`: Annual carbon cost (NZD) - ETS proxy cost based on CO2 emissions

**Electricity Metrics:**
- `annual_electricity_MWh`: Annual electricity consumption (MWh) - total electricity used
- `avg_electricity_tariff_nzd_per_MWh`: Average electricity tariff (NZD/MWh) - weighted average tariff
- `annual_electricity_MWh_effective`: Effective annual electricity consumption (MWh) - uses reported value if non-zero, otherwise derived from EB unit fuel accounting
- `annual_electricity_cost_nzd_effective`: Effective annual electricity cost (NZD) - uses reported value if non-zero, otherwise derived from EB unit fuel accounting
- `avg_electricity_tariff_nzd_per_MWh_effective`: Effective average electricity tariff (NZD/MWh) - computed from effective cost and consumption

**Note:** Electricity effective totals are derived from EB unit fuel accounting when explicit electricity fields are zero. This ensures EB vs BB comparisons are meaningful even when electricity consumption appears under fuel accounting rather than explicit electricity fields.

**Emissions Metrics:**
- `annual_co2_tonnes`: Annual CO2 emissions (tonnes) - total CO2 from fuel combustion
- **Note:** For 2035_BB, biomass is typically treated as carbon-neutral (0 tonnes CO2/GJ) for reporting, but may have upstream emissions factored in

**Penalty/Unserved Metrics:**
- `annual_unserved_MWh`: Annual unserved energy (MWh) - energy that could not be served
- `annual_unserved_cost_nzd`: Annual unserved cost (NZD) - cost of unserved energy (penalty × unserved_MWh)
- `unserved_avg_cost_nzd_per_MWh`: Average unserved cost per MWh (NZD/MWh) - penalty rate used

**Binding Hours:**
- `n_hours_binding`: Number of hours with binding constraints (if available in summary)

**Evaluation:** Compare EB vs BB using pathway comparator (`src/compare_pathways_2035.py`).

### Regional Screen Layer (RDM)

**Source:** `Output/runs/<bundle>/rdm/rdm_summary_<epoch_tag>.csv`

**Key Metrics:**

**Upgrade Metrics:**
- `selected_upgrade_name`: Selected grid upgrade option (e.g., "none", "fonterra_opt1_N_plus21MW")
- `selected_capacity_MW`: Selected upgrade capacity (MW) - additional grid capacity
- `annualised_upgrade_cost_nzd`: Annualised upgrade cost (NZD/year) - CRF × (capex × consents_uplift)

**Shed Metrics:**
- `annual_shed_MWh`: Annual shed energy (MWh) - energy that cannot be served due to grid constraints
- `shed_fraction`: Shed fraction (0-1) - annual_shed_MWh / annual_incremental_MWh
- `annual_shed_cost_nzd`: Annual shed cost (NZD) - annual_shed_MWh × VOLL
- `unserved_peak_MW`: Peak unserved demand (MW) - maximum hourly shortfall

**Total Cost:**
- `total_cost_nzd`: Total cost (upgrade + shed, NZD) - sum of upgrade and shed costs

**Binding Hours:**
- `n_hours_binding`: Number of hours with binding constraints - hours where incremental demand exceeds headroom

**Evaluation:** Compare EB vs BB across futures using RDM comparison (`rdm_compare_2035_EB_vs_BB.csv`). RDM is a **robustness screen** (not an optimization feedback loop) - it evaluates pathway robustness under uncertainty but does not feed back to site dispatch.

### Regional/Grid Robustness Screen

**Purpose:** Evaluate grid upgrade strategies and regional constraints under deep uncertainty.

**Decision Variables:**
- **Upgrade strategies:** Discrete options from upgrade menu (none, +21MW, +32MW, +97MW, +150MW)
- **VOLL (Value of Lost Load):** Economic value of unserved demand (NZD/MWh)
- **Headroom multiplier:** Uncertainty in regional grid headroom availability
- **Capex multiplier:** Uncertainty in grid upgrade capital costs
- **Consents uplift:** Uncertainty in consenting costs for grid upgrades

**Evaluation Logic:**
1. For each future, evaluate all upgrade options
2. Compute total cost = annualised upgrade cost + shed cost (shed_MWh × VOLL)
3. Select upgrade option that minimizes total cost
4. Aggregate results across futures to compute robustness metrics

## One-Pass Coupling Only

**Explicit Statement:** This model implements **one-pass coupling only**. There is no iterative feedback between layers.

**What this means:**
1. **DemandPack → Site Dispatch:** Demand profiles are generated and used as input to site dispatch
2. **Site Dispatch → Incremental Electricity:** Site dispatch computes incremental electricity demand based on site-level constraints only
3. **Incremental Electricity → Regional Screening/RDM:** Regional screening evaluates grid constraints and upgrade options based on incremental demand
4. **Results Aggregation:** Results are aggregated and compared without feedback loops

**What is excluded:**
- Iterative tariff updates from regional screening back to site dispatch
- Feedback from regional constraints to site dispatch quantities
- Multi-pass optimization loops
- Dynamic pricing based on regional congestion
- RDM as an optimization feedback loop (RDM is a robustness screen only)

**Rationale:**
- Ensures deterministic, reproducible results
- Maintains clear separation of concerns
- Provides thesis-defensible methodology
- Avoids circular dependencies in evaluation
- Enables clear interpretation of results (site layer vs regional layer)

## Decision Framework

**Evaluation Process:**
1. Run site dispatch for both EB and BB pathways
2. Export incremental electricity traces
3. Run RDM screening for both pathways using **paired futures**
4. Compare site-layer metrics (cost, emissions, electricity)
5. Compare regional-layer metrics (upgrade costs, shed costs, total costs)
6. Synthesize findings across both layers

**Decision Criteria:**
- Site-layer: Total cost, emissions, electricity consumption
- Regional-layer: Grid upgrade requirements, shed costs, total regional costs
- Combined: Overall system costs and emissions across both layers

