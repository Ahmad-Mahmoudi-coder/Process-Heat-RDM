# PoC Methodological Specification

**Purpose:** This document serves as the definitive methodological specification for the Proof of Concept (PoC) pipeline, providing thesis-ready documentation of methods, data contracts, and limitations. It consolidates and rigorously explains the methodology without replacing existing documentation.

**Cross-references:** This document complements existing documentation:
- [README.md](../README.md): Quickstart, CLI usage, and repository overview
- [docs/POC_PIPELINE.md](POC_PIPELINE.md): One-pass coupling pipeline and folder conventions
- [docs/THESIS_ARTEFACTS.md](THESIS_ARTEFACTS.md): Complete artefact catalogue with schemas
- [docs/THESIS_FIGURES.md](THESIS_FIGURES.md): Figure listing and generation commands
- [docs/git_workflow.md](git_workflow.md): Version control practices

---

## 1. Scope and PoC Claim

### Purpose

The PoC demonstrates site-to-regional signalling with thin-waist artefacts, enabling evaluation of industrial electrification pathways under uncertainty without iterative feedback loops. The architecture separates site-level dispatch from regional grid screening, communicating via standardised CSV interfaces.

### Non-Negotiables

1. **One-pass coupling only:** No iterative feedback between site dispatch and regional screening. The flow is strictly unidirectional: site dispatch → incremental electricity export → regional screening → aggregation.
2. **No feedback optimisation loop:** Regional screening results (upgrade costs, shed penalties) do not feed back into site dispatch quantities or tariffs. Site dispatch uses predetermined signals (tariffs, headroom) from configuration.
3. **Two pathways:** The PoC evaluates two 2035 pathways:
   - **2035_EB:** Electrified Boiler pathway (electric units replace coal)
   - **2035_BB:** Biomass Boiler pathway (biomass units replace coal)

### Regional Layer Definition

In the PoC, the "regional layer" refers to:
- **GXP screening:** Deterministic evaluation of grid capacity constraints using incremental electricity demand from the site
- **Grid upgrade RDM:** Robust Decision Making evaluation of discrete grid upgrade options across uncertainty futures

**What is intentionally out-of-scope:**
- Full electricity market model (e.g., PyPSA-based regional dispatch)
- Iterative tariff updates based on regional screening
- Multi-pass optimisation loops
- Real-time grid feedback to site dispatch
- Full transmission network representation

The regional layer is a **proxy** that demonstrates the signalling mechanism and enables robustness evaluation, not a complete electricity system model.

---

## 2. Repository and Run-Bundle Conventions (Reproducibility)

### Directory Structure

All outputs are organised under `Output/runs/<bundle>/`:

```
Output/runs/<bundle>/
├── epoch<epoch_tag>/
│   ├── demandpack/
│   │   └── demandpack/hourly_heat_demand_<epoch>.csv
│   └── dispatch/<run-id>/
│       ├── site_dispatch_<epoch_tag>_*.csv
│       └── signals/incremental_electricity_MW_<epoch_tag>.csv
├── rdm/
│   ├── rdm_summary_2035_EB.csv
│   ├── rdm_summary_2035_BB.csv
│   ├── rdm_compare_2035_EB_vs_BB.csv
│   └── futures.csv (canonical per bundle)
└── thesis_pack/
    ├── tables/
    ├── figures/
    └── pointers.md
```

### Run-ID Usage

The `run-id` parameter identifies a specific dispatch run within an epoch. Canonical artefact locations are resolved using:
- Bundle name: `poc_YYYYMMDD_HHMMSS` (timestamp-based)
- Epoch tag: `2020`, `2025`, `2028`, `2035_EB`, `2035_BB`
- Run-ID: `dispatch_prop_v2_capfix1` (descriptive identifier)

Scripts use `canonical_output_path()` to resolve paths consistently, supporting both canonical (`epoch<tag>/dispatch/<run-id>/`) and legacy (`epoch<tag>/<run-id>/`) structures.

### CLI Commands for PoC Pipeline

**Complete pipeline (PowerShell):**

```powershell
# Set variables
$bundle = "poc_20260105_115401"
$runId = "dispatch_prop_v2_capfix1"

# Run all layers
.\scripts\run_poc_layers.ps1 -Bundle $bundle -RunId $runId -Layers all
```

**Layer-by-layer (manual):**

```powershell
# 1. Generate DemandPack for all epochs
.\scripts\run_poc_layers.ps1 -Bundle $bundle -Layers demandpack

# 2. Run dispatch (proportional mode) for all epochs
.\scripts\run_poc_layers.ps1 -Bundle $bundle -RunId $runId -Layers dispatch

# 3. Compare 2035 pathways
python -m src.compare_pathways_2035 --bundle $bundle --output-root Output

# 4. Run RDM screening for EB and BB
python -m src.run_rdm_2035 --bundle $bundle --run-id $runId --epoch-tag 2035_EB --output-root Output
python -m src.run_rdm_2035 --bundle $bundle --run-id $runId --epoch-tag 2035_BB --output-root Output

# 5. Create RDM comparison
python -m src.run_rdm_2035 --bundle $bundle --run-id $runId --epoch-tag 2035_EB --output-root Output --create-comparison

# 6. Generate site decision robustness overlay
python -m src.site_decision_robustness --bundle $bundle --init-futures-multipliers
python -m src.site_decision_robustness --bundle $bundle --validate-only
python -m src.site_decision_robustness --bundle $bundle --epoch-eb 2035_EB --epoch-bb 2035_BB

# 7. Curate thesis pack
.\scripts\run_poc_layers.ps1 -Bundle $bundle -Layers thesis_pack
```

**Individual dispatch commands (for specific epochs):**

```bash
python -m src.site_dispatch_2020 --epoch 2035_EB --mode proportional \
  --demand-csv Output/runs/$bundle/epoch2035_EB/demandpack/demandpack/hourly_heat_demand_2035.csv \
  --output-root Output --run-id $bundle/epoch2035_EB/dispatch/$runId
```

---

## 3. Thin-Waist Artefacts (Data Contracts)

Thin-waist artefacts are standardised CSV outputs that serve as interfaces between model layers. Each artefact has a defined schema, units, and producer/consumer contract.

### DemandPack Hourly Heat Demand CSV

**Path:** `Output/runs/<bundle>/epoch<epoch_tag>/demandpack/demandpack/hourly_heat_demand_<epoch>.csv`

**Schema:**
- `timestamp_utc`: Timestamp (UTC, ISO-8601)
- `heat_demand_MW`: Hourly heat demand (MW)

**Units:** MW (average power over the hour)

**Producer:** `src.generate_demandpack`  
**Consumer:** `src.site_dispatch_2020`

**Purpose:** Primary input for site dispatch. Synthetic demand profile generated from annual targets, seasonal factors, weekday patterns, and hourly profiles.

### Dispatch Outputs

#### Long-Form CSV

**Path:** `Output/runs/<bundle>/epoch<epoch_tag>/dispatch/<run-id>/site_dispatch_<epoch_tag>_long.csv`

**Schema:**
- `timestamp_utc`: Timestamp (UTC, ISO-8601)
- `unit_id`: Unit identifier (or `UNSERVED`)
- `heat_MW`: Heat output (MW, average over timestep)
- `fuel_MWh`: Fuel consumption (MWh, energy per timestep)
- `co2_tonnes`: CO2 emissions (tonnes, per timestep)

**Units:** MW (power), MWh (energy), tonnes (mass)

**Producer:** `src.site_dispatch_2020`  
**Consumer:** Costing functions, annual summary aggregation

#### Wide-Form CSV

**Path:** `Output/runs/<bundle>/epoch<epoch_tag>/dispatch/<run-id>/site_dispatch_<epoch_tag>_wide.csv`

**Schema:**
- `timestamp_utc`: Timestamp (UTC, ISO-8601)
- `total_heat_MW`: Total heat demand (MW)
- `unserved_MW`: Unserved demand (MW)
- `<unit_id>_MW`: Heat output per unit (MW, one column per unit)

**Units:** MW

**Producer:** `src.site_dispatch_2020`  
**Consumer:** Visualisation, time-series analysis

#### Summary CSV

**Path:** `Output/runs/<bundle>/epoch<epoch_tag>/dispatch/<run-id>/site_dispatch_<epoch_tag>_summary.csv`

**Schema:**
- `unit_id`: Unit identifier (`TOTAL` = system aggregate, `UNSERVED` = unserved energy)
- `annual_total_cost_nzd`: Total annual cost (NZD)
- `annual_fuel_cost_nzd`: Annual fuel cost (NZD)
- `annual_electricity_cost_nzd`: Annual electricity cost (NZD)
- `annual_electricity_MWh`: Annual electricity consumption (MWh)
- `avg_electricity_tariff_nzd_per_MWh`: Average electricity tariff (NZD/MWh)
- `annual_co2_tonnes`: Annual CO2 emissions (tonnes)
- `annual_carbon_cost_nzd`: Annual carbon cost (NZD)
- `annual_unserved_MWh`: Annual unserved energy (MWh)
- `annual_unserved_cost_nzd`: Annual unserved cost (NZD)

**Units:** NZD (currency), MWh (energy), tonnes (mass)

**Producer:** `src.site_dispatch_2020` (via `compute_annual_summary`)  
**Consumer:** Pathway comparison, robustness overlay, KPI tables

### Incremental Electricity Signal

**Path:** `Output/runs/<bundle>/epoch<epoch_tag>/dispatch/<run-id>/signals/incremental_electricity_MW_<epoch_tag>.csv`

**Schema:**
- `timestamp_utc`: Timestamp (UTC, ISO-8601)
- `incremental_electricity_MW`: Incremental electricity demand (MW)

**Units:** MW

**Producer:** `src.site_dispatch_2020` (exported when electric units are present)  
**Consumer:** `src.run_rdm_2035` (regional screening)

**Purpose:** Exports hourly incremental electricity demand from site dispatch for regional grid screening. This is the primary signal from site to regional layer.

### RDM Outputs

#### RDM Summary CSV

**Path:** `Output/runs/<bundle>/rdm/rdm_summary_2035_<tag>.csv`

**Schema:**
- `epoch_tag`: Epoch tag (`2035_EB` or `2035_BB`)
- `strategy_id`: Strategy ID (`S_AUTO` for auto-select benchmark)
- `strategy_label`: Strategy label
- `future_id`: Future ID (from `Input/rdm/futures_2035.csv`)
- `selected_upgrade_name`: Selected upgrade option name
- `selected_capacity_MW`: Selected upgrade capacity (MW)
- `annualised_upgrade_cost_nzd`: Annualised upgrade cost (NZD/year)
- `annual_incremental_MWh`: Annual incremental electricity demand (MWh)
- `annual_shed_MWh`: Annual shed energy (MWh)
- `shed_fraction`: Shed fraction (0-1)
- `annual_shed_cost_nzd`: Annual shed cost (NZD)
- `total_cost_nzd`: Total cost (upgrade + shed, NZD)
- `n_hours_binding`: Number of hours with binding constraints
- `unserved_peak_MW`: Peak unserved demand (MW)

**Units:** MW, MWh, NZD

**Producer:** `src.run_rdm_2035` (via `src.gxp_rdm_screen.run_rdm_screen`)  
**Consumer:** Robustness overlay, comparison summaries

#### RDM Matrix CSV

**Path:** `Output/runs/<bundle>/rdm/rdm_matrix_2035_<tag>.csv`

**Schema:** Same as summary, plus:
- All upgrade strategies as rows (S0, S1, S2, S3, S4, S_AUTO)
- `regret_vs_benchmark_nzd`: Regret vs S_AUTO benchmark (NZD)
- `satisficing_pass`: Boolean indicating if strategy satisfies constraints

**Producer:** `src.run_rdm_2035` (via `src.gxp_rdm_screen.run_rdm_matrix`)  
**Consumer:** Detailed robustness analysis

#### RDM Comparison CSV

**Path:** `Output/runs/<bundle>/rdm/rdm_compare_2035_EB_vs_BB.csv`

**Schema:**
- `future_id`: Future ID (shared across EB and BB)
- `selected_upgrade_name_EB`, `selected_upgrade_name_BB`: Selected upgrades
- `total_cost_nzd_EB`, `total_cost_nzd_BB`: Total costs
- (Other metrics with `_EB` and `_BB` suffixes)

**Producer:** `src.run_rdm_2035` (via `create_comparison_summary`)  
**Consumer:** Direct pathway comparison

### Canonical Futures CSV

**Path:** `Output/runs/<bundle>/rdm/futures.csv`

**Schema:**
- `future_id`: Future ID (integer, must match RDM summaries)
- **Grid multipliers (from template):**
  - `U_headroom_mult`: Headroom multiplier
  - `U_inc_mult`: Incremental demand multiplier
  - `U_upgrade_capex_mult`: Upgrade capex multiplier
  - `U_voll`: Value of lost load (NZD/MWh)
  - `U_consents_uplift`: Consents uplift multiplier
- **Site multipliers (generated):**
  - `P_elec_mult`: Electricity price multiplier
  - `P_biomass_mult`: Biomass fuel price multiplier
  - `ETS_mult`: Carbon price (ETS) multiplier
  - `D_heat_mult`: Heat demand multiplier (placeholder, not applied in PoC v2)

**Units:** Dimensionless multipliers, NZD/MWh for VOLL

**Producer:** Created from template `Input/rdm/futures_2035.csv` on first `--init-futures-multipliers` run  
**Consumer:** Robustness overlay (applies multipliers to site costs)

**Paired Futures Requirement:** The canonical `futures.csv` must contain exactly the same `future_id` set as both `rdm_summary_2035_EB.csv` and `rdm_summary_2035_BB.csv` to ensure fair comparison.

### Thesis Pack Outputs

**Path:** `Output/runs/<bundle>/thesis_pack/`

**Structure:**
- `tables/`: Curated CSV tables for thesis
- `figures/`: Curated PNG figures for thesis
- `pointers.md`: Mapping of curated items to source paths

**Producer:** `scripts/run_poc_layers.ps1 -Layers thesis_pack`  
**Consumer:** Thesis writing, presentation

---

## 4. Site Demand Layer (DemandPack) – Method Summary

### Synthetic Demand Generation

The DemandPack generator produces synthetic hourly heat demand profiles using a deterministic baseline approach:

1. **Annual target:** Total annual heat demand (GWh/year) from configuration
2. **Seasonal factors:** Monthly scaling factors from `Input/factors/seasonal_factors.csv`
3. **Weekday patterns:** Day-of-week multipliers from `Input/factors/weekday_factors.csv`
4. **Hourly profiles:** Hour-of-day patterns from `Input/factors/daily_factors.csv`

**Formula (simplified):**

\[
Q_{t} = Q_{\text{annual}} \cdot f_{\text{seasonal}}(m) \cdot f_{\text{weekday}}(d) \cdot f_{\text{hourly}}(h)
\]

where:
- \(Q_{t}\) is hourly heat demand (MW) at time \(t\)
- \(Q_{\text{annual}}\) is the annual target (normalised to hourly average)
- \(f_{\text{seasonal}}(m)\) is the seasonal factor for month \(m\)
- \(f_{\text{weekday}}(d)\) is the weekday factor for day-of-week \(d\)
- \(f_{\text{hourly}}(h)\) is the hourly factor for hour-of-day \(h\)

### Assumptions and Limitations

1. **Deterministic baseline:** No stochastic variation or temperature sensitivity in the PoC implementation
2. **Synthetic data:** Profiles are generated from factors, not measured site data
3. **No demand response:** Demand is exogenous and does not respond to prices or grid constraints
4. **Annual closure:** Total annual demand matches the target exactly (no rounding drift)

### Real Data Integration

If real measured data becomes available, it can be integrated as a drop-in replacement via the same CSV contract:
- Same schema: `timestamp_utc`, `heat_demand_MW`
- Same units: MW (average power over the hour)
- Same temporal resolution: Hourly timesteps

The dispatch layer is agnostic to the source of demand data; it only requires the CSV contract to be satisfied.

---

## 5. Site Dispatch Layer – Methods and Modes

The site dispatch layer allocates hourly heat demand across available units using one of three modes. **Only proportional mode is used for reported PoC results.** The other two modes are implemented but not required for PoC reporting.

### 5.1 Proportional Mode (USED IN REPORTED POC RESULTS)

**Function:** `allocate_baseline_dispatch()` in `src/site_dispatch_2020.py`

**Allocation Rule:**

For each timestep \(t\), heat is allocated proportionally by unit capacity:

\[
q_{u,t} = Q_t^{\text{served}} \cdot \frac{P_u \cdot a_u \cdot m_{u,t}}{\sum_{v} P_v \cdot a_v \cdot m_{v,t}}
\]

where:
- \(q_{u,t}\) is heat output from unit \(u\) at time \(t\) (MW)
- \(Q_t^{\text{served}} = \min(Q_t^{\text{demand}}, \sum_{v} P_v \cdot a_v \cdot m_{v,t})\) is served demand (capped at available capacity)
- \(P_u\) is nameplate capacity of unit \(u\) (MW)
- \(a_u\) is static availability factor of unit \(u\) (from utilities CSV)
- \(m_{u,t}\) is time-varying maintenance availability multiplier (0.0 during outages, 1.0 otherwise)

**Constraints:**

1. **Capacity:** \(q_{u,t} \leq P_u \cdot a_u \cdot m_{u,t}\) (unit cannot exceed available capacity)
2. **Maintenance windows:** \(m_{u,t} = 0\) during planned outages (from maintenance CSV)
3. **Energy closure:** \(\sum_{u} q_{u,t} + Q_t^{\text{unserved}} = Q_t^{\text{demand}}\) (demand is either served or unserved)

**Unserved Representation:**

Unserved demand is represented as a virtual unit `UNSERVED` in the long-form output:
- `unit_id = 'UNSERVED'`
- `heat_MW = Q_t^{\text{unserved}}` (unserved demand in MW)
- `fuel_MWh = 0.0` (no fuel consumption)
- `co2_tonnes = 0.0` (no emissions)

Unserved cost is computed as: `annual_unserved_cost_nzd = annual_unserved_MWh * unserved_penalty_nzd_per_MWh` (default penalty: 10,000 NZD/MWh).

### 5.2 Optimal Subset Mode (IMPLEMENTED, NOT REQUIRED FOR POC RESULTS)

**Function:** `allocate_dispatch_optimal_subset()` in `src/site_dispatch_2020.py`

**Commitment Block Concept:**

Units are committed in blocks of fixed duration (default: 168 hours = 1 week). For each block:
1. Enumerate all possible subsets of available units (respecting min up/down time constraints)
2. For each subset, compute hourly dispatch and total cost
3. Choose the subset with minimum total cost
4. Enforce min up/down time constraints across blocks

**Online/No-Load Cost:**

- **Fixed online cost:** `fixed_on_cost_nzd_per_h` (applies when unit is online, regardless of output)
- **No-load cost:** Implicitly included in fixed online cost (no separate no-load term in PoC)
- **Startup cost:** `startup_cost_nzd` (one-time cost when unit turns on)

**Reserve Terms:**

If `reserve_frac > 0`, the model enforces a reserve requirement:
- Reserve requirement: \(R_t = \text{reserve_frac} \cdot Q_t^{\text{demand}}\)
- Reserve available: Headroom = online capacity - total dispatch
- Reserve shortfall: \(\max(0, R_t - \text{headroom}_t)\)
- Reserve penalty: `reserve_shortfall_MW * reserve_penalty_nzd_per_MWh`

**Clarification: Full Optimisation or Heuristic?**

This is a **heuristic search** (not a full MILP optimisation):
- Enumerates all subsets of available units (exponential in number of units)
- For each subset, solves a simple dispatch problem (merit order by marginal cost)
- Selects the subset with minimum total cost
- Respects min up/down time constraints but does not solve a full unit commitment MILP

**Limitations:**
- Exponential complexity in number of units (becomes slow for >10 units)
- No inter-block optimisation (each block is solved independently)
- No transmission constraints or network effects

### 5.3 LP Mode (IMPLEMENTED, NOT REQUIRED FOR POC RESULTS)

**Function:** `allocate_dispatch_lp()` in `src/site_dispatch_2020.py`

**LP Formulation (per hour):**

**Decision Variables:**
- \(x_{u,t} \geq 0\): Heat output from unit \(u\) at time \(t\) (MW)
- \(u_t \geq 0\): Unserved demand at time \(t\) (MW)

**Bounds:**
- \(0 \leq x_{u,t} \leq P_u \cdot a_u \cdot m_{u,t}\) (capacity constraint)
- \(u_t \geq 0\) (non-negativity)

**Equality Constraint:**
\[
\sum_{u} x_{u,t} + u_t = Q_t^{\text{demand}}
\]

This ensures strict hourly heat balance: demand is either served or unserved (no slack).

**Objective:**
\[
\min \sum_{u} c_u \cdot \frac{x_{u,t} \cdot \Delta t}{\eta_u} + \text{VOLL} \cdot u_t \cdot \Delta t
\]

where:
- \(c_u\) is fuel cost for unit \(u\) (NZD/MWh fuel)
- \(\eta_u\) is thermal efficiency of unit \(u\)
- \(\Delta t\) is timestep duration (hours)
- VOLL is value of lost load (default: 10,000 NZD/MWh)

**Clarification: Hourly Independent**

This mode solves a **separate LP for each hour**, with no inter-temporal constraints (no ramping limits, no min up/down time, no startup costs). It is an **optional dispatch kernel** that can be substituted for proportional mode without changing the overall pipeline architecture.

**Where to look in code:**
- `allocate_baseline_dispatch()`: Lines 504-688 in `src/site_dispatch_2020.py`
- `allocate_dispatch_optimal_subset()`: Lines 1139-1650+ in `src/site_dispatch_2020.py`
- `allocate_dispatch_lp()`: Lines 691-1137 in `src/site_dispatch_2020.py`
- `add_costs_to_dispatch()`: Lines 1887-2416 in `src/site_dispatch_2020.py`
- `compute_annual_summary()`: Lines 2417-4652 in `src/site_dispatch_2020.py`
- `validate_dispatch_outputs()`: Lines 4653+ in `src/site_dispatch_2020.py`

---

## 6. Costing and Emissions Accounting (Reporting Layer)

### Cost Components in Dispatch Summaries

**Fuel Cost:**
\[
\text{fuel_cost}_{u,t} = \frac{q_{u,t} \cdot \Delta t}{\eta_u} \cdot p_{\text{fuel},u}
\]

where \(p_{\text{fuel},u}\) is fuel price (NZD/MWh fuel) from signals configuration.

**Variable O&M Cost:**
\[
\text{var_om_cost}_{u,t} = q_{u,t} \cdot \Delta t \cdot \text{var_om}_u
\]

where `var_om_nzd_per_MWh_heat` is variable O&M cost per MWh heat.

**Fixed/Online Cost:**
- Only applies in `optimal_subset` mode: `fixed_on_cost_nzd_per_h` when unit is online
- Not included in proportional mode (assumed zero for PoC reporting)

**Unserved Penalty:**
\[
\text{unserved_cost}_t = Q_t^{\text{unserved}} \cdot \Delta t \cdot \text{VOLL}
\]

where VOLL (value of lost load) defaults to 10,000 NZD/MWh.

### Electricity Cost Handling

**Time-of-Use (ToU) Signals:**

If electricity signals are provided as a DataFrame with `timestamp_utc` and `elec_price_nzd_per_MWh`, the dispatch layer uses time-varying prices. Otherwise, a flat tariff from signals configuration is used.

**Limitations/Warnings:**
- PoC uses deterministic flat tariffs from `Input/signals/signals_config.toml` (no ToU variation)
- Electricity prices do not respond to regional screening results (one-pass coupling)
- No demand response or price elasticity

### Emissions Calculation

**CO2 Emissions:**
\[
\text{CO2}_{u,t} = \frac{q_{u,t} \cdot \Delta t}{\eta_u} \cdot \text{co2_factor}_u
\]

where `co2_factor_t_per_MWh_fuel` is the CO2 emission factor (tonnes CO2 per MWh fuel).

**Carbon Cost:**
\[
\text{carbon_cost}_{u,t} = \text{CO2}_{u,t} \cdot p_{\text{ETS}}
\]

where \(p_{\text{ETS}}\) is the ETS (Emissions Trading Scheme) carbon price (NZD/tonne CO2) from signals configuration.

### ETS Proxy/Multiplier in Robustness Overlay

The site decision robustness overlay (`src/site_decision_robustness.py`) applies a stylised **ETS multiplier** (`ETS_mult`) to carbon costs:

\[
\text{carbon_cost}_{\text{future}} = \text{carbon_cost}_{\text{baseline}} \cdot \text{ETS_mult}
\]

This is a **reporting overlay only**—it does not affect dispatch quantities or core optimisation. The multiplier is generated from a triangular distribution (default range: [0.80, 2.00], mode: 1.0) and applied per-future in the robustness consolidation.

### Reporting Overlays vs Core Logic

**Core Logic (affects dispatch):**
- Fuel costs (affect dispatch in LP/optimal modes)
- Capacity constraints (affect dispatch in all modes)
- Maintenance availability (affects dispatch in all modes)

**Reporting Overlays (do not affect dispatch):**
- ETS multiplier in robustness overlay (applied post-hoc)
- Electricity price multiplier in robustness overlay (applied post-hoc)
- Biomass price multiplier in robustness overlay (applied post-hoc)

---

## 7. GXP Screening + Grid Upgrade RDM (Regional Proxy Layer)

### What `run_rdm_2035` Does

**Inputs:**
1. **Incremental electricity CSV:** `Output/runs/<bundle>/epoch<epoch_tag>/dispatch/<run-id>/signals/incremental_electricity_MW_<epoch_tag>.csv`
2. **Futures CSV:** `Input/rdm/futures_2035.csv` (template with grid multipliers)
3. **Grid upgrades TOML:** `Input/configs/grid_upgrades_southland_edendale.toml` (upgrade menu)

**Outputs:**
1. **RDM summary CSV:** `Output/runs/<bundle>/rdm/rdm_summary_2035_<tag>.csv` (one row per future, S_AUTO strategy)
2. **RDM matrix CSV:** `Output/runs/<bundle>/rdm/rdm_matrix_2035_<tag>.csv` (all strategies for all futures)
3. **RDM comparison CSV:** `Output/runs/<bundle>/rdm/rdm_compare_2035_EB_vs_BB.csv` (side-by-side comparison)

### Futures Parameters

Futures are defined in `Input/rdm/futures_2035.csv` with the following columns:

- **`U_headroom_mult`:** Headroom multiplier (triangular: a=0.75, mode=0.90, b=1.00)
  - Applied to base headroom: `headroom_effective = headroom_base * U_headroom_mult`
- **`U_inc_mult`:** Incremental demand multiplier (triangular: a=0.85, mode=1.00, b=1.15)
  - Applied to incremental electricity: `inc_effective = inc_baseline * U_inc_mult`
- **`U_upgrade_capex_mult`:** Upgrade capex multiplier (triangular: a=0.80, mode=1.00, b=1.50)
  - Applied to upgrade capital costs: `capex_effective = capex_base * U_upgrade_capex_mult`
- **`U_voll`:** Value of lost load (NZD/MWh, discrete: {5000, 10000, 15000, 20000} with weights [0.10, 0.40, 0.35, 0.15])
  - Used for shed cost calculation: `shed_cost = shed_MWh * U_voll`
- **`U_consents_uplift`:** Consents uplift multiplier (triangular: a=1.00, mode=1.20, b=1.60)
  - Applied to upgrade costs: `cost_effective = cost_base * U_consents_uplift`

### Strategies

**Base Strategy (S0):** No upgrade (baseline capacity)

**Upgrade Strategies (S1, S2, S3, S4):** Discrete upgrade options from upgrade menu TOML:
- Each upgrade has: `name`, `capacity_MW`, `capex_nzd`, `lead_time_months`
- Annualised cost: `annual_cost = capex * (r / (1 - (1+r)^{-n}))` where \(r\) is discount rate, \(n\) is asset life

**Auto-Select Strategy (S_AUTO):** Benchmark that selects the upgrade minimising total cost (upgrade + shed) for each future.

### Feasible Definition

A strategy is **feasible** if:
1. **Shedding constraint:** `annual_shed_MWh <= shed_MWh_max` (default: 0.0, no shedding allowed)
2. **Exceedance constraint:** `max_exceed_MW <= max_exceed_MW_max` (default: 0.01 MW tolerance)
3. **Optional shed fraction:** `shed_fraction <= shed_fraction_max` (if specified in config)

A strategy **satisfies** (satisficing) if it meets all constraints. The RDM matrix includes a `satisficing_pass` boolean column.

### Deterministic Headroom Generator Warning

**When it triggers:** If no headroom CSV is provided, the RDM screening uses a deterministic PoC headroom generator with:
- Base headroom: 50 MW
- Seasonal variation: ±20 MW (sine wave, peak in summer)
- Weekly pattern: weekend +5 MW
- Minimum: 10 MW

**Why it exists:** The PoC uses a stylised headroom signal for demonstration. In a full implementation, headroom would come from a regional electricity model (e.g., PyPSA) or measured grid data.

**Warning message:** The RDM runner prints a warning when using deterministic headroom, indicating that results are based on PoC assumptions, not real grid data.

**Where to look in code:**
- `src/run_rdm_2035.py`: Main runner script (lines 235-342)
- `src/gxp_rdm_screen.py`: Core RDM screening logic (`run_rdm_screen`, `run_rdm_matrix`)
- `validate_paired_futures()`: Lines 77-133 in `src/run_rdm_2035.py`
- `create_comparison_summary()`: Lines 136-204 in `src/run_rdm_2035.py`

---

## 8. Robustness Consolidation (EB vs BB)

### What `site_decision_robustness` Does

The robustness consolidation is a **post-processing overlay** that:
1. Joins EB and BB dispatch summaries (TOTAL rows) with EB and BB RDM summaries
2. Loads canonical `futures.csv` (contains both grid and site multipliers)
3. Applies per-future multipliers to site-side cost components:
   - Electricity cost: `elec_cost_future = elec_cost_baseline * P_elec_mult`
   - Biomass cost: `biomass_cost_future = biomass_cost_baseline * P_biomass_mult`
   - Carbon cost: `carbon_cost_future = carbon_cost_baseline * ETS_mult`
4. Computes future-costed system costs: `system_cost_future = site_cost_future + rdm_cost`
5. Evaluates robustness metrics (win rates, regret distributions, satisficing rates)

**Key Concept:** This is **post-processing only**. It does NOT:
- Re-run dispatch
- Modify existing artefacts
- Introduce feedback loops
- Change the pipeline order

### Robustness Metrics

**Regret Definition:**

Regret is the difference between a pathway's system cost and the minimum (winner's) system cost for that future:

\[
\text{regret}_{\text{pathway},f} = \text{system_cost}_{\text{pathway},f} - \min(\text{system_cost}_{\text{EB},f}, \text{system_cost}_{\text{BB},f})
\]

Regret is always non-negative (zero for the winner, positive for the loser).

**CDF Figure:**

The regret CDF (Cumulative Distribution Function) shows the fraction of futures where regret is less than or equal to a given value:
- X-axis: Regret (NZD)
- Y-axis: Cumulative probability (0-100%)
- Two curves: EB (blue) and BB (red)

Lower curves indicate better robustness (less regret across futures).

**Boxplot Meaning:**

The system cost boxplot shows the distribution of future-costed system costs across all futures:
- X-axis: Pathway (EB or BB)
- Y-axis: System cost (NZD)
- Box: Interquartile range (25th-75th percentile)
- Whiskers: 1.5 × IQR
- Outliers: Points beyond whiskers

Lower medians and tighter distributions indicate better robustness.

### Clarification: Post-Processing Only

The robustness screen is **explicitly post-processing**. It:
- Reads existing outputs (does not re-solve)
- Applies multipliers to costs (does not affect dispatch quantities)
- Generates new summary tables and figures (does not modify source artefacts)

This design maintains one-pass coupling: site dispatch and regional screening are independent, and robustness evaluation is a reporting overlay.

**Where to look in code:**
- `src/site_decision_robustness.py`: Main module (lines 1-1260)
- `compute_robustness_table()`: Lines 531-664 in `src/site_decision_robustness.py`
- `compute_summary_metrics()`: Lines 667-735 in `src/site_decision_robustness.py`
- `plot_regret_cdf()`: Lines 738-769 in `src/site_decision_robustness.py`
- `plot_system_cost_boxplot()`: Lines 772-801 in `src/site_decision_robustness.py`
- `decompose_site_costs()`: Lines 279-314 in `src/site_decision_robustness.py`

---

## 9. Validation Gates (Reporting Admission Criteria)

Three validation gates are applied before admitting runs to thesis reporting bundles. These gates ensure data integrity, consistency, and accounting closure.

### Gate 1: Chronology and Schema Validity

**Checks:**
- Timestamps are strictly increasing (no duplicates, no gaps)
- Required columns exist with correct data types
- Units are consistent (MW for power, MWh for energy, NZD for costs)
- Epoch tags match expected values (2020, 2025, 2028, 2035_EB, 2035_BB)

**Implementation:** `validate_dispatch_outputs()` in `src/site_dispatch_2020.py` (lines 4653+)

**FAIL Criteria:**
- Duplicate timestamps
- Missing required columns
- Type mismatches (e.g., string where float expected)
- Invalid epoch tags

### Gate 2: Heat Balance and Annual Closure

**Checks:**
- **Hourly heat balance:** \(\sum_{u} q_{u,t} + Q_t^{\text{unserved}} = Q_t^{\text{demand}}\) for all hours
- **Annual closure:** Total annual heat served + unserved = total annual demand
- **Energy accounting:** Fuel consumption matches heat output via efficiency: \(\text{fuel}_{u,t} = \frac{q_{u,t} \cdot \Delta t}{\eta_u}\)

**Implementation:** `validate_dispatch_outputs()` checks heat balance closure

**FAIL Criteria:**
- Heat balance violation > 1e-6 MW (numerical tolerance)
- Annual closure violation > 1e-3 MWh
- Fuel accounting mismatch > 1% relative error

### Gate 3: Accounting Integrity

**Checks:**
- **Component sums:** `annual_total_cost = annual_fuel_cost + annual_electricity_cost + annual_carbon_cost + annual_unserved_cost + other_cost`
- **Internal consistency:** Electricity cost = electricity consumption × average tariff
- **RDM cost closure:** `total_cost_nzd = annualised_upgrade_cost_nzd + annual_shed_cost_nzd`
- **Paired futures:** EB and BB RDM summaries use identical `future_id` sets

**Implementation:**
- Dispatch: `compute_annual_summary()` validates cost component sums
- RDM: `validate_rdm_summaries()` in `src/site_decision_robustness.py` (lines 494-528)
- Paired futures: `validate_paired_futures()` in `src/run_rdm_2035.py` (lines 77-133)

**FAIL Criteria:**
- Cost component sum mismatch > 1 NZD (absolute tolerance)
- Electricity cost/tariff inconsistency > 1% relative error
- RDM cost closure violation > 1e-6 NZD
- Mismatched `future_id` sets between EB and BB RDM summaries

**Where implemented:**
- `src/site_dispatch_2020.py`: `validate_dispatch_outputs()` (lines 4653+)
- `src/site_dispatch_2020.py`: `compute_annual_summary()` (cost closure checks)
- `src/run_rdm_2035.py`: `validate_paired_futures()` (lines 77-133)
- `src/site_decision_robustness.py`: `validate_rdm_summaries()` (lines 494-528)

---

## 10. Upgrade Roadmap (Layered Extensibility)

The PoC architecture uses thin-waist artefacts to enable upgrades without refactoring. The same CSV contracts allow substitution of individual layers while preserving the overall pipeline structure.

### Data Upgrades

**Measured Headroom:**
- Replace deterministic headroom generator with measured headroom CSV
- Same schema: `timestamp_utc`, `headroom_MW`
- No changes to RDM screening logic (reads CSV directly)

**Measured Tariffs:**
- Replace flat tariff from config with time-varying tariff CSV
- Same schema: `timestamp_utc`, `tariff_nzd_per_MWh`
- Dispatch layer already supports time-varying prices (DataFrame input)

**Improved Emissions Factors:**
- Update `co2_factor_t_per_MWh_fuel` in utilities CSV
- No code changes required (read from CSV)

### Dispatch Kernel Upgrades

**Shift Proportional → LP/MILP:**
- Replace `allocate_baseline_dispatch()` with `allocate_dispatch_lp()` or a new MILP solver
- Same inputs: demand CSV, utilities CSV, signals
- Same outputs: long-form, wide-form, summary CSVs
- No changes to downstream layers (incremental electricity export unchanged)

**Add Unit Commitment:**
- Extend `optimal_subset` mode or implement full MILP unit commitment
- Preserve CSV contracts (outputs remain compatible)
- Add inter-temporal constraints (ramping, min up/down) without breaking interfaces

**Richer Maintenance:**
- Extend maintenance CSV schema (add partial availability, forced outages)
- Update `load_maintenance_availability()` to parse new fields
- Dispatch allocation functions already support time-varying availability matrices

### Electricity Module Substitution

**Replace PoC Screening with PyPSA:**
- Implement PyPSA-based regional electricity module
- Preserve SignalsPack interface: read `incremental_electricity_MW_<epoch_tag>.csv`, write headroom/tariff CSVs
- RDM screening can read headroom from PyPSA outputs (same CSV contract)
- No changes to site dispatch (still reads tariffs from signals config)

**Example interface preservation:**
```python
# PoC screening (current)
headroom = deterministic_headroom_generator(timestamp)

# PyPSA module (future)
headroom = pypsa_model.get_headroom(timestamp, incremental_demand)
# Write to same CSV: signals/headroom_MW_<epoch_tag>.csv
```

### Coupling Upgrades

**Optional Gauss–Seidel Feedback Loop (NOT IN POC):**
- Add iterative loop: site dispatch → regional screening → tariff update → site dispatch (repeat until convergence)
- Requires new orchestration script (not in PoC scope)
- Preserves thin-waist artefacts (each iteration uses same CSV contracts)
- Explicitly excluded from PoC to maintain one-pass coupling claim

### DMDU Upgrades

**PRIM/CART Scenario Discovery:**
- Post-process RDM matrix to identify critical futures using PRIM (Patient Rule Induction Method) or CART (Classification and Regression Trees)
- No changes to core pipeline (uses existing RDM outputs)
- Add new analysis script that reads `rdm_matrix_*.csv`

**Satisficing Thresholds:**
- Already implemented in RDM screening (`satisficing_pass` column)
- Extend thresholds in config (shed fraction, upgrade cost, emissions)
- No code changes required (thresholds are configurable)

**Adaptive Pathways (DAPP):**
- Extend pathway definition to include decision points and signposts
- Add new CSV contract: `pathway_signposts_<epoch_tag>.csv` (timestamp, signpost_name, trigger_value)
- Dispatch layer can read signposts and adjust unit availability accordingly

**Signposts/Triggers:**
- Implement signpost monitoring: if `headroom_MW < threshold`, trigger pathway switch
- Requires new orchestration layer (not in PoC scope)
- Preserves CSV contracts (signposts read from CSV, triggers write to CSV)

---

## 11. Limitations and Interpretation Guidance (For Thesis)

### Valid Conclusions from PoC

**What the PoC demonstrates:**
1. **Signalling mechanism:** Site-to-regional communication via incremental electricity CSV works
2. **Thin-waist architecture:** CSV contracts enable layer substitution without refactoring
3. **Robustness evaluation:** EB vs BB pathway comparison under uncertainty is feasible
4. **One-pass coupling:** Unidirectional flow (site → regional) is sufficient for PoC evaluation

**What the PoC does NOT demonstrate:**
1. **Full electricity market:** Regional layer is a proxy, not a complete market model
2. **Optimal dispatch:** Proportional mode is not cost-optimised (LP/optimal modes exist but are not used for reported results)
3. **Real grid constraints:** Headroom is deterministic/stylised, not from measured data
4. **Demand response:** Demand is exogenous and does not respond to prices or grid signals

### Describing "Regional" in PoC

**Appropriate language:**
- "Regional proxy layer" or "regional screening module"
- "GXP screening with grid upgrade options"
- "Deterministic headroom evaluation"
- "Stylised grid capacity constraints"

**Avoid over-claiming:**
- Do not call it a "full electricity model" or "transmission network model"
- Do not imply it represents real grid operations or market clearing
- Do not suggest it captures all regional electricity system dynamics

**Thesis framing:**
"The PoC uses a regional proxy layer to demonstrate the signalling mechanism and evaluate grid upgrade options under uncertainty. This proxy is not a complete electricity system model but serves to illustrate how site-level electrification decisions can be evaluated against regional grid constraints using thin-waist artefacts."

### Reproducibility and Provenance

**Bundle IDs:**
- All outputs are tagged with bundle name: `poc_YYYYMMDD_HHMMSS`
- Bundle names are timestamp-based for uniqueness and chronological ordering

**Archives:**
- Old overlay outputs are archived to `Output/runs/<bundle>/_archive_overlays/YYYYMMDD_HHMMSS/` before replacement
- DemandPack run manifests include config paths and factor file hashes

**Manifests:**
- DemandPack: `Output/runs/<bundle>/epoch<epoch_tag>/demandpack/meta/run_manifest.json`
- RDM experiments: `Output/runs/<bundle>/epoch<epoch_tag>/dispatch/<run-id>/rdm/<experiment_name>/run_metadata.json`

**Provenance Notes:**
- All scripts log resolved paths and configuration files used
- CSV outputs include epoch tags and run-IDs in filenames
- Thesis pack `pointers.md` maps curated items to source paths

**Reproducibility Requirements:**
- Random seeds are fixed (e.g., multiplier generation uses seed=42 from config)
- Futures generation is reproducible (LHS with fixed seed preserves anchor futures 0-20)
- CLI commands are documented with exact parameters

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Maintained By:** PoC Development Team

