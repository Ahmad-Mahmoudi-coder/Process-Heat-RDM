# Next-input bundle for Electricity PoC (SignalsPack)

This folder contains:
- `signalspack_manifest.toml`: thin-waist mapping from epoch → file paths + column names
- `epochs_registry.csv`: quick audit table (leap-year hours, semantics, file names)
- `incremental_demand_placeholder_*.csv`: zero-demand placeholders (hourly) to let the electricity module run before coupling to the site model

## How to use
1. Copy your generated `gxp_hourly_{epoch}.csv` files into this same folder **or** edit the `gxp_hourly_csv` paths in `signalspack_manifest.toml`.
2. Run your electricity PoC for one epoch by reading:
   - the GXP CSV for headroom/capacity/tariff
   - the incremental demand CSV (placeholder zeros now; replace later)

## Timestamp policy
The placeholder incremental demand files use `YYYY-MM-DD HH:MM:SS` (no timezone suffix).
Interpretation is governed by `time_policy.timestamp_output_tz` in the manifest (default: UTC).

## Notes
- Leap years: 2020 and 2028 have 8784 rows; 2025 and 2035 have 8760 rows.
