"""
GXP SignalsPack Consumer

Consumes GXP hourly signals (headroom, tariff) and site incremental electricity demand
to compute grid constraints, costs, and shedding penalties.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import sys


def load_gxp_hourly(signals_csv_path: Path) -> pd.DataFrame:
    """
    Load and validate GXP hourly signals CSV.
    
    Expected columns:
    gxp_id, timestamp_utc, capacity_mw, baseline_import_mw, reserve_margin_mw,
    headroom_mw, tariff_nzd_per_mwh, epoch
    
    Returns validated DataFrame with parsed timestamps.
    """
    if not signals_csv_path.exists():
        raise FileNotFoundError(f"GXP signals CSV not found: {signals_csv_path}")
    
    df = pd.read_csv(signals_csv_path)
    
    # Validate required columns
    required_cols = [
        'gxp_id', 'timestamp_utc', 'capacity_mw', 'baseline_import_mw',
        'reserve_margin_mw', 'headroom_mw', 'tariff_nzd_per_mwh', 'epoch'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in GXP CSV: {missing}")
    
    # Check for duplicates
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns in GXP CSV: {df.columns[df.columns.duplicated()].tolist()}")
    
    # Parse timestamps
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True, errors='raise')
    
    # Validate timestamps
    if not df['timestamp_utc'].is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonic increasing")
    
    if df['timestamp_utc'].duplicated().any():
        raise ValueError(f"Duplicate timestamps found: {df[df['timestamp_utc'].duplicated()]['timestamp_utc'].tolist()}")
    
    # Compute dt_h from timestamps
    deltas = df['timestamp_utc'].diff().dropna()
    deltas_h = deltas.dt.total_seconds() / 3600.0
    if deltas_h.nunique() != 1:
        raise ValueError(f"Non-constant timestep detected. Unique deltas: {deltas_h.unique()[:5]}")
    dt_h = float(deltas_h.iloc[0])
    
    if abs(dt_h - 1.0) > 1e-6:
        raise ValueError(f"Expected 1-hour timestep, got {dt_h} hours")
    
    # Validate headroom formula
    headroom_calc = (df['capacity_mw'] - df['baseline_import_mw'] - df['reserve_margin_mw']).clip(lower=0.0)
    headroom_diff = (df['headroom_mw'] - headroom_calc).abs()
    max_diff = headroom_diff.max()
    if max_diff > 1e-6:
        print(f"[WARN] Headroom formula mismatch: max abs diff = {max_diff:.6f} MW")
        print(f"  This may be due to floating-point precision. Recomputing headroom from formula.")
        df['headroom_mw'] = headroom_calc
    
    # Ensure numeric columns are numeric
    numeric_cols = ['capacity_mw', 'baseline_import_mw', 'reserve_margin_mw', 'headroom_mw', 'tariff_nzd_per_mwh']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            raise ValueError(f"Non-numeric values in {col}")
    
    print(f"[OK] Loaded GXP signals: {len(df)} rows, dt_h={dt_h:.6f} h")
    return df


def load_incremental_electricity(incremental_csv_path: Path) -> pd.DataFrame:
    """
    Load and validate incremental electricity demand CSV.
    
    Expected columns: timestamp_utc, incremental_electricity_MW
    """
    if not incremental_csv_path.exists():
        raise FileNotFoundError(f"Incremental electricity CSV not found: {incremental_csv_path}")
    
    df = pd.read_csv(incremental_csv_path)
    
    # Validate required columns
    required_cols = ['timestamp_utc']
    if 'incremental_electricity_MW' not in df.columns:
        # Try alternative column name
        if 'incremental_MW' in df.columns:
            df = df.rename(columns={'incremental_MW': 'incremental_electricity_MW'})
        else:
            raise ValueError(f"Missing 'incremental_electricity_MW' or 'incremental_MW' column")
    
    # Parse timestamps
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True, errors='raise')
    
    # Validate timestamps
    if not df['timestamp_utc'].is_monotonic_increasing:
        raise ValueError("Incremental electricity timestamps are not monotonic increasing")
    
    if df['timestamp_utc'].duplicated().any():
        raise ValueError(f"Duplicate timestamps in incremental electricity CSV")
    
    # Ensure numeric
    df['incremental_electricity_MW'] = pd.to_numeric(df['incremental_electricity_MW'], errors='coerce')
    if df['incremental_electricity_MW'].isna().any():
        raise ValueError("Non-numeric values in incremental_electricity_MW")
    
    print(f"[OK] Loaded incremental electricity: {len(df)} rows")
    return df


def load_emissions(emissions_csv_path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load emissions intensity CSV if provided."""
    if emissions_csv_path is None or not emissions_csv_path.exists():
        return None
    
    df = pd.read_csv(emissions_csv_path)
    
    required_cols = ['timestamp_utc', 'grid_co2e_kg_per_mwh_marginal']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing emissions columns: {missing}, skipping emissions")
        return None
    
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True, errors='raise')
    df['grid_co2e_kg_per_mwh_marginal'] = pd.to_numeric(df['grid_co2e_kg_per_mwh_marginal'], errors='coerce')
    
    print(f"[OK] Loaded emissions data: {len(df)} rows")
    return df


def merge_and_compute_costs(
    gxp_df: pd.DataFrame,
    inc_df: pd.DataFrame,
    voll_nzd_per_mwh: float = 10000.0,
    emissions_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Merge GXP signals with incremental electricity and compute costs.
    
    Returns:
        merged_df: DataFrame with all computed columns
        summary_dict: Annual summary statistics
    """
    # Merge on timestamp
    merged = pd.merge(
        gxp_df[['timestamp_utc', 'gxp_id', 'headroom_mw', 'tariff_nzd_per_mwh']],
        inc_df[['timestamp_utc', 'incremental_electricity_MW']],
        on='timestamp_utc',
        how='inner'
    )
    
    # Validate merge
    if len(merged) != len(gxp_df):
        gxp_first = gxp_df['timestamp_utc'].iloc[0]
        gxp_last = gxp_df['timestamp_utc'].iloc[-1]
        inc_first = inc_df['timestamp_utc'].iloc[0]
        inc_last = inc_df['timestamp_utc'].iloc[-1]
        raise ValueError(
            f"Timestamp mismatch after merge: GXP has {len(gxp_df)} rows, merged has {len(merged)} rows.\n"
            f"GXP range: {gxp_first} to {gxp_last}\n"
            f"Incremental range: {inc_first} to {inc_last}"
        )
    
    # Compute dt_h from timestamps
    deltas = merged['timestamp_utc'].diff().dropna()
    deltas_h = deltas.dt.total_seconds() / 3600.0
    dt_h = float(deltas_h.iloc[0])
    
    # Compute metrics
    merged['incremental_electricity_MWh'] = merged['incremental_electricity_MW'] * dt_h
    merged['grid_exceed_MW'] = (merged['incremental_electricity_MW'] - merged['headroom_mw']).clip(lower=0.0)
    merged['shed_MWh'] = merged['grid_exceed_MW'] * dt_h
    merged['electricity_cost_nzd'] = merged['incremental_electricity_MWh'] * merged['tariff_nzd_per_mwh']
    merged['shed_penalty_nzd'] = merged['shed_MWh'] * voll_nzd_per_mwh
    merged['total_grid_cost_nzd'] = merged['electricity_cost_nzd'] + merged['shed_penalty_nzd']
    
    # Add emissions if available
    if emissions_df is not None:
        merged = pd.merge(
            merged,
            emissions_df[['timestamp_utc', 'grid_co2e_kg_per_mwh_marginal']],
            on='timestamp_utc',
            how='left'
        )
        merged['elec_co2e_kg'] = merged['incremental_electricity_MWh'] * merged['grid_co2e_kg_per_mwh_marginal'].fillna(0.0)
    else:
        merged['grid_co2e_kg_per_mwh_marginal'] = np.nan
        merged['elec_co2e_kg'] = np.nan
    
    # Compute annual summary
    summary = {
        'annual_incremental_electricity_MWh': float(merged['incremental_electricity_MWh'].sum()),
        'annual_electricity_cost_nzd': float(merged['electricity_cost_nzd'].sum()),
        'annual_shed_MWh': float(merged['shed_MWh'].sum()),
        'annual_shed_penalty_nzd': float(merged['shed_penalty_nzd'].sum()),
        'annual_total_grid_cost_nzd': float(merged['total_grid_cost_nzd'].sum()),
        'n_hours_binding': int((merged['grid_exceed_MW'] > 0).sum()),
        'pct_hours_binding': float((merged['grid_exceed_MW'] > 0).mean() * 100.0),
        'max_grid_exceed_MW': float(merged['grid_exceed_MW'].max()),
        'p95_grid_exceed_MW': float(np.quantile(merged['grid_exceed_MW'], 0.95)),
        'tariff_stats': {
            'min': float(merged['tariff_nzd_per_mwh'].min()),
            'median': float(merged['tariff_nzd_per_mwh'].median()),
            'max': float(merged['tariff_nzd_per_mwh'].max()),
            'energy_weighted_avg': float(
                (merged['incremental_electricity_MWh'] * merged['tariff_nzd_per_mwh']).sum() /
                merged['incremental_electricity_MWh'].sum()
                if merged['incremental_electricity_MWh'].sum() > 0 else 0.0
            )
        }
    }
    
    if emissions_df is not None and 'elec_co2e_kg' in merged.columns:
        summary['annual_elec_co2e_tonnes'] = float(merged['elec_co2e_kg'].sum() / 1000.0)
    
    return merged, summary


def write_outputs(merged_df: pd.DataFrame, summary_dict: Dict, out_dir: Path, epoch_tag: str) -> None:
    """Write merged CSV and summary JSON to output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Write merged CSV (ordered columns)
    output_cols = [
        'timestamp_utc', 'gxp_id', 'headroom_mw', 'tariff_nzd_per_mwh',
        'incremental_electricity_MW', 'incremental_electricity_MWh',
        'grid_exceed_MW', 'shed_MWh',
        'electricity_cost_nzd', 'shed_penalty_nzd', 'total_grid_cost_nzd'
    ]
    
    # Add emissions columns if present
    if 'grid_co2e_kg_per_mwh_marginal' in merged_df.columns:
        output_cols.extend(['grid_co2e_kg_per_mwh_marginal', 'elec_co2e_kg'])
    
    merged_output = merged_df[output_cols].copy()
    csv_path = out_dir / f'gxp_site_merge_{epoch_tag}.csv'
    merged_output.to_csv(csv_path, index=False)
    print(f"[OK] Saved merged CSV to {csv_path}")
    
    # Write summary JSON
    summaries_dir = out_dir / 'summaries'
    summaries_dir.mkdir(exist_ok=True)
    json_path = summaries_dir / f'gxp_site_cost_summary_{epoch_tag}.json'
    with open(json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"[OK] Saved summary JSON to {json_path}")


def generate_figures(merged_df: pd.DataFrame, figures_dir: Path, epoch_tag: str) -> None:
    """Generate thesis-ready figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping figures")
        return
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Headroom vs incremental (time series sample week + exceed hours)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sample a week (first 168 hours)
    sample_df = merged_df.head(168).copy()
    sample_df['hour_index'] = range(len(sample_df))
    
    ax.plot(sample_df['hour_index'], sample_df['headroom_mw'], label='Headroom (MW)', linewidth=2, color='green')
    ax.plot(sample_df['hour_index'], sample_df['incremental_electricity_MW'], label='Incremental demand (MW)', linewidth=2, color='blue')
    
    # Highlight exceed hours
    exceed_mask = sample_df['grid_exceed_MW'] > 0
    if exceed_mask.any():
        ax.scatter(
            sample_df.loc[exceed_mask, 'hour_index'],
            sample_df.loc[exceed_mask, 'incremental_electricity_MW'],
            color='red', s=50, zorder=5, label='Exceed hours'
        )
    
    ax.set_xlabel('Hour (sample week)')
    ax.set_ylabel('Power (MW)')
    ax.set_title(f'GXP Headroom vs Incremental Demand ({epoch_tag})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = figures_dir / f'fig_gxp_headroom_vs_incremental_{epoch_tag}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_path}")
    
    # Figure 2: Tariff distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(merged_df['tariff_nzd_per_mwh'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Tariff (NZD per MWh)')
    ax.set_ylabel('Frequency (hours)')
    ax.set_title(f'Electricity Tariff Distribution ({epoch_tag})')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig_path = figures_dir / f'fig_tariff_levels_hist_{epoch_tag}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_path}")
    
    # Figure 3: Cost breakdown (stacked annual totals)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    annual_elec_cost = merged_df['electricity_cost_nzd'].sum()
    annual_shed_penalty = merged_df['shed_penalty_nzd'].sum()
    
    categories = ['Electricity cost', 'Shed penalty']
    values = [annual_elec_cost / 1e6, annual_shed_penalty / 1e6]  # Convert to millions
    
    bars = ax.bar(categories, values, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    
    # Annotate bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:.2f}M',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Annual Cost (NZD, millions)')
    ax.set_title(f'Annual Grid Cost Breakdown ({epoch_tag})')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig_path = figures_dir / f'fig_cost_breakdown_{epoch_tag}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_path}")


def extract_eval_year(epoch_tag: str) -> int:
    """Extract evaluation year from epoch tag (e.g., '2035_BB' -> 2035)."""
    # Extract leading digits
    year_str = ''
    for char in epoch_tag:
        if char.isdigit():
            year_str += char
        elif year_str:
            break
    
    if not year_str:
        raise ValueError(f"Could not extract year from epoch_tag: {epoch_tag}")
    
    return int(year_str)


def load_gxp_summary_for_annual_summary(summary_json_path: Path) -> Optional[Dict]:
    """Load GXP cost summary JSON for integration into annual summary."""
    if not summary_json_path.exists():
        return None
    
    with open(summary_json_path, 'r') as f:
        return json.load(f)


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description='GXP SignalsPack Consumer')
    parser.add_argument('--bundle', type=str, required=True, help='Bundle name')
    parser.add_argument('--epoch-tag', type=str, required=True, help='Epoch tag (e.g., 2035_EB)')
    parser.add_argument('--signals-dir', type=Path, required=True, help='Directory containing gxp_hourly_<year>.csv')
    parser.add_argument('--emissions-csv', type=Path, default=None, help='Optional emissions CSV path')
    parser.add_argument('--voll', type=float, default=10000.0, help='Value of lost load (NZD per MWh)')
    parser.add_argument('--output-root', type=Path, default=Path('Output'), help='Output root directory')
    
    args = parser.parse_args()
    
    # Extract eval year
    eval_year = extract_eval_year(args.epoch_tag)
    
    # Auto-discover paths
    gxp_csv = args.signals_dir / f'gxp_hourly_{eval_year}.csv'
    if not gxp_csv.exists():
        raise FileNotFoundError(f"GXP CSV not found: {gxp_csv}")
    
    # Find incremental electricity CSV in run directory (try multiple patterns)
    run_dir_patterns = [
        args.output_root / 'runs' / args.bundle / f'epoch{args.epoch_tag}' / 'dispatch_prop_v2',
        args.output_root / 'runs' / args.bundle / f'epoch{args.epoch_tag}',
        args.output_root / 'runs' / args.bundle
    ]
    
    incremental_csv = None
    run_dir = None
    for pattern in run_dir_patterns:
        candidate = pattern / 'signals' / f'incremental_electricity_MW_{args.epoch_tag}.csv'
        if candidate.exists():
            incremental_csv = candidate
            run_dir = pattern
            break
    
    if incremental_csv is None or not incremental_csv.exists():
        # Try alternative naming
        for pattern in run_dir_patterns:
            candidate = pattern / 'signals' / f'incremental_electricity_MW_{eval_year}.csv'
            if candidate.exists():
                incremental_csv = candidate
                run_dir = pattern
                break
    
    if incremental_csv is None or not incremental_csv.exists():
        raise FileNotFoundError(
            f"Incremental electricity CSV not found. Tried:\n" +
            "\n".join([f"  {p / 'signals' / f'incremental_electricity_MW_{args.epoch_tag}.csv'}" for p in run_dir_patterns])
        )
    
    # Load data
    print(f"[GXP] Loading GXP signals from {gxp_csv}...")
    gxp_df = load_gxp_hourly(gxp_csv)
    
    print(f"[GXP] Loading incremental electricity from {incremental_csv}...")
    inc_df = load_incremental_electricity(incremental_csv)
    
    emissions_df = None
    if args.emissions_csv:
        print(f"[GXP] Loading emissions from {args.emissions_csv}...")
        emissions_df = load_emissions(args.emissions_csv)
    
    # Merge and compute
    print(f"[GXP] Merging and computing costs (VOLL={args.voll} NZD/MWh)...")
    merged_df, summary = merge_and_compute_costs(gxp_df, inc_df, args.voll, emissions_df)
    
    # Write outputs (use run_dir if found, otherwise infer from incremental_csv parent)
    if run_dir is None:
        run_dir = incremental_csv.parent.parent  # signals/ -> run_dir/
    signals_dir = run_dir / 'signals'
    write_outputs(merged_df, summary, signals_dir, args.epoch_tag)
    
    # Generate figures
    figures_dir = run_dir / 'figures'
    print(f"[GXP] Generating figures...")
    generate_figures(merged_df, figures_dir, args.epoch_tag)
    
    # Print summary
    print(f"\n[SUMMARY] GXP SignalsPack consumption for {args.epoch_tag}:")
    print(f"  Annual incremental electricity: {summary['annual_incremental_electricity_MWh']:.2f} MWh")
    print(f"  Annual electricity cost: ${summary['annual_electricity_cost_nzd']/1e6:.2f}M NZD")
    print(f"  Annual shed: {summary['annual_shed_MWh']:.2f} MWh")
    print(f"  Annual shed penalty: ${summary['annual_shed_penalty_nzd']/1e6:.2f}M NZD")
    print(f"  Total grid cost: ${summary['annual_total_grid_cost_nzd']/1e6:.2f}M NZD")
    print(f"  Binding hours: {summary['n_hours_binding']} ({summary['pct_hours_binding']:.1f}%)")
    print(f"  Max exceed: {summary['max_grid_exceed_MW']:.2f} MW")


if __name__ == '__main__':
    main()

