"""
Grid secondary effects plotting (thesis-ready).

Supports the claim:
- Site-side multipliers (P_elec, P_biomass) are primary drivers of flips.
- Grid uncertainties are secondary but not negligible; U_inc is clearest, then U_head.
- Grid mainly matters at the boundary: when EB is competitive on site-side, AUTO grid
  response can still decide whether EB remains competitive.

Read-only: consumes existing outputs only and writes new figures.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.path_utils import repo_root


@dataclass(frozen=True)
class ColMap:
    future_id_overlay: str
    delta_c_sys: str
    winner: Optional[str]
    p_elec: Optional[str]
    p_biomass: Optional[str]

    future_id_futures: str
    u_inc: str
    u_head: str
    u_voll: Optional[str]
    u_upgrade_capex: Optional[str]
    u_consents: Optional[str]

    future_id_compare: str
    total_cost_eb: str
    total_cost_bb: str


def _first_present(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        key = cand.lower()
        if key in cols_l:
            return cols_l[key]
    return None


def detect_columns(overlay_cols: Sequence[str], futures_cols: Sequence[str], compare_cols: Sequence[str]) -> ColMap:
    fid_overlay = _first_present(overlay_cols, ["future_id", "future", "id"])
    fid_futures = _first_present(futures_cols, ["future_id", "future", "id"])
    fid_compare = _first_present(compare_cols, ["future_id", "future", "id"])
    if not fid_overlay or not fid_futures or not fid_compare:
        raise ValueError("Could not detect future_id in one of the inputs.")

    delta_c_sys = _first_present(
        overlay_cols,
        [
            "Delta_C_sys",
            "delta_C_sys",
            "delta_system_cost_future_nzd",
            "delta_system_cost_nzd",
        ],
    )
    if delta_c_sys is None:
        # fall back to explicit EB/BB system cost
        eb = _first_present(overlay_cols, ["EB_system_cost_future_nzd", "EB_system_cost_nzd"])
        bb = _first_present(overlay_cols, ["BB_system_cost_future_nzd", "BB_system_cost_nzd"])
        if eb is None or bb is None:
            raise ValueError(
                "Overlay is missing Delta_C_sys and cannot derive from EB/BB system costs.\n"
                f"Overlay columns: {list(overlay_cols)}"
            )
        delta_c_sys = "__DERIVE_FROM_EB_BB_SYSTEM_COST__"

    winner = _first_present(
        overlay_cols,
        ["winner_system_cost_future", "winner", "winner_label"],
    )

    p_elec = _first_present(overlay_cols, ["P_elec_mult", "p_elec_mult"]) or _first_present(futures_cols, ["P_elec_mult", "p_elec_mult"])
    p_bio = _first_present(overlay_cols, ["P_biomass_mult", "p_biomass_mult", "P_bio_mult"]) or _first_present(futures_cols, ["P_biomass_mult", "p_biomass_mult", "P_bio_mult"])

    u_inc = _first_present(futures_cols, ["U_inc_mult", "u_inc_mult"])
    u_head = _first_present(futures_cols, ["U_headroom_mult", "u_headroom_mult"])
    if u_inc is None or u_head is None:
        raise ValueError(
            "Missing required grid multipliers in futures.csv.\n"
            f"  U_inc_mult: {u_inc}\n"
            f"  U_headroom_mult: {u_head}"
        )

    u_voll = _first_present(futures_cols, ["U_voll", "u_voll"])
    u_upgrade_capex = _first_present(futures_cols, ["U_upgrade_capex_mult", "u_upgrade_capex_mult"])
    u_consents = _first_present(futures_cols, ["U_consents_uplift", "u_consents_uplift"])

    total_cost_eb = _first_present(compare_cols, ["total_cost_nzd_EB", "total_cost_EB", "total_cost_nzd_eb"])
    total_cost_bb = _first_present(compare_cols, ["total_cost_nzd_BB", "total_cost_BB", "total_cost_nzd_bb"])
    if total_cost_eb is None or total_cost_bb is None:
        raise ValueError(
            "Missing required total_cost columns in rdm_compare.\n"
            f"  EB: {total_cost_eb}\n"
            f"  BB: {total_cost_bb}"
        )

    return ColMap(
        future_id_overlay=fid_overlay,
        delta_c_sys=delta_c_sys,
        winner=winner,
        p_elec=p_elec,
        p_biomass=p_bio,
        future_id_futures=fid_futures,
        u_inc=u_inc,
        u_head=u_head,
        u_voll=u_voll,
        u_upgrade_capex=u_upgrade_capex,
        u_consents=u_consents,
        future_id_compare=fid_compare,
        total_cost_eb=total_cost_eb,
        total_cost_bb=total_cost_bb,
    )


def compute_winner_from_delta(delta_eb_minus_bb: pd.Series) -> pd.Series:
    # EB wins if EB cost < BB cost -> (EB - BB) < 0
    return np.where(delta_eb_minus_bb < 0, "EB", "BB")


def _safe_outpath(outdir: Path, basename: str, ext: str = "png") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{basename}.{ext}"
    if not path.exists():
        return path
    i = 2
    while True:
        cand = outdir / f"{basename}_v{i}.{ext}"
        if not cand.exists():
            return cand
        i += 1


def _winner_palette() -> Dict[str, str]:
    return {"EB": "tab:blue", "BB": "tab:orange"}


def option1_boundary_strip(df: pd.DataFrame, outdir: Path) -> Path:
    pal = _winner_palette()

    x = df["U_inc_mult"].values
    y = df["U_headroom_mult"].values
    winner = df["winner_sys"].values
    grid_penalises_eb = (df["Delta_C_grid_auto"] > 0).values  # EB grid cost higher

    # emphasize EB-cheaper futures
    is_eb_site = (winner == "EB")
    sizes = np.where(is_eb_site, 85, 55)
    edgew = np.where(is_eb_site, 1.4, 0.6)

    fig, ax = plt.subplots(figsize=(7, 5))
    for shape, label in [(False, "ΔC_grid_auto ≤ 0 (grid helps/neutral for EB)"), (True, "ΔC_grid_auto > 0 (grid penalises EB)")]:
        mask = (grid_penalises_eb == shape)
        if mask.sum() == 0:
            continue
        markers = np.where((winner[mask] == "EB"), "o", "s")
        # plot per winner to keep 2 colors only
        for w in ["EB", "BB"]:
            m2 = mask & (winner == w)
            if m2.sum() == 0:
                continue
            mk = "^" if shape else "o"
            ax.scatter(
                x[m2],
                y[m2],
                c=pal[w],
                s=sizes[m2],
                marker=mk,
                edgecolor="black",
                linewidth=edgew[m2],
                alpha=0.85,
            )

    ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Incremental-load multiplier, $U_{inc}$ (×)")
    ax.set_ylabel("Headroom multiplier, $U_{head}$ (×)")
    ax.set_title("Grid multipliers: boundary strip view (winner + grid penalty)")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="w", label="EB cheaper (site overlay)", markerfacecolor=pal["EB"], markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="BB cheaper (site overlay)", markerfacecolor=pal["BB"], markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="ΔC_grid_auto ≤ 0", markerfacecolor="0.8", markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="^", color="w", label="ΔC_grid_auto > 0", markerfacecolor="0.8", markeredgecolor="black", markersize=8),
    ]
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    outpath = _safe_outpath(outdir, "fig8_grid_secondary_boundary_strip")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def option2_conditional_distributions(df: pd.DataFrame, outdir: Path) -> Path:
    pal = _winner_palette()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    groups = ["EB", "BB"]
    data_inc = [df.loc[df["winner_sys"] == g, "U_inc_mult"].values for g in groups]
    data_head = [df.loc[df["winner_sys"] == g, "U_headroom_mult"].values for g in groups]

    # box + jitter overlay (no seaborn)
    bp1 = ax1.boxplot(data_inc, labels=[f"{g} (n={len(d)})" for g, d in zip(groups, data_inc)], patch_artist=True)
    bp2 = ax2.boxplot(data_head, labels=[f"{g} (n={len(d)})" for g, d in zip(groups, data_head)], patch_artist=True)

    for bp, g in zip(bp1["boxes"], groups):
        bp.set_facecolor(pal[g])
        bp.set_alpha(0.35)
    for bp, g in zip(bp2["boxes"], groups):
        bp.set_facecolor(pal[g])
        bp.set_alpha(0.35)

    rng = np.random.default_rng(42)
    for i, g in enumerate(groups, start=1):
        vals = df.loc[df["winner_sys"] == g, "U_inc_mult"].values
        xj = i + rng.normal(0, 0.05, size=len(vals))
        ax1.scatter(xj, vals, s=18, color=pal[g], alpha=0.5, edgecolor="none")

        vals2 = df.loc[df["winner_sys"] == g, "U_headroom_mult"].values
        xj2 = i + rng.normal(0, 0.05, size=len(vals2))
        ax2.scatter(xj2, vals2, s=18, color=pal[g], alpha=0.5, edgecolor="none")

    ax1.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)

    ax1.set_title("$U_{inc}$ by site winner group")
    ax2.set_title("$U_{head}$ by site winner group")
    ax1.set_ylabel("Multiplier (×)")
    ax1.grid(True, alpha=0.25, axis="y")
    ax2.grid(True, alpha=0.25, axis="y")
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    fig.suptitle("Grid-side uncertainty: conditional distributions (secondary signal)", y=1.02, fontsize=12)
    fig.tight_layout()

    outpath = _safe_outpath(outdir, "fig8_grid_secondary_conditional_distributions")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def option3_delta_effect_scatter(df: pd.DataFrame, outdir: Path) -> List[Path]:
    pal = _winner_palette()
    written: List[Path] = []

    def _one(xcol: str, xlabel: str, basename: str) -> Path:
        x = df[xcol].values
        y = df["Delta_C_grid_auto"].values
        winner = df["winner_sys"].values

        fig, ax = plt.subplots(figsize=(7, 5))
        for w in ["EB", "BB"]:
            mask = (winner == w)
            ax.scatter(
                x[mask],
                y[mask],
                c=pal[w],
                s=55,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.4,
                label=f"{w} cheaper (site)",
            )
        ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0, alpha=0.6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("ΔC_grid_auto = J_grid,AUTO(EB) − J_grid,AUTO(BB) (NZD)")
        ax.set_title("Grid penalty vs grid multiplier (winner-colored)")
        ax.legend(loc="best", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        outpath = _safe_outpath(outdir, basename)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return outpath

    written.append(_one("U_inc_mult", "Incremental-load multiplier, $U_{inc}$ (×)", "fig8_grid_secondary_delta_vs_uinc"))
    written.append(_one("U_headroom_mult", "Headroom multiplier, $U_{head}$ (×)", "fig8_grid_secondary_delta_vs_uhead"))
    return written


def option4_survival_heatmap(df: pd.DataFrame, outdir: Path) -> Path:
    # survival among EB-cheaper on site: stays EB-cheaper after adding grid AUTO delta
    eb_site = df[df["winner_sys"] == "EB"].copy()
    if len(eb_site) < 3:
        raise ValueError("Not enough EB-cheaper futures for survival heatmap.")

    # define total delta = site delta + grid delta (both EB - BB)
    eb_site["Delta_C_total"] = eb_site["Delta_C_sys"] + eb_site["Delta_C_grid_auto"]
    eb_site["survives"] = (eb_site["Delta_C_total"] < 0).astype(int)

    # bin in U_inc and U_head
    n_bins = 6
    x = eb_site["U_inc_mult"].values
    y = eb_site["U_headroom_mult"].values

    x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    y_edges = np.linspace(y.min(), y.max(), n_bins + 1)

    # compute survival rate per bin
    rates = np.full((n_bins, n_bins), np.nan)
    counts = np.zeros((n_bins, n_bins), dtype=int)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (x >= x_edges[i]) & (x < x_edges[i + 1]) & (y >= y_edges[j]) & (y < y_edges[j + 1])
            if mask.sum() > 0:
                rates[j, i] = eb_site.loc[mask, "survives"].mean()
                counts[j, i] = int(mask.sum())

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(rates, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Survival rate (EB remains favourable after grid AUTO)")

    ax.set_xlabel("$U_{inc}$ bin (low → high)")
    ax.set_ylabel("$U_{head}$ bin (low → high)")
    ax.set_title("Boundary effect: EB survival rate among EB-cheaper site futures")

    # annotate with n and %
    for j in range(n_bins):
        for i in range(n_bins):
            if np.isnan(rates[j, i]):
                continue
            ax.text(i, j, f"{rates[j,i]*100:.0f}%\n(n={counts[j,i]})", ha="center", va="center", color="white" if rates[j,i] > 0.5 else "black", fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    outpath = _safe_outpath(outdir, "fig8_grid_secondary_survival_heatmap")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot grid secondary effects for flip futures (thesis-ready)")
    parser.add_argument("--bundle", type=str, default="poc_20260105_release02")
    parser.add_argument("--output-root", type=str, default="Output")
    args = parser.parse_args()

    root = repo_root()
    bundle_dir = root / args.output_root / "runs" / args.bundle
    overlay_path = bundle_dir / "rdm" / "site_decision_robustness_2035_EB_vs_2035_BB.csv"
    futures_path = bundle_dir / "rdm" / "futures.csv"
    compare_path = bundle_dir / "rdm" / "rdm_compare_2035_EB_vs_BB.csv"
    outdir = bundle_dir / "thesis_pack" / "figures"

    print("[INPUTS]")
    print(f"  overlay: {overlay_path}")
    print(f"  futures: {futures_path}")
    print(f"  rdm_compare: {compare_path}")

    overlay = pd.read_csv(overlay_path)
    futures = pd.read_csv(futures_path)
    compare = pd.read_csv(compare_path)

    colmap = detect_columns(overlay.columns, futures.columns, compare.columns)

    # prepare overlay deltas / winner
    df_overlay = overlay[[colmap.future_id_overlay]].copy()
    df_overlay["future_id"] = df_overlay[colmap.future_id_overlay]

    if colmap.delta_c_sys == "__DERIVE_FROM_EB_BB_SYSTEM_COST__":
        eb = _first_present(overlay.columns, ["EB_system_cost_future_nzd", "EB_system_cost_nzd"])
        bb = _first_present(overlay.columns, ["BB_system_cost_future_nzd", "BB_system_cost_nzd"])
        df_overlay["Delta_C_sys"] = pd.to_numeric(overlay[eb], errors="coerce") - pd.to_numeric(overlay[bb], errors="coerce")
        delta_src = f"derived: {eb} - {bb}"
    else:
        df_overlay["Delta_C_sys"] = pd.to_numeric(overlay[colmap.delta_c_sys], errors="coerce")
        delta_src = colmap.delta_c_sys

    if colmap.winner and colmap.winner in overlay.columns:
        w_raw = overlay[colmap.winner].astype(str).str.upper().str.strip()
        winner_series = pd.Series(pd.NA, index=overlay.index, dtype="object")
        winner_series[w_raw.str.contains("EB", na=False)] = "EB"
        winner_series[w_raw.str.contains("BB", na=False)] = "BB"
        df_overlay["winner_sys"] = winner_series
        if df_overlay["winner_sys"].isna().any():
            df_overlay["winner_sys"] = compute_winner_from_delta(df_overlay["Delta_C_sys"])
    else:
        df_overlay["winner_sys"] = compute_winner_from_delta(df_overlay["Delta_C_sys"])

    # futures
    df_fut = futures[[colmap.future_id_futures, colmap.u_inc, colmap.u_head]].copy()
    df_fut = df_fut.rename(columns={colmap.future_id_futures: "future_id", colmap.u_inc: "U_inc_mult", colmap.u_head: "U_headroom_mult"})
    for opt_col, out_col in [(colmap.u_voll, "U_voll"), (colmap.u_upgrade_capex, "U_upgrade_capex_mult"), (colmap.u_consents, "U_consents_uplift")]:
        if opt_col and opt_col in futures.columns:
            df_fut[out_col] = pd.to_numeric(futures[opt_col], errors="coerce")
    if colmap.p_elec and colmap.p_elec in futures.columns:
        df_fut["P_elec_mult"] = pd.to_numeric(futures[colmap.p_elec], errors="coerce")
    if colmap.p_biomass and colmap.p_biomass in futures.columns:
        df_fut["P_biomass_mult"] = pd.to_numeric(futures[colmap.p_biomass], errors="coerce")

    # compare -> Delta_C_grid_auto (EB - BB)
    df_cmp = compare[[colmap.future_id_compare, colmap.total_cost_eb, colmap.total_cost_bb]].copy()
    df_cmp = df_cmp.rename(columns={colmap.future_id_compare: "future_id"})
    df_cmp["Delta_C_grid_auto"] = pd.to_numeric(df_cmp[colmap.total_cost_eb], errors="coerce") - pd.to_numeric(df_cmp[colmap.total_cost_bb], errors="coerce")

    # join
    df = df_overlay.merge(df_fut, on="future_id", how="inner").merge(df_cmp[["future_id", "Delta_C_grid_auto"]], on="future_id", how="inner")
    df = df.dropna(subset=["Delta_C_sys", "winner_sys", "U_inc_mult", "U_headroom_mult", "Delta_C_grid_auto"]).copy()

    n = len(df)
    n_eb = int((df["winner_sys"] == "EB").sum())

    print("\n[COLUMN MAPPING]")
    print(f"  overlay.future_id: {colmap.future_id_overlay}")
    print(f"  overlay.Delta_C_sys (EB - BB): {delta_src} -> Delta_C_sys")
    print(f"  overlay.winner: {colmap.winner or '(computed from Delta_C_sys)'} -> winner_sys (EB if Delta_C_sys < 0)")
    print(f"  futures.U_inc_mult: {colmap.u_inc} -> U_inc_mult")
    print(f"  futures.U_headroom_mult: {colmap.u_head} -> U_headroom_mult")
    print(f"  compare.total_cost_EB: {colmap.total_cost_eb}")
    print(f"  compare.total_cost_BB: {colmap.total_cost_bb}")
    print("  Delta_C_grid_auto = total_cost_EB - total_cost_BB")

    print("\n[COUNTS]")
    print(f"  futures joined: {n}")
    print(f"  EB-cheaper (site overlay): {n_eb}")
    print(f"  BB-cheaper (site overlay): {n - n_eb}")

    # correlations
    corr_uinc = float(df["U_inc_mult"].corr(df["Delta_C_grid_auto"]))
    corr_uhead = float(df["U_headroom_mult"].corr(df["Delta_C_grid_auto"]))
    print("\n[CORRELATIONS]")
    print(f"  corr(U_inc_mult, Delta_C_grid_auto): {corr_uinc:.3f}")
    print(f"  corr(U_headroom_mult, Delta_C_grid_auto): {corr_uhead:.3f}")

    print("\n[WRITING FIGURES]")
    written: List[Path] = []
    written.append(option1_boundary_strip(df, outdir))
    written.append(option2_conditional_distributions(df, outdir))
    written.extend(option3_delta_effect_scatter(df, outdir))
    try:
        written.append(option4_survival_heatmap(df, outdir))
    except Exception as e:
        print(f"  [INFO] Option 4 skipped: {e}")

    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()

