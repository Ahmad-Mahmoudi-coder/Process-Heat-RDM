"""
Grid boundary flip-region plotting (thesis-ready).

Objective:
- Demonstrate "no grid rescue" (Site prefers BB but Total prefers EB) == 0 futures.
- Identify "boundary reversals" (Site prefers EB but Total prefers BB), expected small (~4),
  near the (P_elec, P_bio) boundary and associated with high U_inc and/or low U_head.

Inputs (read-only):
- Output/runs/<bundle>/rdm/site_decision_robustness_2035_EB_vs_2035_BB.csv
- Output/runs/<bundle>/rdm/futures.csv
- Output/runs/<bundle>/rdm/rdm_compare_2035_EB_vs_BB.csv

Outputs:
- Writes new figures into Output/runs/<bundle>/thesis_pack/figures
  (never overwrites; appends _v2, _v3, ...)
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
    fid_overlay: str
    delta_c_sys: str
    winner_site: Optional[str]
    p_elec: str
    p_bio: str

    fid_futures: str
    u_inc: str
    u_head: str

    fid_compare: str
    total_cost_eb: str
    total_cost_bb: str
    upgrade_eb: Optional[str]
    upgrade_bb: Optional[str]


def _first_present(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        key = cand.lower()
        if key in cols_l:
            return cols_l[key]
    return None


def detect_columns(overlay_cols: Sequence[str], futures_cols: Sequence[str], compare_cols: Sequence[str]) -> ColMap:
    fid_overlay = _first_present(overlay_cols, ["future_id"])
    fid_futures = _first_present(futures_cols, ["future_id"])
    fid_compare = _first_present(compare_cols, ["future_id"])
    if not fid_overlay or not fid_futures or not fid_compare:
        raise ValueError("Missing future_id in one or more inputs.")

    delta_c_sys = _first_present(
        overlay_cols,
        ["Delta_C_sys", "delta_system_cost_future_nzd", "delta_system_cost_nzd"],
    )
    if delta_c_sys is None:
        eb = _first_present(overlay_cols, ["EB_system_cost_future_nzd", "EB_system_cost_nzd"])
        bb = _first_present(overlay_cols, ["BB_system_cost_future_nzd", "BB_system_cost_nzd"])
        if eb is None or bb is None:
            raise ValueError("Overlay missing Delta_C_sys and cannot derive from EB/BB system costs.")
        delta_c_sys = "__DERIVE_FROM_EB_BB_SYSTEM_COST__"

    winner_site = _first_present(overlay_cols, ["winner_system_cost_future", "winner"])

    p_elec = _first_present(overlay_cols, ["P_elec_mult", "p_elec_mult"]) or _first_present(futures_cols, ["P_elec_mult", "p_elec_mult"])
    p_bio = _first_present(overlay_cols, ["P_biomass_mult", "p_biomass_mult", "P_bio_mult"]) or _first_present(futures_cols, ["P_biomass_mult", "p_biomass_mult", "P_bio_mult"])
    if p_elec is None or p_bio is None:
        raise ValueError("Missing P_elec_mult and/or P_biomass_mult in overlay/futures.")

    u_inc = _first_present(futures_cols, ["U_inc_mult", "u_inc_mult"])
    u_head = _first_present(futures_cols, ["U_headroom_mult", "u_headroom_mult"])
    if u_inc is None or u_head is None:
        raise ValueError("Missing U_inc_mult and/or U_headroom_mult in futures.csv.")

    total_cost_eb = _first_present(compare_cols, ["total_cost_nzd_EB", "total_cost_EB"])
    total_cost_bb = _first_present(compare_cols, ["total_cost_nzd_BB", "total_cost_BB"])
    if total_cost_eb is None or total_cost_bb is None:
        raise ValueError("Missing total_cost_nzd_EB/BB in rdm_compare.")

    upgrade_eb = _first_present(compare_cols, ["selected_upgrade_name_EB", "upgrade_EB", "auto_upgrade_EB"])
    upgrade_bb = _first_present(compare_cols, ["selected_upgrade_name_BB", "upgrade_BB", "auto_upgrade_BB"])

    return ColMap(
        fid_overlay=fid_overlay,
        delta_c_sys=delta_c_sys,
        winner_site=winner_site,
        p_elec=p_elec,
        p_bio=p_bio,
        fid_futures=fid_futures,
        u_inc=u_inc,
        u_head=u_head,
        fid_compare=fid_compare,
        total_cost_eb=total_cost_eb,
        total_cost_bb=total_cost_bb,
        upgrade_eb=upgrade_eb,
        upgrade_bb=upgrade_bb,
    )


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


def _winner_from_delta(delta: pd.Series) -> pd.Series:
    return np.where(delta < 0, "EB", "BB")


def _palette_categories() -> Dict[str, str]:
    # colorblind friendly-ish (Okabe-Ito subset)
    return {
        "A": "#0072B2",  # blue
        "B": "#E69F00",  # orange
        "C": "#009E73",  # green
        "D": "#D55E00",  # vermillion
    }


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["site_pref"] = np.where(df["Delta_C_sys"] < 0, "EB", "BB")
    df["total_pref"] = np.where(df["Delta_C_total"] < 0, "EB", "BB")

    df["category"] = "B"
    df.loc[(df["site_pref"] == "EB") & (df["total_pref"] == "EB"), "category"] = "A"
    df.loc[(df["site_pref"] == "BB") & (df["total_pref"] == "BB"), "category"] = "B"
    df.loc[(df["site_pref"] == "BB") & (df["total_pref"] == "EB"), "category"] = "C"
    df.loc[(df["site_pref"] == "EB") & (df["total_pref"] == "BB"), "category"] = "D"
    return df


def fig1_quadrant_scatter(df: pd.DataFrame, outdir: Path) -> Path:
    pal = _palette_categories()
    fig, ax = plt.subplots(figsize=(7, 5))

    # plot all except D first
    for cat in ["A", "B", "C"]:
        d = df[df["category"] == cat]
        if len(d) == 0:
            continue
        ax.scatter(
            d["P_elec_mult"],
            d["P_biomass_mult"],
            s=55,
            c=pal[cat],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.4,
            label=cat,
        )

    # category D: boundary reversals with X marker
    d = df[df["category"] == "D"]
    if len(d) > 0:
        ax.scatter(
            d["P_elec_mult"],
            d["P_biomass_mult"],
            s=110,
            c=pal["D"],
            alpha=0.9,
            marker="X",
            edgecolor="black",
            linewidth=0.6,
            label="D",
        )
        for _, row in d.iterrows():
            ax.text(
                float(row["P_elec_mult"]) + 0.01,
                float(row["P_biomass_mult"]) + 0.01,
                str(int(row["future_id"])),
                fontsize=8,
                color="black",
            )

    ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Electricity price multiplier, $P_{elec}$ (×)")
    ax.set_ylabel("Biomass price multiplier, $P_{bio}$ (×)")
    ax.set_title("Grid boundary flips in price-multiplier space")

    legend_labels = {
        "A": "A: Site EB, Total EB (stable EB)",
        "B": "B: Site BB, Total BB (stable BB)",
        "C": "C: Site BB, Total EB (grid rescue)",
        "D": "D: Site EB, Total BB (boundary reversal)",
    }
    from matplotlib.lines import Line2D
    handles = []
    for cat in ["A", "B", "C", "D"]:
        if cat == "D":
            handles.append(Line2D([0], [0], marker="X", color="w", label=legend_labels[cat], markerfacecolor=pal[cat], markeredgecolor="black", markersize=9))
        else:
            handles.append(Line2D([0], [0], marker="o", color="w", label=legend_labels[cat], markerfacecolor=pal[cat], markeredgecolor="black", markersize=7))
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=8)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    outpath = _safe_outpath(outdir, "fig8_31a_grid_boundary_quadrant_scatter")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def fig2_margin_vs_penalty(df: pd.DataFrame, outdir: Path) -> Path:
    # x=Delta_C_sys, y=Delta_C_grid_auto; highlight D; diagonal y=-x
    fig, ax = plt.subplots(figsize=(7, 5))

    total_winner = np.where(df["Delta_C_total"] < 0, "EB", "BB")
    pal = {"EB": "tab:blue", "BB": "tab:orange"}

    for w in ["EB", "BB"]:
        m = (total_winner == w)
        ax.scatter(
            df.loc[m, "Delta_C_sys"],
            df.loc[m, "Delta_C_grid_auto"],
            s=55,
            c=pal[w],
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
            label=f"Total winner: {w}",
        )

    # highlight D
    d = df[df["category"] == "D"]
    if len(d) > 0:
        ax.scatter(
            d["Delta_C_sys"],
            d["Delta_C_grid_auto"],
            s=120,
            c="#D55E00",
            marker="X",
            alpha=0.95,
            edgecolor="black",
            linewidth=0.6,
            label="Boundary reversal (D)",
        )
        for _, row in d.iterrows():
            ax.text(float(row["Delta_C_sys"]), float(row["Delta_C_grid_auto"]), str(int(row["future_id"])), fontsize=8, ha="left", va="bottom")

    ax.axvline(0.0, color="0.3", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0, alpha=0.6)

    # diagonal y = -x where total delta is zero
    xlim = np.array(ax.get_xlim())
    ax.plot(xlim, -xlim, color="0.5", linestyle=":", linewidth=1.2, alpha=0.8, label="ΔC_total = 0 (y = −x)")

    ax.set_xlabel("Site margin: ΔC_sys = C_sys(EB) − C_sys(BB) (NZD)")
    ax.set_ylabel("Grid penalty: ΔC_grid_auto = J_grid,AUTO(EB) − J_grid,AUTO(BB) (NZD)")
    ax.set_title("Grid penalty flips marginal site advantages (near-boundary)")
    ax.legend(loc="best", frameon=True, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    outpath = _safe_outpath(outdir, "fig8_31b_margin_vs_grid_penalty")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def fig3_penalty_vs_multipliers(df: pd.DataFrame, outdir: Path) -> Tuple[Path, Path]:
    dmask = (df["category"] == "D")
    pal = {"other": "0.65", "D": "#D55E00"}

    def _one(xcol: str, xlabel: str, basename: str) -> Path:
        fig, ax = plt.subplots(figsize=(7, 5))
        # others
        ax.scatter(df.loc[~dmask, xcol], df.loc[~dmask, "Delta_C_grid_auto"], s=45, c=pal["other"], alpha=0.55, edgecolor="none", label="Other futures")
        # D
        if dmask.sum() > 0:
            ax.scatter(df.loc[dmask, xcol], df.loc[dmask, "Delta_C_grid_auto"], s=130, c=pal["D"], marker="X", alpha=0.95, edgecolor="black", linewidth=0.6, label="Boundary reversal (D)")
            for _, row in df.loc[dmask].iterrows():
                ax.text(float(row[xcol]), float(row["Delta_C_grid_auto"]), str(int(row["future_id"])), fontsize=8, ha="left", va="bottom")

        ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ΔC_grid_auto (EB − BB) (NZD)")
        ax.set_title("Boundary reversals align with grid adequacy exposure")
        ax.legend(loc="best", frameon=True, fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        outpath = _safe_outpath(outdir, basename)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return outpath

    p1 = _one("U_inc_mult", "Incremental-load multiplier, $U_{inc}$ (×)", "fig8_31c_grid_penalty_vs_Uinc")
    p2 = _one("U_headroom_mult", "Headroom multiplier, $U_{head}$ (×)", "fig8_31d_grid_penalty_vs_Uhead")
    return p1, p2


def fig4_upgrade_switching(df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if "upgrade_EB" not in df.columns or "upgrade_BB" not in df.columns:
        return None
    d = df[df["category"] == "D"].copy()
    if len(d) == 0:
        return None

    # Table-like plot
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.35 * len(d)))
    ax.axis("off")
    ax.set_title("Boundary reversal futures: AUTO upgrade choices (EB vs BB)", fontsize=12, pad=10)

    # build table data
    rows = [["future_id", "AUTO upgrade (EB)", "AUTO upgrade (BB)", "U_inc", "U_head"]]
    for _, r in d.sort_values("future_id").iterrows():
        rows.append([
            str(int(r["future_id"])),
            str(r["upgrade_EB"]),
            str(r["upgrade_BB"]),
            f"{r['U_inc_mult']:.2f}",
            f"{r['U_headroom_mult']:.2f}",
        ])

    table = ax.table(cellText=rows, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    outpath = _safe_outpath(outdir, "fig8_31e_upgrade_switching_boundary_reversals")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot grid boundary flip region (thesis-ready)")
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

    # overlay
    df_overlay = pd.DataFrame({"future_id": overlay[colmap.fid_overlay]})
    if colmap.delta_c_sys == "__DERIVE_FROM_EB_BB_SYSTEM_COST__":
        eb = _first_present(overlay.columns, ["EB_system_cost_future_nzd", "EB_system_cost_nzd"])
        bb = _first_present(overlay.columns, ["BB_system_cost_future_nzd", "BB_system_cost_nzd"])
        df_overlay["Delta_C_sys"] = pd.to_numeric(overlay[eb], errors="coerce") - pd.to_numeric(overlay[bb], errors="coerce")
        delta_src = f"derived: {eb} - {bb}"
    else:
        df_overlay["Delta_C_sys"] = pd.to_numeric(overlay[colmap.delta_c_sys], errors="coerce")
        delta_src = colmap.delta_c_sys

    # Futures multipliers (use futures.csv as canonical)
    df_fut = pd.DataFrame({
        "future_id": futures[colmap.fid_futures],
        "P_elec_mult": pd.to_numeric(futures[colmap.p_elec], errors="coerce"),
        "P_biomass_mult": pd.to_numeric(futures[colmap.p_bio], errors="coerce"),
        "U_inc_mult": pd.to_numeric(futures[colmap.u_inc], errors="coerce"),
        "U_headroom_mult": pd.to_numeric(futures[colmap.u_head], errors="coerce"),
    })

    # compare (AUTO outcomes)
    df_cmp = pd.DataFrame({
        "future_id": compare[colmap.fid_compare],
        "Delta_C_grid_auto": pd.to_numeric(compare[colmap.total_cost_eb], errors="coerce") - pd.to_numeric(compare[colmap.total_cost_bb], errors="coerce"),
    })
    if colmap.upgrade_eb and colmap.upgrade_bb:
        df_cmp["upgrade_EB"] = compare[colmap.upgrade_eb].astype(str)
        df_cmp["upgrade_BB"] = compare[colmap.upgrade_bb].astype(str)

    # join
    df = df_overlay.merge(df_fut, on="future_id", how="inner").merge(df_cmp, on="future_id", how="inner")
    df = df.dropna(subset=["Delta_C_sys", "Delta_C_grid_auto", "P_elec_mult", "P_biomass_mult", "U_inc_mult", "U_headroom_mult"]).copy()
    df["Delta_C_total"] = df["Delta_C_sys"] + df["Delta_C_grid_auto"]

    df = classify(df)

    # counts
    counts = df["category"].value_counts().to_dict()
    c_ids = df.loc[df["category"] == "C", "future_id"].astype(int).tolist()
    d_ids = df.loc[df["category"] == "D", "future_id"].astype(int).tolist()

    print("\n[DEFINITIONS]")
    print(f"  Delta_C_sys = {delta_src}  (EB - BB). EB preferred on site if Delta_C_sys < 0.")
    print(f"  Delta_C_grid_auto = total_cost_nzd_EB - total_cost_nzd_BB  (EB - BB).")
    print(f"  Delta_C_total = Delta_C_sys + Delta_C_grid_auto. EB preferred in total if Delta_C_total < 0.")
    print("  Categories: A(site EB,total EB) B(site BB,total BB) C(site BB,total EB) D(site EB,total BB)")

    print("\n[COUNTS A/B/C/D]")
    for k in ["A", "B", "C", "D"]:
        print(f"  {k}: {int(counts.get(k, 0))}")
    print(f"\n[LIST] C (grid rescue) future_ids: {c_ids}")
    print(f"[LIST] D (boundary reversal) future_ids: {d_ids}")

    if len(d_ids) > 0:
        d = df[df["category"] == "D"]
        print("\n[STATS] Category D ranges")
        for col in ["P_elec_mult", "P_biomass_mult", "U_inc_mult", "U_headroom_mult"]:
            print(f"  {col}: min={d[col].min():.3f}, max={d[col].max():.3f}")

    # figures
    print("\n[WRITING FIGURES]")
    written: List[Path] = []
    written.append(fig1_quadrant_scatter(df, outdir))
    written.append(fig2_margin_vs_penalty(df, outdir))
    p3a, p3b = fig3_penalty_vs_multipliers(df, outdir)
    written.extend([p3a, p3b])
    p4 = fig4_upgrade_switching(df, outdir)
    if p4:
        written.append(p4)

    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()

