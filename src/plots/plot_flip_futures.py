"""
Flip futures plotting (thesis-ready).

Goal: Make the "flip region" visually obvious: EB wins only in a small
high-biomass / low-electricity corner, with grid conditions as a secondary
boundary effect.

Read-only: consumes existing outputs only.
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
import matplotlib.tri as mtri

from src.path_utils import repo_root


@dataclass(frozen=True)
class ColMap:
    future_id: str
    p_elec: str
    p_biomass: str
    delta_c_sys: str
    winner: Optional[str]
    u_headroom: Optional[str]
    u_inc: Optional[str]
    u_voll: Optional[str]
    p_ets: Optional[str]
    d_heat: Optional[str]
    u_upgrade_capex: Optional[str]
    u_consents: Optional[str]


def _first_present(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        key = cand.lower()
        if key in cols_l:
            return cols_l[key]
    return None


def detect_columns(overlay_cols: Sequence[str], futures_cols: Sequence[str]) -> ColMap:
    # join key
    future_id = _first_present(overlay_cols, ["future_id", "future", "id"])
    if future_id is None:
        raise ValueError(f"Could not find future_id column in overlay. Available: {list(overlay_cols)}")

    # primary axes drivers
    p_elec = _first_present(overlay_cols, ["P_elec_mult", "p_elec_mult", "P_electricity_mult"])
    p_biomass = _first_present(overlay_cols, ["P_biomass_mult", "p_biomass_mult", "P_bio_mult"])

    # fall back to futures.csv if missing in overlay
    if p_elec is None:
        p_elec = _first_present(futures_cols, ["P_elec_mult", "p_elec_mult", "P_electricity_mult"])
    if p_biomass is None:
        p_biomass = _first_present(futures_cols, ["P_biomass_mult", "p_biomass_mult", "P_bio_mult"])

    if p_elec is None or p_biomass is None:
        raise ValueError(
            "Could not detect required driver columns.\n"
            f"  P_elec_mult found: {p_elec}\n"
            f"  P_biomass_mult found: {p_biomass}\n"
            f"  overlay cols sample: {list(overlay_cols)[:30]}\n"
            f"  futures cols sample: {list(futures_cols)[:30]}"
        )

    # delta system cost (EB - BB)
    delta_c_sys = _first_present(
        overlay_cols,
        [
            "Delta_C_sys",
            "delta_C_sys",
            "delta_system_cost_future_nzd",
            "delta_system_cost_nzd",
            "delta_total_cost_nzd",
        ],
    )
    if delta_c_sys is None:
        # derive from explicit EB/BB system cost if present
        eb = _first_present(overlay_cols, ["EB_system_cost_future_nzd", "EB_system_cost_nzd"])
        bb = _first_present(overlay_cols, ["BB_system_cost_future_nzd", "BB_system_cost_nzd"])
        if eb is None or bb is None:
            raise ValueError(
                "Could not detect Delta_C_sys (EB - BB) or the EB/BB system cost columns to derive it.\n"
                f"  Delta_C_sys candidates missing. EB col: {eb}, BB col: {bb}"
            )
        # we'll create a derived column name later; return the EB col key here and treat specially
        delta_c_sys = "__DERIVE_FROM_EB_BB_SYSTEM_COST__"

    winner = _first_present(
        overlay_cols,
        [
            "winner_system_cost_future",
            "winner",
            "winner_label",
            "winner_system_cost",
        ],
    )

    # secondary grid drivers (prefer overlay, fallback futures)
    def opt(cands: Sequence[str]) -> Optional[str]:
        return _first_present(overlay_cols, cands) or _first_present(futures_cols, cands)

    return ColMap(
        future_id=future_id,
        p_elec=p_elec,
        p_biomass=p_biomass,
        delta_c_sys=delta_c_sys,
        winner=winner,
        u_headroom=opt(["U_headroom_mult", "u_headroom_mult"]),
        u_inc=opt(["U_inc_mult", "u_inc_mult"]),
        u_voll=opt(["U_voll", "u_voll"]),
        p_ets=opt(["ETS_mult", "P_ETS_mult", "ets_mult"]),
        d_heat=opt(["D_heat_mult", "d_heat_mult"]),
        u_upgrade_capex=opt(["U_upgrade_capex_mult", "u_upgrade_capex_mult"]),
        u_consents=opt(["U_consents_uplift", "u_consents_uplift"]),
    )


def compute_winner_from_delta(delta_eb_minus_bb: pd.Series) -> pd.Series:
    # EB wins if EB cost < BB cost -> (EB - BB) < 0
    return np.where(delta_eb_minus_bb < 0, "EB", "BB")


def bin_abs_delta(delta: pd.Series) -> Tuple[pd.Categorical, List[Tuple[float, float]]]:
    absd = delta.abs().astype(float)
    # robust quantile bins; ensure strictly increasing edges
    qs = [0.0, 0.5, 0.8, 0.95, 1.0]
    edges = [absd.quantile(q) for q in qs]
    edges = [float(e) for e in edges]
    edges[0] = 0.0
    edges[-1] = float(absd.max()) if float(absd.max()) > 0 else 1.0
    # de-duplicate edges
    cleaned = [edges[0]]
    for e in edges[1:]:
        if e > cleaned[-1]:
            cleaned.append(e)
    if len(cleaned) < 3:
        cleaned = [0.0, float(absd.max()) * 0.5 + 1e-9, float(absd.max()) + 1e-6]

    bins = pd.cut(absd, bins=cleaned, include_lowest=True)
    ranges = [(float(b.left), float(b.right)) for b in bins.cat.categories]
    return bins, ranges


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotonic chain convex hull. Returns hull points in order (closed not included)."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1])
    return hull


def _safe_outpath(outdir: Path, basename: str, ext: str) -> Path:
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
    # Colorblind-friendly two-color choice (matplotlib tab10)
    return {"EB": "tab:blue", "BB": "tab:orange"}


def figure_option_a(df: pd.DataFrame, outdir: Path, formats: List[str]) -> List[Path]:
    # scatter with winner color + size bins + EB hull
    x = df["P_elec_mult_used"].values
    y = df["P_biomass_mult_used"].values
    delta = df["Delta_C_sys_used"].values
    winner = df["winner_used"].values

    bins, ranges = bin_abs_delta(pd.Series(delta))
    size_map = [25, 55, 90, 130][: len(bins.cat.categories)]
    sizes = bins.cat.codes.map(lambda i: size_map[int(i)]).values

    pal = _winner_palette()
    colors = [pal[w] for w in winner]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, c=colors, s=sizes, alpha=0.8, edgecolor="black", linewidth=0.4)

    ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)

    # hull around EB winners if possible
    eb_mask = (winner == "EB")
    eb_pts = np.column_stack([x[eb_mask], y[eb_mask]])
    if eb_pts.shape[0] >= 3:
        hull = _convex_hull(eb_pts)
        if hull.shape[0] >= 3:
            ax.plot(
                np.r_[hull[:, 0], hull[0, 0]],
                np.r_[hull[:, 1], hull[0, 1]],
                color=pal["EB"],
                linewidth=2.0,
                alpha=0.9,
            )
    elif eb_pts.shape[0] >= 1:
        # fallback: dashed box around EB points
        xmin, xmax = float(eb_pts[:, 0].min()), float(eb_pts[:, 0].max())
        ymin, ymax = float(eb_pts[:, 1].min()), float(eb_pts[:, 1].max())
        pad_x = max(0.01, 0.05 * (xmax - xmin + 1e-9))
        pad_y = max(0.01, 0.05 * (ymax - ymin + 1e-9))
        ax.add_patch(
            plt.Rectangle(
                (xmin - pad_x, ymin - pad_y),
                (xmax - xmin) + 2 * pad_x,
                (ymax - ymin) + 2 * pad_y,
                fill=False,
                linestyle="--",
                linewidth=2.0,
                edgecolor=pal["EB"],
                alpha=0.9,
            )
        )

    ax.set_xlabel("Electricity price multiplier, $P_{elec}$ (×)")
    ax.set_ylabel("Biomass price multiplier, $P_{bio}$ (×)")
    ax.set_title("Flip region in price-multiplier space (winner by future)")

    # legend: winner
    from matplotlib.lines import Line2D

    winner_handles = [
        Line2D([0], [0], marker="o", color="w", label="EB wins", markerfacecolor=pal["EB"], markersize=8, markeredgecolor="black", markeredgewidth=0.4),
        Line2D([0], [0], marker="o", color="w", label="BB wins", markerfacecolor=pal["BB"], markersize=8, markeredgecolor="black", markeredgewidth=0.4),
    ]
    # legend: size bins
    size_handles = []
    for i, (lo, hi) in enumerate(ranges):
        lbl = f"|ΔC_sys| ∈ [{lo/1e6:.1f}, {hi/1e6:.1f}] M NZD" if hi >= 1e6 else f"|ΔC_sys| ∈ [{lo:.0f}, {hi:.0f}] NZD"
        size_handles.append(Line2D([0], [0], marker="o", color="w", label=lbl, markerfacecolor="0.6", markersize=np.sqrt(size_map[i]), markeredgecolor="black", markeredgewidth=0.4))

    leg1 = ax.legend(handles=winner_handles, loc="upper right", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=size_handles, loc="lower left", frameon=True, fontsize=8)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    written: List[Path] = []
    base = "fig8_flip_scatter_winner_sizebins"
    for ext in formats:
        outpath = _safe_outpath(outdir, base, ext)
        if ext.lower() == "png":
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(outpath, bbox_inches="tight")
        written.append(outpath)
    plt.close(fig)
    return written


def figure_option_b(df: pd.DataFrame, outdir: Path, formats: List[str]) -> List[Path]:
    # Same axes + winner, but grid metric (U_inc_mult) as edge linewidth bins
    if "U_inc_mult_used" not in df.columns or df["U_inc_mult_used"].isna().all():
        raise ValueError("U_inc_mult not available for Option B.")

    x = df["P_elec_mult_used"].values
    y = df["P_biomass_mult_used"].values
    winner = df["winner_used"].values
    uinc = df["U_inc_mult_used"].astype(float).values

    pal = _winner_palette()
    colors = [pal[w] for w in winner]

    # bins for U_inc
    q = np.quantile(uinc, [0.0, 0.33, 0.67, 1.0])
    q = np.unique(q)
    if len(q) < 3:
        q = np.array([uinc.min(), (uinc.min() + uinc.max()) / 2.0, uinc.max()])
    bins = pd.cut(uinc, bins=q, include_lowest=True)
    lw_map = [0.4, 1.2, 2.2][: len(bins.categories)]
    lws = pd.Series(bins.codes).map(lambda i: lw_map[int(i)]).values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, c=colors, s=65, alpha=0.85, edgecolor="black", linewidth=lws)

    ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Electricity price multiplier, $P_{elec}$ (×)")
    ax.set_ylabel("Biomass price multiplier, $P_{bio}$ (×)")
    ax.set_title("Flip region with grid-stress as secondary effect")

    from matplotlib.lines import Line2D

    winner_handles = [
        Line2D([0], [0], marker="o", color="w", label="EB wins", markerfacecolor=pal["EB"], markersize=8, markeredgecolor="black"),
        Line2D([0], [0], marker="o", color="w", label="BB wins", markerfacecolor=pal["BB"], markersize=8, markeredgecolor="black"),
    ]
    inc_handles = []
    for i, cat in enumerate(bins.categories):
        inc_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"U_inc ∈ [{cat.left:.2f}, {cat.right:.2f}]",
                markerfacecolor="0.8",
                markeredgecolor="black",
                markeredgewidth=lw_map[i],
                markersize=8,
            )
        )

    leg1 = ax.legend(handles=winner_handles, loc="upper right", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=inc_handles, loc="lower left", frameon=True, fontsize=8)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    written: List[Path] = []
    base = "fig8_flip_scatter_winner_gridbins"
    for ext in formats:
        outpath = _safe_outpath(outdir, base, ext)
        if ext.lower() == "png":
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(outpath, bbox_inches="tight")
        written.append(outpath)
    plt.close(fig)
    return written


def figure_option_c(df: pd.DataFrame, outdir: Path, formats: List[str]) -> List[Path]:
    # winner-colored scatter + light contour background of Delta_C_sys
    x = df["P_elec_mult_used"].values
    y = df["P_biomass_mult_used"].values
    z = df["Delta_C_sys_used"].astype(float).values
    winner = df["winner_used"].values

    pal = _winner_palette()
    colors = [pal[w] for w in winner]

    fig, ax = plt.subplots(figsize=(7, 5))

    # triangulation contour background (light)
    try:
        tri = mtri.Triangulation(x, y)
        vmax = np.nanmax(np.abs(z))
        vmax = float(vmax) if vmax and vmax > 0 else 1.0
        levels = np.linspace(-vmax, vmax, 12)
        cf = ax.tricontourf(tri, z, levels=levels, cmap="coolwarm", alpha=0.25)
        ax.tricontour(tri, z, levels=[0.0], colors="black", linewidths=1.2, alpha=0.8)
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("ΔC_sys = EB − BB (NZD)  (EB wins if < 0)")
    except Exception:
        # proceed without background if unstable
        pass

    ax.scatter(x, y, c=colors, s=55, alpha=0.85, edgecolor="black", linewidth=0.4)

    ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Electricity price multiplier, $P_{elec}$ (×)")
    ax.set_ylabel("Biomass price multiplier, $P_{bio}$ (×)")
    ax.set_title("Flip diagnostic: winner scatter with ΔC_sys background")

    from matplotlib.lines import Line2D

    winner_handles = [
        Line2D([0], [0], marker="o", color="w", label="EB wins", markerfacecolor=pal["EB"], markersize=8, markeredgecolor="black"),
        Line2D([0], [0], marker="o", color="w", label="BB wins", markerfacecolor=pal["BB"], markersize=8, markeredgecolor="black"),
    ]
    ax.legend(handles=winner_handles, loc="upper right", frameon=True)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    written: List[Path] = []
    base = "fig8_flip_scatter_contour_boundary"
    for ext in formats:
        outpath = _safe_outpath(outdir, base, ext)
        if ext.lower() == "png":
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(outpath, bbox_inches="tight")
        written.append(outpath)
    plt.close(fig)
    return written


def figure_option_d(df: pd.DataFrame, outdir: Path, formats: List[str]) -> List[Path]:
    # two-panel: left option A-ish, right grid metric as color (U_inc_mult) with winner marker shape
    if "U_inc_mult_used" not in df.columns or df["U_inc_mult_used"].isna().all():
        raise ValueError("U_inc_mult not available for Option D.")

    x = df["P_elec_mult_used"].values
    y = df["P_biomass_mult_used"].values
    z = df["U_inc_mult_used"].astype(float).values
    winner = df["winner_used"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5), sharex=True, sharey=True)
    pal = _winner_palette()

    # Left: winner color
    ax1.scatter(x, y, c=[pal[w] for w in winner], s=55, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax1.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
    ax1.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
    ax1.set_title("Winner (color)")
    ax1.set_xlabel("$P_{elec}$ (×)")
    ax1.set_ylabel("$P_{bio}$ (×)")
    ax1.grid(True, alpha=0.25)

    # Right: U_inc as color, winner as marker
    markers = {"EB": "o", "BB": "s"}
    sc = None
    for w in ["EB", "BB"]:
        m = markers[w]
        mask = (winner == w)
        sc = ax2.scatter(
            x[mask],
            y[mask],
            c=z[mask],
            cmap="viridis",
            s=55,
            alpha=0.85,
            marker=m,
            edgecolor="black",
            linewidth=0.4,
        )
    ax2.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.set_title("Grid driver $U_{inc}$ (color) + winner (marker)")
    ax2.set_xlabel("$P_{elec}$ (×)")
    ax2.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("U_inc multiplier (×)")

    fig.suptitle("Flip region with secondary grid-driver view", y=1.02, fontsize=12)
    fig.tight_layout()

    written: List[Path] = []
    base = "fig8_flip_scatter_twopanel_grid"
    for ext in formats:
        outpath = _safe_outpath(outdir, base, ext)
        if ext.lower() == "png":
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(outpath, bbox_inches="tight")
        written.append(outpath)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot flip futures and key drivers (thesis-ready)")
    parser.add_argument("--bundle", type=str, default="poc_20260105_release02")
    parser.add_argument("--output-root", type=str, default="Output")
    parser.add_argument("--formats", type=str, default="png", help="Comma-separated: png,pdf")
    args = parser.parse_args()

    root = repo_root()
    bundle_dir = root / args.output_root / "runs" / args.bundle
    overlay_path = bundle_dir / "rdm" / "site_decision_robustness_2035_EB_vs_2035_BB.csv"
    futures_path = bundle_dir / "rdm" / "futures.csv"
    outdir = bundle_dir / "thesis_pack" / "figures"

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    formats = [f for f in formats if f in ("png", "pdf")]
    if not formats:
        formats = ["png"]

    print("[INPUTS]")
    print(f"  overlay: {overlay_path}")
    print(f"  futures: {futures_path}")

    overlay = pd.read_csv(overlay_path)
    futures = pd.read_csv(futures_path)

    colmap = detect_columns(overlay.columns, futures.columns)

    # join
    df = overlay.merge(futures, on=colmap.future_id, how="left", suffixes=("", "__f"))

    # choose columns (overlay preferred)
    df["P_elec_mult_used"] = pd.to_numeric(df[colmap.p_elec], errors="coerce")
    df["P_biomass_mult_used"] = pd.to_numeric(df[colmap.p_biomass], errors="coerce")

    # delta EB - BB
    if colmap.delta_c_sys == "__DERIVE_FROM_EB_BB_SYSTEM_COST__":
        eb = _first_present(overlay.columns, ["EB_system_cost_future_nzd", "EB_system_cost_nzd"])
        bb = _first_present(overlay.columns, ["BB_system_cost_future_nzd", "BB_system_cost_nzd"])
        df["Delta_C_sys_used"] = pd.to_numeric(df[eb], errors="coerce") - pd.to_numeric(df[bb], errors="coerce")
        delta_src = f"derived: {eb} - {bb}"
    else:
        df["Delta_C_sys_used"] = pd.to_numeric(df[colmap.delta_c_sys], errors="coerce")
        delta_src = colmap.delta_c_sys

    # winner
    if colmap.winner and colmap.winner in df.columns:
        w_raw = df[colmap.winner].astype(str).str.upper().str.strip()
        # accept values like 'BB', 'EB'
        df["winner_used"] = np.where(w_raw.str.contains("EB"), "EB", np.where(w_raw.str.contains("BB"), "BB", None))
        if df["winner_used"].isna().any():
            df["winner_used"] = compute_winner_from_delta(df["Delta_C_sys_used"])
    else:
        df["winner_used"] = compute_winner_from_delta(df["Delta_C_sys_used"])

    # optional grid columns
    if colmap.u_inc and colmap.u_inc in df.columns:
        df["U_inc_mult_used"] = pd.to_numeric(df[colmap.u_inc], errors="coerce")
    if colmap.u_headroom and colmap.u_headroom in df.columns:
        df["U_headroom_mult_used"] = pd.to_numeric(df[colmap.u_headroom], errors="coerce")

    # drop bad rows
    df = df.dropna(subset=["P_elec_mult_used", "P_biomass_mult_used", "Delta_C_sys_used", "winner_used"]).copy()

    n = len(df)
    n_eb = int((df["winner_used"] == "EB").sum())

    print("\n[COLUMN MAPPING]")
    print(f"  future_id: {colmap.future_id}")
    print(f"  P_elec_mult: {colmap.p_elec} -> P_elec_mult_used")
    print(f"  P_biomass_mult: {colmap.p_biomass} -> P_biomass_mult_used")
    print(f"  Delta_C_sys (EB - BB): {delta_src} -> Delta_C_sys_used")
    print(f"  winner: {colmap.winner or '(computed from Delta_C_sys)'} -> winner_used (EB if Delta_C_sys < 0)")
    if "U_inc_mult_used" in df.columns:
        print(f"  U_inc_mult: {colmap.u_inc} -> U_inc_mult_used")
    if "U_headroom_mult_used" in df.columns:
        print(f"  U_headroom_mult: {colmap.u_headroom} -> U_headroom_mult_used")

    print("\n[COUNTS]")
    print(f"  futures plotted: {n}")
    print(f"  EB wins: {n_eb}")
    print(f"  BB wins: {n - n_eb}")

    # Plot
    written: List[Path] = []
    print("\n[WRITING FIGURES]")
    written += figure_option_a(df, outdir, formats)
    try:
        written += figure_option_b(df, outdir, formats)
    except Exception as e:
        print(f"  [WARN] Option B skipped: {e}")
    written += figure_option_c(df, outdir, formats)
    try:
        written += figure_option_d(df, outdir, formats)
    except Exception as e:
        print(f"  [INFO] Option D skipped: {e}")

    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()

