import numpy as np
import pandas as pd
import scipy.stats as st
import yaml

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.cm as cm

from pathlib import Path
from typing import Union

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

label_csv = Path(cfg["label_index"])
label_df = pd.read_csv(label_csv)
ID2NAME = dict(zip(label_df["IDX"], label_df["LABEL"]))


def plot_roi_compare(
    vdf_base: pd.DataFrame,
    vdf_12m: pd.DataFrame,
    roi_id: Union[int, str],
    out_dir: str | Path = "stats_png",
    show_plot: bool = False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    col = f"L{roi_id}" if isinstance(roi_id, int) else str(roi_id)
    if col not in vdf_base.columns or col not in vdf_12m.columns:
        raise KeyError(f"Column '{col}' not found in DataFrames.")
    roi_name = ID2NAME[roi_id]

    vol0 = vdf_base[col].dropna()
    vol1 = vdf_12m[col].dropna()
    mean0, mean1 = vol0.mean(), vol1.mean()
    d_mean = mean1 - mean0

    fig, (ax_vio, ax_hist) = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )

    # Violin + Box
    parts = ax_vio.violinplot(
        [vol0, vol1], showmedians=True, widths=0.8, positions=[1, 2]
    )
    for pc, c in zip(parts["bodies"], ["#4F81BD", "#F28E2B"]):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    ax_vio.boxplot([vol0, vol1], widths=0.25, showfliers=False, positions=[1, 2])
    ax_vio.set_xticks([1, 2])
    ax_vio.set_xticklabels(["Baseline", "12 mo"])
    ax_vio.set_ylabel("Volume (mL)")
    ax_vio.set_title("Violin / Box")

    # Histogram + KDE
    bins = np.linspace(min(vol0.min(), vol1.min()), max(vol0.max(), vol1.max()), 30)
    ax_hist.hist(
        vol0, bins=bins, alpha=0.5, density=True, color="#4F81BD", label="Baseline"
    )
    ax_hist.hist(
        vol1, bins=bins, alpha=0.5, density=True, color="#F28E2B", label="12 mo"
    )
    x_kde = np.linspace(bins[0], bins[-1], 200)
    ax_hist.plot(x_kde, st.gaussian_kde(vol0)(x_kde), color="#4F81BD", lw=2)
    ax_hist.plot(x_kde, st.gaussian_kde(vol1)(x_kde), color="#F28E2B", lw=2)
    ax_hist.axvline(mean0, ls="--", color="#4F81BD")
    ax_hist.axvline(mean1, ls="--", color="#F28E2B")
    ax_hist.set_xlabel("Volume (mL)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Histogram / KDE")
    ax_hist.legend(fontsize=8)

    txt = (
        f"Mean baseline : {mean0:.6f} mL\n"
        f"Mean 12 months: {mean1:.6f} mL\n"
        f"Δ mean (12 - 0): {d_mean:+.6f} mL"
    )
    ax_hist.text(
        0.02,
        0.97,
        txt,
        transform=ax_hist.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
    )
    fig.suptitle(f"{roi_name} Volume (Baseline vs 12 mo)", fontsize=14)

    out_path = out_dir / f"{roi_name.replace(' ', '_')}_vol_distribution.png"
    fig.savefig(out_path, dpi=300)
    if show_plot:
        plt.show(fig)
    else:
        plt.close()
    print(f"Saved {out_path}")


def plot_delta_bar(
    vdf: pd.DataFrame,
    roi_id: Union[int, str],
    out_dir: str | Path = "stats_png",
    dpi: int = 300,
    show_plot: bool = False,
    cmap_name: str = "RdBu_r",
) -> None:

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    col = f"L{roi_id}" if isinstance(roi_id, int) else str(roi_id)
    if col not in vdf.columns:
        raise KeyError(f"ROI column '{col}' not found.")

    delta = vdf[col].dropna().sort_values(ascending=False)
    roi_name = ID2NAME.get(int(str(roi_id).lstrip("L")), col)

    norm = plt.Normalize(vmin=delta.min(), vmax=delta.max())
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(norm(delta.values))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(delta.index, delta.values, color=colors, width=0.8)
    ax.axhline(0, color="gray", lw=1)

    ax.set_xlabel(f"{len(vdf[col])} different subjects")
    ax.set_ylabel("Δ Volume (12mo - baseline) (%)")
    ax.set_title(f"{roi_name} Volume Change (Baseline vs 12 mo)", fontsize=14)
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.margins(x=0.01)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Δ Volume (12mo - baseline) (%)")

    plt.tight_layout()
    out_path = out_dir / f"{roi_name.replace(' ', '_')}_delta_pct.png"
    plt.savefig(out_path, dpi=dpi)
    print("Saved", out_path)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
