import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
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


def plot_roi_hist(
    vdf: pd.DataFrame,
    roi_id: Union[int, str],
    color: str = "#4F81BD",
    out_dir: str | Path = "stats_png",
    show_plot: bool = False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    col = f"L{roi_id}" if isinstance(roi_id, int) else str(roi_id)
    if col not in vdf.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    roi_name = ID2NAME.get(roi_id, col) if "ID2NAME" in globals() else col
    vol = vdf[col].dropna()
    mean_val = vol.mean()

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    bins = np.linspace(vol.min(), vol.max(), 30)
    ax.hist(vol, bins=bins, density=True, alpha=0.5, color=color, label="Volume")
    x_kde = np.linspace(bins[0], bins[-1], 200)
    ax.plot(x_kde, st.gaussian_kde(vol)(x_kde), color=color, lw=2)
    ax.axvline(mean_val, ls="--", color=color, label=f"Mean = {mean_val:.3f} mL")

    ax.set_xlabel("Volume (mL)")
    ax.set_ylabel("Density")
    ax.set_title("Histogram / KDE")
    ax.legend(fontsize=9)

    fig.suptitle(f"{roi_name} Volume Distribution", fontsize=14)

    out_path = out_dir / f"{roi_name.replace(' ', '_')}_vol_hist.png"
    fig.savefig(out_path, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved {out_path}")


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

    ax.bar(range(len(delta)), delta.values, color=colors, width=0.8)
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


def plot_delta_hist_kde(
    vdf: pd.DataFrame,
    roi_id: Union[int, str],
    out_dir: str | Path = "stats_png",
    dpi: int = 300,
    bins: int = 30,
    show_plot: bool = False,
    kde_bw: float | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    col = f"L{roi_id}" if isinstance(roi_id, int) else str(roi_id)
    if col not in vdf.columns:
        raise KeyError(f"ROI column '{col}' not found in DataFrame")
    delta = vdf[col].dropna()
    if delta.empty:
        raise ValueError(f"No Δ% values found for ROI {col}")

    roi_name = ID2NAME.get(int(str(roi_id).lstrip("L")), col)
    mean, median, sd = delta.mean(), delta.median(), delta.std(ddof=1)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        delta,
        bins=bins,
        alpha=0.5,
        color="#4F81BD",
        edgecolor="white",
        density=True,
        label="Histogram",
    )
    sns.kdeplot(
        delta, bw_adjust=kde_bw or 1.0, color="#CA0020", lw=2, ax=ax, label="KDE"
    )
    ax.axvline(mean, color="#4F81BD", ls="--", lw=1)
    ax.axvline(median, color="#CA0020", ls=":", lw=1)
    ax.text(
        mean,
        ax.get_ylim()[1] * 0.9,
        f"μ={mean:+.2f}%",
        color="#4F81BD",
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.text(
        median,
        ax.get_ylim()[1] * 0.8,
        f"Med={median:+.2f}%",
        color="#CA0020",
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.set_xlabel("Δ Volume (12 m − baseline) (%)")
    ax.set_ylabel("Density")
    ax.set_title(f"{roi_name}  Δ% Distribution  (n={len(delta)})", fontsize=13)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / f"{roi_name.replace(' ', '_')}_delta_hist_kde.png"
    fig.savefig(out_path, dpi=dpi)
    print("Saved", out_path)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
