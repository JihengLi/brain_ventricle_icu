import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pathlib import Path

from visualization import *
from choose_best_t1w import *
from stats_plot import *

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

roi_volumes_csv = Path(cfg["roi_volumes_csv"])
roi_volumes_df = pd.read_csv(roi_volumes_csv)

label_csv = Path(cfg["label_index"])
label_df = pd.read_csv(label_csv)
ID2NAME = dict(zip(label_df["IDX"], label_df["LABEL"]))

roi_list = [4, 51, 52]
hist_color = "#4F81BD"
out_dir = Path("stats_png")
out_dir.mkdir(exist_ok=True)


def fig_to_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    buf.shape = (h, w, 3)
    return buf


for roi_id in roi_list:
    col = f"L{roi_id}"
    tmp = roi_volumes_df[["subject", col]].dropna().sort_values(col)
    vols = tmp[col].values
    subs = tmp["subject"].values

    min_val, med_val, max_val = vols[0], np.median(vols), vols[-1]
    min_sub, med_sub, max_sub = subs[0], subs[len(subs) // 2], subs[-1]
    selected = [
        ("Min", min_val, min_sub, "#D73027"),
        ("Med", med_val, med_sub, "#762A83"),
        ("Max", max_val, max_sub, "#FEE08B"),
    ]

    roi_name = ID2NAME.get(roi_id, col).replace(" ", "_")

    fig = plt.figure(constrained_layout=True, figsize=(8, 16))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 1])

    ax_hist = fig.add_subplot(gs[0, 0])
    bins = np.linspace(vols.min(), vols.max(), 30)
    ax_hist.hist(vols, bins=bins, density=True, alpha=0.5, color=hist_color)
    x_kde = np.linspace(bins[0], bins[-1], 200)
    ax_hist.plot(x_kde, st.gaussian_kde(vols)(x_kde), color=hist_color, lw=2)
    for label, val, _, axvline_color in selected:
        ax_hist.axvline(
            val, linestyle="--", label=f"{label} ({val:.2f} mL)", color=axvline_color
        )
    ax_hist.set_xlabel("Volume (mL)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"{roi_name} Volume Distribution")
    ax_hist.legend(fontsize=8)

    for i, (label, _, subj, _) in enumerate(selected, start=1):
        seg_fig = visualize_slant_subjectid(
            subjectid=subj,
            lut_file="labels/segmentation_QA.label",
            keep_roi_list=[roi_id],
            auto_slice=True,
            bg_t1_file=True,
            show_img=False,
        )
        arr = fig_to_array(seg_fig)
        plt.close(seg_fig)

        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(arr)
        ax.set_title(f"{label}: {subj}", fontsize=10)
        ax.axis("off")

    out_path = out_dir / f"{roi_name}_QA.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
