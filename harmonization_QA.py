import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

from visualization import *
from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
)

df = pd.read_csv("data_cache/data.csv", index=False)
root_out = Path("data_cache/montage_pngs")
root_out.mkdir(exist_ok=True)


def five_slice_lists_center(shape3, step1: int = 5, step2: int = 10, ax_shift: int = 0):
    idx_all = []
    for dim_i, dim in enumerate(shape3):
        c = dim // 2
        offsets = [-step2, -step1, 0, step1, step2]
        if dim_i == 2:
            c += ax_shift
        idx = [min(max(c + off, 0), dim - 1) for off in offsets]
        seen = set()
        idx = [x for x in idx if not (x in seen or seen.add(x))]
        k = 1
        while len(idx) < 5:
            for off in [step2 + k, -(step2 + k)]:
                cand = min(max(c + off, 0), dim - 1)
                if cand not in idx:
                    idx.append(cand)
                    if len(idx) == 5:
                        break
            k += 1
        idx_all.append(np.array(idx[:5]))
    return idx_all


def load_to_ras(nifti_path: Path):
    img = nib.load(str(nifti_path), mmap=True)
    data_native = img.get_fdata(dtype=np.float32)

    src_ornt = io_orientation(img.affine)
    tgt_ornt = axcodes2ornt(("R", "A", "S"))
    transform = ornt_transform(src_ornt, tgt_ornt)
    data_ras = apply_orientation(data_native, transform)
    return data_ras


def extract_slice_ras(img_ras, axis, idx):
    if axis == 0:
        return np.rot90(img_ras[idx, :, :])
    if axis == 1:
        return np.rot90(img_ras[:, idx, :])
    return np.rot90(img_ras[:, :, idx])


for f in df["filepath"]:
    p = Path(f)
    subj, sess = p.parts[6], p.parts[7]
    ori_path = Path(*p.parts[:8]) / "nii" / (p.parts[9].split("_n4_reg")[0] + ".nii.gz")

    img_ori_ras = load_to_ras(ori_path)
    img_ham_ras = load_to_ras(p)

    vmin_o, vmax_o = np.percentile(img_ori_ras, (1, 99))
    vmin_h, vmax_h = np.percentile(img_ham_ras, (1, 99))
    sag_o, cor_o, ax_o = five_slice_lists_center(img_ori_ras.shape, ax_shift=25)
    sag_h, cor_h, ax_h = five_slice_lists_center(img_ham_ras.shape)

    fig, axes = plt.subplots(6, 5, figsize=(11, 13), dpi=100)
    for ax in axes.ravel():
        ax.axis("off")
    for k, idx in enumerate(np.concatenate([sag_o, cor_o, ax_o])):
        axes[k // 5, k % 5].imshow(
            extract_slice_ras(
                img_ori_ras, axis=0 if k < 5 else 1 if k < 10 else 2, idx=idx
            ),
            cmap="gray",
            vmin=vmin_o,
            vmax=vmax_o,
        )
    for k, idx in enumerate(np.concatenate([sag_h, cor_h, ax_h]), start=15):
        r, c = divmod(k, 5)
        axes[r, c].imshow(
            extract_slice_ras(
                img_ham_ras, axis=0 if k < 20 else 1 if k < 25 else 2, idx=idx
            ),
            cmap="gray",
            vmin=vmin_h,
            vmax=vmax_h,
        )
    axes[0, 0].set_title("Original T1 Image", fontsize=12, loc="center")
    axes[3, 0].set_title("Harmonized T1 Image", fontsize=12, loc="center")
    fig.suptitle(f"sub-{subj} | ses-{sess}", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0.03, 0, 0.97, 0.95])
    fig.savefig(root_out / f"sub-{subj}_ses-{sess}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("Saved", root_out / f"sub-{subj}_ses-{sess}.png")

print("\nAll compare PNGs in", root_out.resolve())
