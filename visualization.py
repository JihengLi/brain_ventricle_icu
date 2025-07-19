import nibabel as nib
import numpy as np
import yaml
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from pathlib import Path
from collections.abc import Sequence
from typing import List
from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
)

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

slant_root_dir = cfg["root_dir"]
t1_root_dir = cfg["t1_root_dir"]
lut_addr = cfg["lut_addr"]


def load_lut(lut_path: str, bg_transparent: bool = True):
    idx_list, rgba_list = [], []
    with open(lut_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            idx = int(parts[0])
            r, g, b = map(int, parts[1:4])
            a = float(parts[4])
            idx_list.append(idx)
            rgba_list.append((r / 255.0, g / 255.0, b / 255.0, a))

    max_idx = max(idx_list)
    full_rgba = [(0, 0, 0, 0)] * (max_idx + 1)
    for i, idx in enumerate(idx_list):
        full_rgba[idx] = rgba_list[i]

    if bg_transparent and 0 <= max_idx:
        full_rgba[0] = (0, 0, 0, 0)

    cmap = ListedColormap(full_rgba, name="slant_lut")
    bounds = np.arange(max_idx + 2) - 0.5
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    return cmap, norm


def _normalize_slices(
    val: int | str | Sequence[int | str], size: int
) -> tuple[int, ...]:
    if isinstance(val, (str, int)):
        val = (val,)
    idxs = []
    for v in val:
        idx = size // 2 if v == "mid" else int(v)
        if not 0 <= idx < size:
            raise ValueError(f"Slice index {idx} out of bounds (0‥{size-1})")
        idxs.append(idx)
    return tuple(idxs)


def _keep_roi(arr: np.ndarray, roi: List) -> np.ndarray:
    mask = np.isin(arr, roi)
    out = np.where(mask, arr, 0)
    return out


def _pick_first_exist(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the following paths exist:\n" + "\n".join(map(str, paths))
    )


def _load_as_ras(path):
    img = nib.load(path, mmap=True)
    data = img.get_fdata(dtype=np.float32)
    src = io_orientation(img.affine)
    tgt = axcodes2ornt(("R", "A", "S"))
    trf = ornt_transform(src, tgt)
    return apply_orientation(data, trf)


def visualize_slant(
    seg_file: str | Path,
    lut_file: str | Path,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    keep_roi_list: List | None = None,
    auto_slice: bool = False,
    t1_file: str | Path | None = None,
    alpha_seg: float = 0.6,
    save_path: Path | None = None,
    show_img: bool = True,
) -> plt.Figure:
    seg_data = _load_as_ras(seg_file)
    cmap, norm = load_lut(lut_file, bg_transparent=t1_file != None)

    if t1_file:
        t1_data = _load_as_ras(t1_file)
        if t1_data.shape != seg_data.shape:
            raise ValueError("T1 shape ≠ seg shape")
    else:
        t1_data = None
        alpha_seg = 1.0

    if auto_slice:
        if keep_roi_list is None:
            raise ValueError("auto_slice=True requires keep_roi_list to be provided")
        if sagittal_slices != "mid" or coronal_slices != "mid" or axial_slices != "mid":
            warnings.warn(
                "auto_slice=True: Ignoring provided slices, choosing best slices based on area.",
                UserWarning,
            )
        areas_x = [
            np.count_nonzero(np.isin(seg_data[i, :, :], keep_roi_list))
            for i in range(seg_data.shape[0])
        ]
        best_x = int(np.argmax(areas_x))
        areas_y = [
            np.count_nonzero(np.isin(seg_data[:, j, :], keep_roi_list))
            for j in range(seg_data.shape[1])
        ]
        best_y = int(np.argmax(areas_y))
        areas_z = [
            np.count_nonzero(np.isin(seg_data[:, :, k], keep_roi_list))
            for k in range(seg_data.shape[2])
        ]
        best_z = int(np.argmax(areas_z))

        sagittal_slices = [best_x]
        coronal_slices = [best_y]
        axial_slices = [best_z]

    x_idxs = _normalize_slices(sagittal_slices, seg_data.shape[0])
    y_idxs = _normalize_slices(coronal_slices, seg_data.shape[1])
    z_idxs = _normalize_slices(axial_slices, seg_data.shape[2])

    combos = [(x, y, z) for x in x_idxs for y in y_idxs for z in z_idxs]
    n_rows = len(combos)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, 3)

    for row_idx, (x, y, z) in enumerate(combos):
        slices_seg = (
            np.rot90(seg_data[x, :, :]),
            np.rot90(seg_data[:, y, :]),
            np.rot90(seg_data[:, :, z]),
        )
        if keep_roi_list:
            slices_seg = tuple(_keep_roi(slc, keep_roi_list) for slc in slices_seg)
        if t1_data is not None:
            slices_t1 = (
                np.rot90(t1_data[x, :, :]),
                np.rot90(t1_data[:, y, :]),
                np.rot90(t1_data[:, :, z]),
            )
        titles = (
            f"Sagittal X={x}",
            f"Coronal  Y={y}",
            f"Axial    Z={z}",
        )
        for col_idx, (seg_slc, title) in enumerate(zip(slices_seg, titles)):
            ax = axes[row_idx, col_idx]
            if t1_data is not None:
                ax.imshow(slices_t1[col_idx], cmap="gray", interpolation="nearest")
            ax.imshow(
                seg_slc, cmap=cmap, norm=norm, interpolation="nearest", alpha=alpha_seg
            )
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_img:
        plt.show()
    else:
        plt.close(fig)
    return fig


def visualize_slant_compare(
    seg_file_a: str | Path,
    seg_file_b: str | Path,
    t1_file_a: str | Path | None = None,
    t1_file_b: str | Path | None = None,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    keep_roi_list: List[int] | None = None,
    auto_slice: bool = False,
    alpha_seg: float = 0.7,
    save_path: Path | None = None,
    show_img: bool = True,
) -> plt.Figure:
    segA = _load_as_ras(seg_file_a)
    segB = _load_as_ras(seg_file_b)
    if segA.shape != segB.shape:
        raise ValueError("Two segmentations shape mismatch")

    if (t1_file_a is None) ^ (t1_file_b is None):
        raise ValueError("Either provide both t1_file_a & t1_file_b, or neither.")

    if t1_file_a and t1_file_b:
        t1A = _load_as_ras(t1_file_a)
        t1B = _load_as_ras(t1_file_b)
        if t1A.shape != segA.shape or t1B.shape != segA.shape:
            raise ValueError("T1 shape mismatch")
        t1_data = (t1A.astype(np.float32) + t1B.astype(np.float32)) / 2
    else:
        t1_data = None

    if keep_roi_list is None:
        raise ValueError("keep_roi_list must be set")
    maskA = np.isin(segA, keep_roi_list)
    maskB = np.isin(segB, keep_roi_list)

    diff_map = np.zeros(segA.shape, dtype=np.uint8)
    diff_map[maskA & maskB] = 1
    diff_map[maskA & ~maskB] = 2
    diff_map[~maskA & maskB] = 3

    if auto_slice:
        best_x = np.argmax(diff_map.sum(axis=(1, 2)))
        best_y = np.argmax(diff_map.sum(axis=(0, 2)))
        best_z = np.argmax(diff_map.sum(axis=(0, 1)))
        sagittal_slices, coronal_slices, axial_slices = [best_x], [best_y], [best_z]

    x_idx = _normalize_slices(sagittal_slices, segA.shape[0])
    y_idx = _normalize_slices(coronal_slices, segA.shape[1])
    z_idx = _normalize_slices(axial_slices, segA.shape[2])

    combos = [(x, y, z) for x in x_idx for y in y_idx for z in z_idx]
    fig, axes = plt.subplots(len(combos), 3, figsize=(15, 5 * len(combos)))
    axes = np.asarray(axes).reshape(len(combos), 3)

    cmap = ListedColormap(
        [
            (0, 0, 0, 0),
            (77 / 255, 175 / 255, 74 / 255, alpha_seg),  # Common: green
            (215 / 255, 25 / 255, 28 / 255, alpha_seg),  # Only A: red
            (255 / 255, 215 / 255, 0 / 255, alpha_seg),  # Only B: yellow
        ]
    )

    orient_titles = ("Sagittal", "Coronal", "Axial")
    for row, (x, y, z) in enumerate(combos):
        diff_slices = (
            np.rot90(diff_map[x, :, :]),
            np.rot90(diff_map[:, y, :]),
            np.rot90(diff_map[:, :, z]),
        )
        if t1_data is not None:
            t1_slices = (
                np.rot90(t1_data[x, :, :]),
                np.rot90(t1_data[:, y, :]),
                np.rot90(t1_data[:, :, z]),
            )
        for col, (diff_slc, title_base) in enumerate(zip(diff_slices, orient_titles)):
            ax = axes[row, col]
            if t1_data is not None:
                ax.imshow(t1_slices[col], cmap="gray", interpolation="nearest")
            ax.imshow(diff_slc, cmap=cmap, interpolation="nearest", vmin=0, vmax=3)
            ax.set_title(
                f"{title_base} ({['X','Y','Z'][col]}={[x,y,z][col]})", fontsize=9
            )
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_img:
        plt.show()
    else:
        plt.close(fig)
    return fig


def visualize_slant_subjectid(
    subjectid: str,
    lut_file: str | Path = lut_addr,
    run_number: int = 1,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    keep_roi_list: List | None = None,
    auto_slice: bool = False,
    bg_t1_file: bool = False,
    alpha_seg: float = 0.6,
    save_path: Path | None = None,
    show_img: bool = True,
) -> plt.Figure:
    sub, ses = subjectid.split("_")
    base_seg_dir = Path(slant_root_dir) / sub / ses
    seg_candidates = [
        base_seg_dir
        / f"SLANT-TICVv1.2run-{run_number}/post/FinalResult"
        / f"{subjectid}_run-{run_number}_T1w_seg.nii.gz",
        base_seg_dir
        / "SLANT-TICVv1.2/post/FinalResult"
        / f"{subjectid}_T1w_seg.nii.gz",
    ]
    seg_file = _pick_first_exist(seg_candidates)
    t1_file = None
    if bg_t1_file:
        base_t1_dir = Path(t1_root_dir) / sub / ses / "anat"
        t1_candidates = [
            base_t1_dir / f"{subjectid}_run-{run_number}_T1w.nii.gz",
            base_t1_dir / f"{subjectid}_T1w.nii.gz",
        ]
        t1_file = _pick_first_exist(t1_candidates)
    return visualize_slant(
        seg_file,
        lut_file,
        sagittal_slices,
        coronal_slices,
        axial_slices,
        keep_roi_list,
        auto_slice,
        t1_file,
        alpha_seg,
        save_path,
        show_img,
    )


def visualize_slant_subjectid_compare(
    subjectid: str,
    run_number: int = 1,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    keep_roi_list: List[int] | None = None,
    auto_slice: bool = False,
    bg_t1_file: bool = False,
    alpha_seg: float = 0.7,
    save_path: Path | None = None,
    show_img: bool = True,
) -> plt.Figure:
    base_seg_dir = Path(slant_root_dir) / subjectid
    seg_candidates_a = [
        base_seg_dir
        / "ses-00"
        / f"SLANT-TICVv1.2run-{run_number}/post/FinalResult"
        / f"{subjectid}_ses-00_run-{run_number}_T1w_seg.nii.gz",
        base_seg_dir
        / "ses-00"
        / "SLANT-TICVv1.2/post/FinalResult"
        / f"{subjectid}_ses-00_T1w_seg.nii.gz",
    ]
    seg_file_a = _pick_first_exist(seg_candidates_a)

    seg_candidates_b = [
        base_seg_dir
        / "ses-12"
        / f"SLANT-TICVv1.2run-{run_number}/post/FinalResult"
        / f"{subjectid}_ses-12_run-{run_number}_T1w_seg.nii.gz",
        base_seg_dir
        / "ses-12"
        / "SLANT-TICVv1.2/post/FinalResult"
        / f"{subjectid}_ses-12_T1w_seg.nii.gz",
    ]
    seg_file_b = _pick_first_exist(seg_candidates_b)

    t1_file_a, t1_file_b = None, None
    if bg_t1_file:
        t1_root = Path(t1_root_dir)
        ses_tags = {"00": "a", "12": "b"}
        for ses, var_tag in ses_tags.items():
            base_dir = t1_root / subjectid / f"ses-{ses}" / "anat"
            t1_candidates = [
                base_dir / f"{subjectid}_ses-{ses}_run-{run_number}_T1w.nii.gz",
                base_dir / f"{subjectid}_ses-{ses}_T1w.nii.gz",
            ]
            found = _pick_first_exist(t1_candidates)
            if var_tag == "a":
                t1_file_a = found
            else:
                t1_file_b = found

    return visualize_slant_compare(
        seg_file_a,
        seg_file_b,
        t1_file_a,
        t1_file_b,
        sagittal_slices,
        coronal_slices,
        axial_slices,
        keep_roi_list,
        auto_slice,
        alpha_seg,
        save_path,
        show_img,
    )


def visualize_t1w(
    t1_file: str | Path,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    perc: tuple[float, float] = (1, 99),
    save_path: Path | None = None,
    show_img: bool = True,
) -> plt.Figure:
    t1_file = Path(t1_file)
    if not t1_file.exists():
        raise FileNotFoundError(t1_file)

    data_ras = _load_as_ras(t1_file)

    x_idxs = _normalize_slices(sagittal_slices, data_ras.shape[0])
    y_idxs = _normalize_slices(coronal_slices, data_ras.shape[1])
    z_idxs = _normalize_slices(axial_slices, data_ras.shape[2])

    combos = [(x, y, z) for x in x_idxs for y in y_idxs for z in z_idxs]
    n_rows = len(combos)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, 3)
    vmin, vmax = np.percentile(data_ras, perc)

    for row_idx, (x, y, z) in enumerate(combos):
        slices = (
            np.rot90(data_ras[x, :, :]),
            np.rot90(data_ras[:, y, :]),
            np.rot90(data_ras[:, :, z]),
        )
        titles = (
            f"Sagittal X={x}",
            f"Coronal  Y={y}",
            f"Axial    Z={z}",
        )
        for col_idx, (slc, title) in enumerate(zip(slices, titles)):
            ax = axes[row_idx, col_idx]
            ax.imshow(slc, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_img:
        plt.show()
    else:
        plt.close(fig)
    return fig


def visualize_t1w_subjectid(
    subjectid: str,
    root: str | Path = t1_root_dir,
    run_number: int = 1,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    perc: tuple[float, float] = (1, 99),
    save_path: Path | None = None,
    show_img: bool = True,
) -> plt.Figure:
    root = Path(root)
    sub, ses = subjectid.split("_")
    base_t1_dir = Path(root) / sub / ses / "anat"
    t1_candidates = [
        base_t1_dir / f"{subjectid}_run-{run_number}_T1w.nii.gz",
        base_t1_dir / f"{subjectid}_T1w.nii.gz",
    ]
    t1_file = _pick_first_exist(t1_candidates)
    return visualize_t1w(
        t1_file,
        sagittal_slices,
        coronal_slices,
        axial_slices,
        perc,
        save_path,
        show_img,
    )
