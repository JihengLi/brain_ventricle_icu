import numpy as np
import pandas as pd
import nibabel as nib
import yaml, re

from scipy.ndimage import laplace, binary_opening
from pathlib import Path
from typing import Sequence, Tuple, Dict, List

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

slant_root_dir = cfg["root_dir"]
t1_root_dir = cfg["t1_root_dir"]


def _pick_exist(paths: Sequence[Path]) -> List[Path]:
    exist_paths: List[Path] = [p for p in paths if p.exists()]
    if not exist_paths:
        raise FileNotFoundError(
            "None of the following paths exist:\n" + "\n".join(map(str, paths))
        )
    return exist_paths


def choose_best_t1w(
    paths: Sequence[str | Path],
    seg_paths: Sequence[str | Path] | None = None,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[pd.DataFrame, Path]:
    rows: list[Dict] = []
    for i, p in enumerate(paths):
        p = Path(p)
        img = nib.load(p, mmap=True).get_fdata(dtype=np.float32)
        if seg_paths:
            seg = nib.load(seg_paths[i], mmap=True).get_fdata(dtype=np.float32)
            mask = seg != 0
        else:
            thr = np.percentile(img, 50)
            mask = binary_opening(img > thr, iterations=2)

        brain, bg = img[mask], img[~mask]
        snr = brain.mean() / (bg.std() + 1e-5)
        sharp = laplace(img).var()
        size = np.count_nonzero(img)
        rows.append({"file": p, "SNR": snr, "Sharp": sharp, "Size": size})

    df = pd.DataFrame(rows)
    for col in ("SNR", "Sharp", "Size"):
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    w_snr, w_sharp, w_size = weights
    df["Score"] = w_snr * df["SNR_z"] + w_sharp * df["Sharp_z"] + w_size * df["Size_z"]
    best_path = Path(df.loc[df["Score"].idxmax(), "file"])
    return df, best_path


def choose_best_t1w_subjectid(subjectid: str) -> Tuple[pd.DataFrame | None, Path]:
    RUN_DIR_RE = re.compile(r"SLANT-TICVv1\.2run-(\d+)$")
    sub, ses = subjectid.split("_")
    base_seg_dir = Path(slant_root_dir) / sub / ses
    base_t1_dir = Path(t1_root_dir) / sub / ses
    seg_candidates: List[Path] = []
    t1_candidates: List[Path] = []

    for run_dir in sorted(base_seg_dir.glob("SLANT-TICVv1.2run-*")):
        m = RUN_DIR_RE.match(run_dir.name)
        if not m:
            continue
        run_num = m.group(1)
        seg_path = (
            run_dir / "post/FinalResult" / f"{subjectid}_run-{run_num}_T1w_seg.nii.gz"
        )
        seg_candidates.append(seg_path)
        t1_path = base_t1_dir / "anat" / f"{subjectid}_run-{run_num}_T1w.nii.gz"
        t1_candidates.append(t1_path)

    seg_candidates.append(
        base_seg_dir / "SLANT-TICVv1.2/post/FinalResult" / f"{subjectid}_T1w_seg.nii.gz"
    )
    t1_candidates.append(base_t1_dir / "anat" / f"{subjectid}_T1w.nii.gz")
    seg_files = _pick_exist(seg_candidates)
    t1_files = _pick_exist(t1_candidates)
    if len(seg_files) == 1 or len(t1_files) == 1 or len(seg_files) != len(t1_files):
        return None, t1_files[0]
    df, best_path = choose_best_t1w(t1_files, seg_files)
    return df, best_path
