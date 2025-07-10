#!/usr/bin/env python3
# nohup python roi_volumes.py > logs/out_roi_volumes.log 2>&1 &

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from pathlib import Path
from tqdm.auto import tqdm
from choose_best_t1w import *

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

root_dir = Path(cfg["root_dir"])
pattern_run = cfg["pattern_run"]

csv = cfg["roi_volumes_csv"]
csv_z = cfg["roi_volumes_z_csv"]
csv00 = cfg["roi_volumes_ses00_csv"]
csv12 = cfg["roi_volumes_ses12_csv"]
csv00_z = cfg["roi_volumes_ses00_z_csv"]
csv12_z = cfg["roi_volumes_ses12_z_csv"]

label_df = pd.read_csv(cfg["label_index"], usecols=["IDX"])
LABEL_LIST = sorted(label_df["IDX"].astype(int))

subjectids = {
    f"{p.parents[4].name}_{p.parents[3].name}" for p in root_dir.glob(pattern_run)
}
print(f"Found {len(subjectids)} unique subject-session pairs")

seg_paths_best = []
for sid in tqdm(sorted(subjectids), desc="Picking best run"):
    _, best_t1 = choose_best_t1w_subjectid(sid)
    best_t1 = Path(best_t1)
    if "_run-" in best_t1.name:
        run_tag = best_t1.name.split("_run-")[1].split("_")[0]
        seg_path = (
            root_dir
            / sid.split("_")[0]
            / sid.split("_")[1]
            / f"SLANT-TICVv1.2run-{run_tag}/post/FinalResult"
            / best_t1.name.replace(f"_run-{run_tag}_T1w", f"_run-{run_tag}_T1w_seg")
        )
    else:
        seg_path = (
            root_dir
            / sid.split("_")[0]
            / sid.split("_")[1]
            / "SLANT-TICVv1.2/post/FinalResult"
            / best_t1.name.replace("_T1w", "_T1w_seg")
        )
    if seg_path.exists():
        seg_paths_best.append(seg_path)
print(f"Selected {len(seg_paths_best)} best segmentation files")

subject_rows = {}
for seg_path in tqdm(seg_paths_best, desc="Computing ROI volumes"):
    subject_dir = seg_path.parents[4].name
    session_dir = seg_path.parents[3].name
    sess_id = session_dir.replace("ses-", "")

    img = nib.load(seg_path)
    seg = img.get_fdata().astype(np.int32)
    vox_vol = np.prod(img.header.get_zooms()[:3]) / 1e3

    row = {"subject": f"{subject_dir}_{session_dir}"}
    for lab in LABEL_LIST:
        row[f"L{lab}"] = (seg == lab).sum() * vox_vol

    subject_rows.setdefault(subject_dir, {})[sess_id] = row

rows00, rows12, rows_all = [], [], []
for subj, sess_dict in subject_rows.items():
    if {"00", "12"} <= sess_dict.keys():
        rows00.append(sess_dict["00"])
        rows12.append(sess_dict["12"])
        rows_all.extend([sess_dict["00"], sess_dict["12"]])

print(f"Subjects with both sessions: {len(rows00)}")


def make_df(rows, out_csv, z_csv):
    if not rows:
        print(f"No rows for {out_csv}; skipped.")
        return None, None
    df = pd.DataFrame(rows).set_index("subject").sort_index()
    zdf = df.apply(lambda col: (col - col.mean()) / col.std(ddof=0), axis=0)
    df.to_csv(out_csv)
    zdf.to_csv(z_csv)
    print(f"Saved: {out_csv}  (shape {df.shape})")
    print(f"Saved: {z_csv}")
    return df, zdf


df00, _ = make_df(rows00, csv00, csv00_z)
df12, _ = make_df(rows12, csv12, csv12_z)
df_all, _ = make_df(rows_all, csv, csv_z)

for name, df in [("ses-00", df00), ("ses-12", df12), ("combined", df_all)]:
    if df is not None:
        print(f"\nFirst 5 rows ({name}):")
        print(df.iloc[:5, :8])

print("\nDone.")
