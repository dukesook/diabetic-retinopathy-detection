"""Create or import the patient-level split -> data/splits.csv

Columns: image, patient_id, eye, level, split

Option A (preferred): import your existing Oct 1 split
    python -m src.make_splits --labels trainLabels.csv --existing path/to/your_split.csv
    The existing file needs a 'split' column (train/val/test; 'validation' is accepted)
    and either 'patient_id' or 'image'.

Option B: make a new 70/15/15 patient split, stratified by patient grade, seed 42
    python -m src.make_splits --labels trainLabels.csv
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from .common import DATA, ensure_dirs, load_labels, parse_image_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--existing", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop_bad", default=str(DATA / "bad_images.csv"),
                    help="drop patients with a missing/unreadable image (from preprocess)")
    a = ap.parse_args()
    ensure_dirs()

    eyes = load_labels(a.labels)
    both = eyes.groupby("patient_id")["eye"].nunique()
    eyes = eyes[eyes["patient_id"].isin(both[both == 2].index)]

    try:
        bad = pd.read_csv(a.drop_bad)
        if len(bad):
            bad_pids = {parse_image_id(i)[0] for i in bad["image"]}
            print(f"dropping {len(bad_pids)} patients with bad images")
            eyes = eyes[~eyes["patient_id"].isin(bad_pids)]
    except FileNotFoundError:
        pass

    if a.existing:
        ex = pd.read_csv(a.existing)
        ex["split"] = ex["split"].str.lower().replace({"validation": "val", "valid": "val"})
        if "patient_id" not in ex.columns:
            ex["patient_id"] = ex["image"].map(lambda s: parse_image_id(s)[0])
        ex["patient_id"] = ex["patient_id"].astype(str)
        pmap = ex.drop_duplicates("patient_id").set_index("patient_id")["split"]
        eyes["split"] = eyes["patient_id"].map(pmap)
        missing = eyes["split"].isna().sum()
        if missing:
            print(f"WARNING: {missing} eye images not in the existing split; they are dropped")
            eyes = eyes.dropna(subset=["split"])
    else:
        pat = eyes.groupby("patient_id")["level"].max().reset_index()
        tr, rest = train_test_split(pat, test_size=0.30, stratify=pat["level"], random_state=a.seed)
        va, te = train_test_split(rest, test_size=0.50, stratify=rest["level"], random_state=a.seed)
        pmap = {**{p: "train" for p in tr.patient_id}, **{p: "val" for p in va.patient_id},
                **{p: "test" for p in te.patient_id}}
        eyes["split"] = eyes["patient_id"].map(pmap)

    # leakage check: one patient, one split
    assert eyes.groupby("patient_id")["split"].nunique().max() == 1, "patient in two splits!"

    eyes.to_csv(DATA / "splits.csv", index=False)
    print("patients per split:")
    print(eyes.groupby("split")["patient_id"].nunique())
    print("\npatient grade (max of eyes) per split:")
    pg = eyes.groupby(["split", "patient_id"])["level"].max().reset_index()
    print(pd.crosstab(pg["split"], pg["level"]))
    print(f"\nsaved {DATA / 'splits.csv'}")


if __name__ == "__main__":
    main()
