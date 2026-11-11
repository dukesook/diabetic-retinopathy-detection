"""Average calibrated per-eye probabilities from several runs into one ensemble.

Example:
    python -m src.ensemble --tags b3_seed42 b3_seed1 b3_seed2 --out b3_ens
Writes outputs/probs_{out}_val.csv and outputs/probs_{out}_test.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .common import OUTPUTS, RESULTS, ensure_dirs

PCOLS = [f"p{k}" for k in range(5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ensure_dirs()

    for split in ("val", "test"):
        frames = [pd.read_csv(OUTPUTS / f"probs_{t}_{split}.csv", dtype={"patient_id": str})
                  .sort_values("image").reset_index(drop=True) for t in a.tags]
        base = frames[0]
        for f in frames[1:]:
            if not (f["image"].to_numpy() == base["image"].to_numpy()).all():
                raise ValueError(f"{split}: runs do not cover the same images")
        avg = np.mean([f[PCOLS].to_numpy() for f in frames], axis=0)
        avg = avg / avg.sum(1, keepdims=True)
        out = base[["image", "patient_id", "eye", "level"]].copy()
        out[PCOLS] = avg
        out.to_csv(OUTPUTS / f"probs_{a.out}_{split}.csv", index=False)
        acc = (avg.argmax(1) == out["level"].to_numpy()).mean()
        print(f"{split}: {len(out)} images, ensemble per-eye accuracy {acc:.4f}")

    infos = []
    for t in a.tags:
        p = RESULTS / f"predict_info_{t}.json"
        if p.exists():
            infos.append(json.loads(p.read_text()))
    ms = {s: float(sum(i["cnn_ms_per_image"][s] for i in infos)) for s in ("val", "test")} if infos else {}
    with open(RESULTS / f"predict_info_{a.out}.json", "w") as f:
        json.dump({"ensemble_of": a.tags, "cnn_ms_per_image": ms,
                   "device": infos[0]["device"] if infos else "n/a"}, f, indent=2)
    print(f"saved outputs/probs_{a.out}_*.csv")


if __name__ == "__main__":
    main()
