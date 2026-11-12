"""Shared helpers: paths, seeds, label parsing."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

NUM_CLASSES = 5
REFERRAL_THRESHOLD = 2  # referable DR = patient grade >= 2


def ensure_dirs() -> None:
    for d in (DATA, OUTPUTS, RESULTS, RESULTS / "plots", CHECKPOINTS):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_image_id(image: str) -> tuple[str, str]:
    """'10_left' -> ('10', 'left')."""
    pid, eye = image.rsplit("_", 1)
    if eye not in ("left", "right"):
        raise ValueError(f"unexpected image id: {image}")
    return pid, eye


def load_labels(labels_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if not {"image", "level"} <= set(df.columns):
        raise ValueError("labels CSV needs columns: image, level")
    parsed = df["image"].map(parse_image_id)
    df["patient_id"] = parsed.map(lambda x: x[0])
    df["eye"] = parsed.map(lambda x: x[1])
    return df[["image", "patient_id", "eye", "level"]]


def to_patient_table(eye_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pivot one-row-per-eye into one-row-per-patient with *_left / *_right columns.
    Drops patients that do not have both eyes."""
    wide = eye_df.pivot(index="patient_id", columns="eye", values=cols)
    wide.columns = [f"{c}_{e}" for c, e in wide.columns]
    wide = wide.dropna().reset_index()
    return wide
