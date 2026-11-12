"""Save per-eye probabilities for val and test (everything after this uses only these CSVs).
Temperature scaling is fit on VAL only and applied to both.

Example:
    python -m src.predict --seed 42
Writes outputs/probs_seed42_val.csv and outputs/probs_seed42_test.csv
(columns: image, patient_id, eye, level, p0..p4) and the raw (uncalibrated) versions.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from .common import CHECKPOINTS, DATA, NUM_CLASSES, OUTPUTS, RESULTS, ensure_dirs, set_seed
from .dataset import EyeDataset


@torch.no_grad()
def get_logits(model, loader, device, amp, tta: bool):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            z = model(x).float()
            if tta:
                z = (z + model(torch.flip(x, dims=[3])).float()) / 2
        out.append(z.cpu())
    return torch.cat(out)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().detach())


def save(df, probs, path):
    out = df[["image", "patient_id", "eye", "level"]].copy().reset_index(drop=True)
    for k in range(NUM_CLASSES):
        out[f"p{k}"] = probs[:, k]
    out.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="checkpoint/run name; default seed{seed}")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tta", action="store_true", help="average with horizontal flip")
    ap.add_argument("--img_dir", default=None, help="default: data/processed (or data/processed512 if size 512)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ensure_dirs()
    set_seed(a.seed)
    tag = a.tag or f"seed{a.seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    ck = torch.load(CHECKPOINTS / f"{tag}.pt", map_location="cpu", weights_only=False)
    model = timm.create_model(ck["model"], pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    img_dir = a.img_dir or str(DATA / ("processed" if ck["size"] == 384 else f"processed{ck['size']}"))
    print(f"checkpoint {tag}: {ck['model']} size {ck['size']}, images from {img_dir}")
    df = pd.read_csv(DATA / "splits.csv", dtype={"patient_id": str})
    logits, frames, timing = {}, {}, {}
    for split in ("val", "test"):
        d = df[df.split == split]
        if a.limit:  # keep whole patients
            pids = d["patient_id"].drop_duplicates().iloc[: a.limit // 2]
            d = d[d["patient_id"].isin(pids)]
        d = d.reset_index(drop=True)
        dl = DataLoader(EyeDataset(d, img_dir, ck["size"], False), batch_size=a.batch,
                        num_workers=a.workers, pin_memory=amp)
        t0 = time.time()
        logits[split] = get_logits(model, dl, device, amp, a.tta)
        timing[split] = (time.time() - t0) / len(d) * 1000
        frames[split] = d
        print(f"{split}: {len(d)} images, {timing[split]:.2f} ms/image")

    yv = torch.tensor(frames["val"]["level"].to_numpy(), dtype=torch.long)
    T = fit_temperature(logits["val"], yv)
    print(f"temperature (fit on val): {T:.3f}")

    for split in ("val", "test"):
        raw = torch.softmax(logits[split], 1).numpy()
        cal = torch.softmax(logits[split] / T, 1).numpy()
        save(frames[split], raw, OUTPUTS / f"probs_raw_{tag}_{split}.csv")
        save(frames[split], cal, OUTPUTS / f"probs_{tag}_{split}.csv")

    with open(RESULTS / f"predict_info_{tag}.json", "w") as f:
        json.dump({"temperature": T, "cnn_ms_per_image": timing,
                   "device": torch.cuda.get_device_name(0) if amp else "cpu",
                   "tta": a.tta, "checkpoint_epoch": ck["epoch"],
                   "checkpoint_val_qwk": ck["val_qwk"]}, f, indent=2)
    print("saved probabilities to outputs/")


if __name__ == "__main__":
    main()
