"""Train the learned component: per-eye DR grade classifier (0-4).

Example:
    python -m src.train_cnn --seed 42 --limit 500 --epochs 1        # smoke test
    python -m src.train_cnn --seed 42                                # full run
    python -m src.train_cnn --seed 42 --model efficientnet_b3 --size 512 --img_dir data/processed512 --tag b3_seed42
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .common import CHECKPOINTS, DATA, NUM_CLASSES, RESULTS, ensure_dirs, set_seed
from .dataset import EyeDataset
from .metrics import qwk


def class_weights(levels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(levels, minlength=NUM_CLASSES).astype(float)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))  # inverse sqrt frequency (milder than inverse)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    return qwk(y, p), float((p == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--tag", default=None, help="run name; default seed{seed}")
    ap.add_argument("--img_dir", default=str(DATA / "processed"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    ensure_dirs()
    set_seed(a.seed)
    tag = a.tag or f"seed{a.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"device: {device} {torch.cuda.get_device_name(0) if amp else ''}")

    df = pd.read_csv(DATA / "splits.csv")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if a.limit:
        tr = tr.sample(min(a.limit, len(tr)), random_state=a.seed)
        va = va.sample(min(max(a.limit // 4, 50), len(va)), random_state=a.seed)
    print(f"train images: {len(tr)}  val images: {len(va)}")

    g = torch.Generator().manual_seed(a.seed)
    dl_tr = DataLoader(EyeDataset(tr, a.img_dir, a.size, True), batch_size=a.batch, shuffle=True,
                       num_workers=a.workers, pin_memory=amp, drop_last=True, generator=g,
                       persistent_workers=a.workers > 0)
    dl_va = DataLoader(EyeDataset(va, a.img_dir, a.size, False), batch_size=a.batch * 2,
                       num_workers=a.workers, pin_memory=amp)

    model = timm.create_model(a.model, pretrained=not a.no_pretrained, num_classes=NUM_CLASSES).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights(tr["level"].to_numpy()).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(dl_tr))
    scaler = torch.amp.GradScaler(enabled=amp)

    ckpt = CHECKPOINTS / f"{tag}.pt"
    best, bad_epochs, log = -1.0, 0, []
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for step, (x, y) in enumerate(dl_tr, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            if step % 100 == 0:
                print(f"  ep {ep} step {step}/{len(dl_tr)} loss {run_loss/step:.4f}", flush=True)
        k, acc = evaluate(model, dl_va, device, amp)
        row = {"epoch": ep, "train_loss": run_loss / max(len(dl_tr), 1), "val_qwk": k,
               "val_acc": acc, "epoch_sec": time.time() - t0}
        log.append(row)
        print(row, flush=True)
        if k > best:
            best, bad_epochs = k, 0
            torch.save({"model": a.model, "size": a.size, "state_dict": model.state_dict(),
                        "epoch": ep, "val_qwk": k, "seed": a.seed}, ckpt)
            print(f"  saved best -> {ckpt}")
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                print("early stopping")
                break

    info = {"tag": tag, "seed": a.seed, "model": a.model, "timm_version": timm.__version__,
            "torch_version": torch.__version__, "size": a.size, "best_val_qwk": best,
            "train_minutes": (time.time() - t_start) / 60,
            "device": torch.cuda.get_device_name(0) if amp else "cpu",
            "n_train": len(tr), "n_val": len(va), "epochs_log": log}
    with open(RESULTS / f"train_log_{tag}.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"best val QWK {best:.4f}; log at {RESULTS / f'train_log_seed{a.seed}.json'}")


if __name__ == "__main__":
    main()
