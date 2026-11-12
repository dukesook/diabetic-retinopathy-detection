"""Crop the black border around the retina and resize. Run once; resumable.

Example:
    python -m src.preprocess --src /data/eyepacs/train --labels /data/eyepacs/trainLabels.csv --size 384 --limit 200
    python -m src.preprocess --src /data/eyepacs/train --labels /data/eyepacs/trainLabels.csv --size 384
"""
from __future__ import annotations

import argparse
import csv
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np

from .common import DATA, ensure_dirs, load_labels

EXTS = (".jpeg", ".jpg", ".png", ".tif", ".tiff")


def find_file(src: Path, image_id: str) -> Path | None:
    for ext in EXTS:
        p = src / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def crop_and_resize(img: np.ndarray, size: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 10
    if mask.any():
        ys, xs = np.where(mask)
        img = img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=img.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0: y0 + h, x0: x0 + w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _work(args):
    src, out_dir, image_id, size = args
    out = out_dir / f"{image_id}.jpg"
    if out.exists():
        return image_id, "skip"
    path = find_file(src, image_id)
    if path is None:
        return image_id, "missing"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return image_id, "unreadable"
    img = crop_and_resize(img, size)
    tmp = out_dir / f"{image_id}.tmp.jpg"
    cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    tmp.replace(out)  # atomic: half-written files never look finished
    return image_id, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with raw Kaggle train images")
    ap.add_argument("--labels", required=True, help="trainLabels.csv")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--out", default=str(DATA / "processed"))
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    ensure_dirs()
    src, out_dir = Path(a.src), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = load_labels(a.labels)["image"].tolist()
    if a.limit:
        ids = ids[: a.limit]

    jobs = [(src, out_dir, i, a.size) for i in ids]
    counts = {"ok": 0, "skip": 0, "missing": 0, "unreadable": 0}
    bad = []
    t0 = time.time()
    with Pool(a.workers) as pool:
        for n, (image_id, status) in enumerate(pool.imap_unordered(_work, jobs, chunksize=16), 1):
            counts[status] += 1
            if status in ("missing", "unreadable"):
                bad.append((image_id, status))
            if n % 1000 == 0:
                el = time.time() - t0
                print(f"{n}/{len(jobs)}  {el/60:.1f} min  {counts}", flush=True)

    el = time.time() - t0
    with open(DATA / "bad_images.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "status"])
        w.writerows(bad)
    done = counts["ok"]
    print(f"finished in {el:.1f}s  {counts}")
    if done:
        per = el / done
        print(f"time per processed image: {per*1000:.1f} ms "
              f"(projected for all 35,126 images: {per*35126/60:.1f} min with {a.workers} workers)")
    print(f"bad images logged to {DATA / 'bad_images.csv'}")


if __name__ == "__main__":
    main()
