from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transform(size: int, train: bool):
    ops = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    if train:
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [T.Resize((size, size), antialias=True), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


class EyeDataset(Dataset):
    """Streams preprocessed images from disk (never loads all into memory)."""

    def __init__(self, df, img_dir: str | Path, size: int, train: bool):
        self.images = df["image"].tolist()
        self.levels = df["level"].astype(int).tolist()
        self.img_dir = Path(img_dir)
        self.tf = build_transform(size, train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        path = self.img_dir / f"{self.images[i]}.jpg"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read {path}; run preprocess first")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img), self.levels[i]
