"""Study-level Dataset and DataLoader for knee MRI multi-label classification.

Supports two loading strategies:
  - DICOM: reads raw .dcm files from <image_dir>/<study_uid>/<series_uid>/
  - PNG/numpy: reads pre-cached images from <image_dir>/<study_uid>/*.png

Returns (image_tensor, label_tensor) where image_tensor is (C, H, W) or
(N_series, C, H, W) for multi-series inputs.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..schema import LABEL_NAMES


def load_labels_csv(csv_path: str | Path) -> dict[str, np.ndarray]:
    """Load competition-format labels CSV → {study_uid: float32[12]}."""
    labels = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["StudyInstanceUID"]
            vals = np.array([float(row[c]) for c in LABEL_NAMES], dtype=np.float32)
            labels[uid] = vals
    return labels


def load_series_csv(csv_path: str | Path) -> dict[str, dict[str, str]]:
    """Load train_series_descriptions.csv → {study_uid: {series_uid: description}}."""
    series_map: dict[str, dict[str, str]] = {}
    if not csv_path or not Path(csv_path).exists():
        return series_map
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            study = row["StudyInstanceUID"]
            s_uid = row["SeriesInstanceUID"]
            desc = row.get("SeriesDescription", "")
            series_map.setdefault(study, {})[s_uid] = desc
    return series_map


class KneeStudyDataset(Dataset):
    """Multi-label knee MRI study dataset.

    Args:
        image_dir: root directory containing study folders
        labels: {study_uid: float32[12]} label dict (or None for test)
        series_map: {study_uid: {series_uid: desc}} (optional, for series filtering)
        img_size: target (H, W)
        max_slices_per_series: cap on slices loaded per series
        max_series: cap on number of series loaded per study
        study_uids: explicit list of study UIDs (subset)
    """

    def __init__(
        self,
        image_dir: str | Path,
        labels: Optional[dict[str, np.ndarray]] = None,
        series_map: Optional[dict[str, dict[str, str]]] = None,
        img_size: tuple[int, int] = (224, 224),
        max_slices_per_series: int = 1,
        max_series: int = 6,
        study_uids: Optional[list[str]] = None,
        transforms=None,
    ):
        self.image_dir = Path(image_dir)
        self.labels = labels or {}
        self.series_map = series_map or {}
        self.img_size = img_size
        self.max_slices = max_slices_per_series
        self.max_series = max_series
        self.transforms = transforms

        if study_uids is not None:
            self.study_uids = [u for u in study_uids if u in self.labels or labels is None]
        else:
            self.study_uids = sorted(
                d.name for d in self.image_dir.iterdir()
                if d.is_dir() and (d.name in self.labels or labels is None)
            )

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int):
        from .dicom import scan_study_dir, pick_representative_slices, load_series_images

        study_uid = self.study_uids[idx]
        study_dir = self.image_dir / study_uid

        # load images
        series_dict = scan_study_dir(study_dir)
        all_series_images = []
        for s_uid, paths in list(series_dict.items())[: self.max_series]:
            selected = pick_representative_slices(paths, self.max_slices)
            arr = load_series_images(selected, self.img_size)  # (N, H, W)
            all_series_images.append(arr)

        if not all_series_images:
            # fallback: empty tensor (will be zeroed)
            image = torch.zeros(1, *self.img_size, dtype=torch.float32)
        else:
            # stack all series → (N_total, H, W) then to (N_total, 1, H, W)
            stacked = np.concatenate(all_series_images, axis=0)
            image = torch.from_numpy(stacked).unsqueeze(1).float() / 255.0

        # label
        if study_uid in self.labels:
            label = torch.from_numpy(self.labels[study_uid])
        else:
            label = torch.zeros(len(LABEL_NAMES), dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)

        return image, label, study_uid
