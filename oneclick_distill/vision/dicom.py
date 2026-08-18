"""DICOM loading, preprocessing, and series organisation for knee MRI.

Handles the RSNA competition layout:
    <image_dir>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm

Returns uint8 2-D arrays (H, W) with default windowing applied.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import numpy as np


def load_dicom_bytes(raw: bytes) -> dict:
    """Parse raw DICOM bytes without importing pydicom at module level."""
    import pydicom

    return pydicom.dcmread(io.BytesIO(raw), force=True)


def load_dicom_file(path: Path) -> dict:
    import pydicom

    return pydicom.dcmread(str(path), force=True)


def apply_window(pixel_array: np.ndarray, center: float, width: float) -> np.ndarray:
    lower = center - width / 2
    upper = center + width / 2
    out = np.clip(pixel_array, lower, upper)
    out = (out - lower) / max(upper - lower, 1)
    return (out * 255).astype(np.uint8)


def default_window(ds) -> tuple[float, float]:
    """Choose centre/width from DICOM header or fall back to soft-tissue."""
    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    if center is not None and width is not None:
        c = float(center[0]) if isinstance(center, (list, tuple)) else float(center)
        w = float(width[0]) if isinstance(width, (list, tuple)) else float(width)
        return c, w
    # soft-tissue default for knee MRI
    return 400.0, 600.0


def dicom_to_u8(ds) -> np.ndarray:
    """Convert a pydicom Dataset to a uint8 2-D image."""
    arr = ds.pixel_array.astype(np.float32)
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    slope = float(getattr(ds, "RescaleSlope", 1))
    arr = arr * slope + intercept
    c, w = default_window(ds)
    return apply_window(arr, c, w)


def scan_study_dir(study_dir: Path) -> dict[str, list[Path]]:
    """Return {series_uid: [dicom_paths]} for a study directory."""
    result: dict[str, list[Path]] = {}
    for series_dir in sorted(study_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        dicoms = sorted(series_dir.glob("*.dcm"))
        if dicoms:
            result[series_dir.name] = dicoms
    return result


def pick_representative_slices(
    dicom_paths: Sequence[Path],
    n_slices: int = 1,
) -> list[Path]:
    """Select evenly-spaced slices from a series (for memory-efficient loading)."""
    if len(dicom_paths) <= n_slices:
        return list(dicom_paths)
    indices = np.linspace(0, len(dicom_paths) - 1, n_slices, dtype=int)
    return [dicom_paths[i] for i in indices]


def load_series_images(
    dicom_paths: Sequence[Path],
    img_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Load and resize a list of DICOM slices → (N, H, W) uint8 array."""
    from torchvision.transforms.functional import resize, to_tensor

    images = []
    for p in dicom_paths:
        ds = load_dicom_file(p)
        u8 = dicom_to_u8(ds)
        t = to_tensor(u8)  # (1, H, W) float32 [0,1]
        t = resize(t, list(img_size), antialias=True)
        images.append(t.squeeze(0).numpy())
    return np.stack(images, axis=0)  # (N, H, W)
