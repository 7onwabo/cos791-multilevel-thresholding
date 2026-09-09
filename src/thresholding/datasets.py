"""Dataset loaders for the two experiment phases.

Phase 1 (proof of concept): BDS500 -- 10 natural images (``imgN.png``) each with a
ground-truth segmentation (``imgN_gt.png``).
Phase 2 (medical): CHAOS -- 15 abdominal MRI slices (``IMG-XXXX-00010.png``).

Loaders return plain uint8 grayscale numpy arrays keyed by a short id, so the rest
of the pipeline never touches the filesystem layout directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .config import BDS500_DIR, CHAOS_DIR
from .histogram import to_grayscale


def _load_gray(path: Path) -> np.ndarray:
    return to_grayscale(iio.imread(path))


def load_bds500(directory: Path = BDS500_DIR) -> dict[str, np.ndarray]:
    """Return {"img1": array, ...} for the 10 BDS500 source images.

    Ground-truth masks (``*_gt.png``) are excluded here; load them via
    :func:`load_bds500_ground_truth` when computing supervised metrics.
    """
    images: dict[str, np.ndarray] = {}
    for path in sorted(directory.glob("img*.png")):
        if path.stem.endswith("_gt"):
            continue
        images[path.stem] = _load_gray(path)
    return images


def load_bds500_ground_truth(directory: Path = BDS500_DIR) -> dict[str, np.ndarray]:
    """Return {"img1": gt_array, ...} for the BDS500 ground-truth masks."""
    gts: dict[str, np.ndarray] = {}
    for path in sorted(directory.glob("img*_gt.png")):
        key = path.stem[: -len("_gt")]
        gts[key] = _load_gray(path)
    return gts


def _natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.stem)]


def load_chaos(directory: Path = CHAOS_DIR) -> dict[str, np.ndarray]:
    """Return {"IMG-0003-00010": array, ...} for the 15 CHAOS MRI slices."""
    paths = sorted(directory.glob("*.png"), key=_natural_key)
    return {path.stem: _load_gray(path) for path in paths}
