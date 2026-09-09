"""Histogram utilities shared by every objective function.

The optimisation problem is defined purely over the image's intensity histogram,
so all DE variants evaluate fitness against the normalised histogram produced
here -- never against the raw pixels. Computing it once per image keeps the inner
optimisation loop cheap.
"""

from __future__ import annotations

import numpy as np

from .config import GRAY_LEVELS


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit (uint8) single-channel view of ``image``.

    Accepts grayscale or RGB(A) arrays. RGB is collapsed with the standard
    luma weights; floating images in [0, 1] are rescaled to 0..255.
    """
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., :3] @ np.array([0.2989, 0.5870, 0.1140])
    if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
        arr = arr * (GRAY_LEVELS - 1)
    return np.clip(np.rint(arr), 0, GRAY_LEVELS - 1).astype(np.uint8)


def histogram_counts(image: np.ndarray) -> np.ndarray:
    """Integer frequency of each intensity level, length ``GRAY_LEVELS``."""
    gray = to_grayscale(image)
    counts = np.bincount(gray.ravel(), minlength=GRAY_LEVELS)
    return counts[:GRAY_LEVELS].astype(np.int64)


def normalised_histogram(image: np.ndarray) -> np.ndarray:
    """Probability distribution p_i = h(i) / (M*N), length ``GRAY_LEVELS``.

    This is the ``hist`` argument every objective in :mod:`thresholding.objectives`
    expects: a non-negative float array that sums to 1.
    """
    counts = histogram_counts(image)
    total = counts.sum()
    if total == 0:
        return np.full(GRAY_LEVELS, 1.0 / GRAY_LEVELS)
    return counts / total
