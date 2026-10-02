"""Evaluation metrics (Assignment section 2).

Phase 1 (BDS500) reconstruction quality:
    psnr, ssim, uniformity          ← Dev A
Phase 2 (CHAOS MRI) segmentation overlap vs ground truth:
    jaccard, dice                   ← Dev B (stubs)

The segmentation helper ``_segment`` uses the same class convention as
``objectives._base`` (edges = [-1, t_1, ..., t_K, L-1]) so that the regions
measured here match the regions the optimizer actually optimised.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

from ..objectives._base import sanitise_thresholds


def _segment(image: np.ndarray, thresholds) -> np.ndarray:
    """Map each pixel to its class mean intensity (reconstructed image).

    Uses the same boundary convention as ``objectives._base.class_masses``:
        C_0 = [0, t_1],  C_1 = (t_1, t_2],  ...,  C_K = (t_K, 255]
    """
    t = sanitise_thresholds(thresholds)
    edges = [-1, *t, 255]

    reconstructed = np.zeros_like(image, dtype=np.float64)
    for c in range(len(t) + 1):
        lo = edges[c] + 1
        hi = edges[c + 1]
        mask = (image >= lo) & (image <= hi)
        if np.any(mask):
            reconstructed[mask] = np.mean(image[mask].astype(np.float64))

    return reconstructed


def psnr(original: np.ndarray, thresholds) -> float:
    """Peak Signal-to-Noise Ratio between original and thresholded reconstruction."""
    reconstructed = _segment(original, thresholds)
    return float(skimage_psnr(original.astype(np.float64), reconstructed, data_range=255))


def ssim(original: np.ndarray, thresholds) -> float:
    """Structural Similarity Index (wraps ``skimage.metrics.structural_similarity``)."""
    reconstructed = _segment(original, thresholds)
    return float(skimage_ssim(original.astype(np.float64), reconstructed, data_range=255))


def uniformity(original: np.ndarray, thresholds) -> float:
    """Feature Uniformity metric U across the K+1 thresholded regions.

    U = 1 − (within-class variance) / (total variance)

    Ranges from 0 (poor segmentation) to 1 (perfect: all variance is
    between classes, none within). Equivalent to σ²_between / σ²_total.
    """
    t = sanitise_thresholds(thresholds)
    edges = [-1, *t, 255]

    img = original.astype(np.float64)
    total_var = float(np.var(img))
    if total_var < 1e-12:
        return 1.0  # constant image is perfectly uniform

    N = img.size
    within_var = 0.0
    for c in range(len(t) + 1):
        lo = edges[c] + 1
        hi = edges[c + 1]
        mask = (img >= lo) & (img <= hi)
        n_c = int(np.sum(mask))
        if n_c > 0:
            within_var += n_c * float(np.var(img[mask]))

    within_var /= N
    return float(1.0 - within_var / total_var)


def jaccard(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Jaccard index (IoU) between predicted and ground-truth masks."""
    raise NotImplementedError("metrics.jaccard -- Assignment section 2 (Phase 2, Dev B)")


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient between predicted and ground-truth masks."""
    raise NotImplementedError("metrics.dice -- Assignment section 2 (Phase 2, Dev B)")


__all__ = ["psnr", "ssim", "uniformity", "jaccard", "dice"]
