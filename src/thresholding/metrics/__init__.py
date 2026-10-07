"""Evaluation metrics (Assignment section 2) -- STUBS.

Phase 1 (BDS500) reconstruction quality:
    psnr, ssim, uniformity
Phase 2 (CHAOS MRI) unsupervised segmentation quality:
    class_separability  (eta = sigma_B^2 / sigma_T^2)

The brief's updated Experiment 2 drops the ground-truth overlap metrics in
favour of eta; ``jaccard``/``dice`` are kept only in case the lecturer restores
them.

Implemented later; signatures are fixed now so the runner/report code can be
written against them. ``ssim`` will wrap ``skimage.metrics.structural_similarity``.
"""

from __future__ import annotations

import numpy as np

from ..config import EPSILON
from ..histogram import normalised_histogram
from ..objectives.otsu import otsu


def _segment(image: np.ndarray, thresholds) -> np.ndarray:
    """Map each pixel to its class mean intensity (reconstructed image)."""
    raise NotImplementedError("metrics._segment -- Assignment section 2")


def psnr(original: np.ndarray, thresholds) -> float:
    """Peak Signal-to-Noise Ratio between original and thresholded reconstruction."""
    raise NotImplementedError("metrics.psnr -- Assignment section 2")


def ssim(original: np.ndarray, thresholds) -> float:
    """Structural Similarity Index (wrap skimage.metrics.structural_similarity)."""
    raise NotImplementedError("metrics.ssim -- Assignment section 2")


def uniformity(original: np.ndarray, thresholds) -> float:
    """Feature Uniformity metric U across the K+1 thresholded regions."""
    raise NotImplementedError("metrics.uniformity -- Assignment section 2")


def class_separability(image: np.ndarray, thresholds) -> float:
    """Class Separability eta = sigma_B^2 / sigma_T^2 (Experiment 2, CHAOS MRI).

    How distinctly the thresholded intensity classes are separated, as a
    fraction of the image's total intensity variance. Unsupervised -- no ground
    truth needed, which is why the brief uses it for CHAOS. Range [0, 1]:
    eta -> 1 means the thresholds explain all of the image's variance.

    sigma_B^2 is the between-class variance, i.e. exactly the Otsu objective, so
    this reuses ``objectives.otsu`` rather than recomputing it.
    """
    hist = normalised_histogram(image)
    levels = np.arange(hist.shape[0])
    mean = float(np.dot(levels, hist))
    total_var = float(np.dot((levels - mean) ** 2, hist))
    if total_var < EPSILON:
        return 0.0                      # flat image: nothing to separate
    return otsu(hist, thresholds) / total_var


def jaccard(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Jaccard index (IoU) between predicted and ground-truth masks."""
    raise NotImplementedError("metrics.jaccard -- Assignment section 2 (Phase 2)")


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient between predicted and ground-truth masks."""
    raise NotImplementedError("metrics.dice -- Assignment section 2 (Phase 2)")


__all__ = ["psnr", "ssim", "uniformity", "class_separability", "jaccard", "dice"]
