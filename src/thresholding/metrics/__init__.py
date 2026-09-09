"""Evaluation metrics (Assignment section 2) -- STUBS.

Phase 1 (BDS500) reconstruction quality:
    psnr, ssim, uniformity
Phase 2 (CHAOS MRI) segmentation overlap vs ground truth:
    jaccard, dice

Implemented later; signatures are fixed now so the runner/report code can be
written against them. ``ssim`` will wrap ``skimage.metrics.structural_similarity``.
"""

from __future__ import annotations

import numpy as np


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


def jaccard(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Jaccard index (IoU) between predicted and ground-truth masks."""
    raise NotImplementedError("metrics.jaccard -- Assignment section 2 (Phase 2)")


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient between predicted and ground-truth masks."""
    raise NotImplementedError("metrics.dice -- Assignment section 2 (Phase 2)")


__all__ = ["psnr", "ssim", "uniformity", "jaccard", "dice"]
