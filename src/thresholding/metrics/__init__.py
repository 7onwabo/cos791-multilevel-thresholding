"""Image reconstruction and unsupervised segmentation metrics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as _sk_ssim


def _segment(image: np.ndarray, thresholds) -> np.ndarray:
    """Map each pixel to its class mean intensity (reconstructed image)."""
    img = np.asanyarray(image, dtype=np.float64)
    t = sorted(int(round(x)) for x in thresholds)

    lo = float(img.min())
    hi = float(img.max())
    edges = [lo - 1.0] + [float(x) for x in t] + [hi]

    class_map = np.zeros(img.shape, dtype=np.int32)
    reconstructed = np.zeros(img.shape, dtype=np.float64)

    for k in range(len(edges) - 1):
        low, high = edges[k], edges[k + 1]
        mask = (img <= high) if k == 0 else ((img > low) & (img <= high))
        if not np.any(mask):
            continue

        class_map[mask] = k
        reconstructed[mask] = img[mask].mean()

    return reconstructed, class_map

def psnr(original: np.ndarray, thresholds, data_range: float | None = None) -> float:
    """Peak Signal-to-Noise Ratio between original and thresholded reconstruction."""
    img = np.asarray(original, dtype=np.float64)
    reconstructed, _ = _segment(img, thresholds)

    mse = np.mean((img - reconstructed) ** 2)
    if mse == 0:
        return float("inf")

    if data_range is None:
        data_range = img.max() - img.min()
        if data_range == 0:
            data_range = 1.0

    return 10.0 * np.log10((data_range ** 2) / mse)

def ssim(original: np.ndarray, thresholds, data_range: float | None = None) -> float:
    """Structural Similarity Index (wrap skimage.metrics.structural_similarity)."""
    img = np.asarray(original, dtype=np.float64)
    reconstructed, _ = _segment(img, thresholds)

    if data_range is None:
        data_range = img.max() - img.min()
        if data_range == 0:
            data_range = 1.0

    return float(_sk_ssim(img, reconstructed, data_range=data_range))


def uniformity(original: np.ndarray, thresholds) -> float:
    """Feature Uniformity metric U across the K+1 thresholded regions."""
    img = np.asarray(original, dtype=np.float64)
    _, class_map = _segment(img, thresholds)

    n = img.size
    intensity_rate_sq = (img.max() - img.min()) ** 2
    if intensity_rate_sq == 0:
        return 1.0

    total_within_class_variance = 0.0
    for c in np.unique(class_map):
        region = img[class_map == c]
        mu = region.mean()
        total_within_class_variance += np.sum((region - mu) ** 2)

    u = 1.0 - (2.0 * total_within_class_variance) / (n * intensity_rate_sq)
    return float(u)

def class_separability(original: np.ndarray, thresholds) -> float:
    """Return weighted between-class variance divided by total image variance.

    The score is 0 when threshold classes have identical means and approaches
    1 as the classes explain more of the image's intensity variance.
    """
    img = np.asarray(original, dtype=np.float64)
    _, class_map = _segment(img, thresholds)
    total_variance = float(np.var(img))
    if total_variance == 0.0:
        return 0.0

    global_mean = float(np.mean(img))
    between_class_variance = 0.0
    n = img.size
    for c in np.unique(class_map):
        region = img[class_map == c]
        between_class_variance += (region.size / n) * (float(region.mean()) - global_mean) ** 2

    return float(between_class_variance / total_variance)


__all__ = ["psnr", "ssim", "uniformity", "class_separability"]
