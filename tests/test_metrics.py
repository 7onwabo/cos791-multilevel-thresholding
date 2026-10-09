"""Correctness tests for the Phase-1 reconstruction metrics (Dev A).

Tests cover PSNR, SSIM, and Feature Uniformity (U).  Jaccard/Dice are Dev B
stubs and are not tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from thresholding.metrics import psnr, ssim, uniformity, _segment


# ---------- helpers -----------------------------------------------------------
def _ramp_image(rows: int = 64, cols: int = 64) -> np.ndarray:
    """Gradient image 0..255 (uint8), good for deterministic metric checks."""
    return np.tile(np.linspace(0, 255, cols, dtype=np.uint8), (rows, 1))


def _constant_image(val: int = 128, rows: int = 64, cols: int = 64) -> np.ndarray:
    return np.full((rows, cols), val, dtype=np.uint8)


def _random_image(seed: int = 0, rows: int = 64, cols: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (rows, cols), dtype=np.uint8)


# ---------- _segment ----------------------------------------------------------
def test_segment_returns_correct_shape():
    img = _ramp_image()
    out, _ = _segment(img, [64, 128, 192])
    assert out.shape == img.shape


def test_segment_values_are_class_means():
    """Each pixel in the reconstructed image should equal its class mean."""
    img = _ramp_image()
    thresholds = [100, 200]
    out, _ = _segment(img, thresholds)

    # Class 0: [0, 100], Class 1: [101, 200], Class 2: [201, 255]
    for lo, hi in [(0, 100), (101, 200), (201, 255)]:
        mask = (img >= lo) & (img <= hi)
        if np.any(mask):
            expected_mean = np.mean(img[mask].astype(np.float64))
            np.testing.assert_allclose(out[mask], expected_mean, atol=1e-10)


def test_segment_single_threshold():
    img = _ramp_image()
    out, _ = _segment(img, [128])
    # Should have exactly two distinct reconstructed values
    unique = np.unique(out)
    assert len(unique) == 2


# ---------- PSNR --------------------------------------------------------------
def test_psnr_identical_image_is_high():
    """Identical reconstruction should give very high (or infinite) PSNR."""
    img = _ramp_image()
    # Use thresholds that exactly match class boundaries so reconstruction Γëê original
    # With 256 thresholds we'd get perfect reconstruction, but even a few should be high
    val = psnr(img, [64, 128, 192])
    assert val > 20.0, f"PSNR too low: {val}"


def test_psnr_constant_image():
    """Constant image thresholded at any point has zero error -> infinite PSNR."""
    img = _constant_image(100)
    val = psnr(img, [50, 150])
    assert val == float("inf") or val > 300, f"Expected very high PSNR for constant image, got {val}"


def test_psnr_is_positive():
    img = _random_image()
    val = psnr(img, [64, 128, 192])
    assert val > 0, f"PSNR should be positive, got {val}"


def test_psnr_more_thresholds_is_better():
    """More thresholds -> finer reconstruction -> higher PSNR (usually)."""
    img = _ramp_image()
    p3 = psnr(img, [85, 170])
    p7 = psnr(img, [32, 64, 96, 128, 160, 192, 224])
    assert p7 > p3, f"7 thresholds ({p7:.1f}) should beat 2 ({p3:.1f}) on a gradient"


# ---------- SSIM --------------------------------------------------------------
def test_ssim_constant_image():
    """Constant image thresholded at any point has perfect similarity."""
    img = _constant_image(100)
    val = ssim(img, [50, 150])
    assert val == pytest.approx(1.0, abs=1e-5), f"Expected SSIM Γëê 1.0, got {val}"


def test_ssim_in_valid_range():
    img = _random_image()
    val = ssim(img, [64, 128, 192])
    assert -1.0 <= val <= 1.0, f"SSIM out of range: {val}"


def test_ssim_more_thresholds_is_better():
    img = _ramp_image()
    s2 = ssim(img, [85, 170])
    s7 = ssim(img, [32, 64, 96, 128, 160, 192, 224])
    assert s7 > s2, f"7 thresholds ({s7:.4f}) should beat 2 ({s2:.4f}) on a gradient"


# ---------- Uniformity --------------------------------------------------------
def test_uniformity_constant_image():
    """Constant image should have U = 1.0 (zero within-class variance)."""
    img = _constant_image(100)
    val = uniformity(img, [50, 150])
    assert val == pytest.approx(1.0, abs=1e-10)


def test_uniformity_in_zero_one():
    img = _random_image()
    val = uniformity(img, [64, 128, 192])
    assert 0.0 <= val <= 1.0, f"Uniformity out of range: {val}"


def test_uniformity_more_thresholds_is_better():
    img = _ramp_image()
    u2 = uniformity(img, [85, 170])
    u7 = uniformity(img, [32, 64, 96, 128, 160, 192, 224])
    assert u7 > u2, f"7 thresholds ({u7:.4f}) should beat 2 ({u2:.4f}) on a gradient"


def test_uniformity_scales_with_number_of_thresholds():
    """Sahoo's U weights within-class scatter by c = number of thresholds."""
    img = np.array([[0, 10], [100, 110]], dtype=np.uint8)
    # c=1: classes {0,10},{100,110}; scatter 100.  c=2: {0,10},{100},{110}; scatter 50.
    expected = 1.0 - 200.0 / (4 * 110 ** 2)
    assert uniformity(img, [50]) == pytest.approx(expected)
    assert uniformity(img, [50, 105]) == pytest.approx(expected)


def test_uniformity_perfect_split():
    """Two-class image with threshold at class boundary -> U Γëê 1.0."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:32, :] = 50
    img[32:, :] = 200
    val = uniformity(img, [125])
    assert val == pytest.approx(1.0, abs=1e-10)
