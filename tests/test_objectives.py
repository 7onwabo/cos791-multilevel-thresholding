"""Correctness tests for the Assignment 1.2 objective functions."""

from __future__ import annotations

import numpy as np
import pytest

from thresholding.config import GRAY_LEVELS
from thresholding.objectives import OBJECTIVES, kapur, otsu, tsallis
from thresholding.objectives._base import class_masses, sanitise_thresholds


def bimodal_hist(peak_a: int = 60, peak_b: int = 190, spread: float = 8.0) -> np.ndarray:
    """Clean two-lobe histogram with a valley between the peaks."""
    levels = np.arange(GRAY_LEVELS)
    hist = (
        np.exp(-0.5 * ((levels - peak_a) / spread) ** 2)
        + np.exp(-0.5 * ((levels - peak_b) / spread) ** 2)
    )
    return hist / hist.sum()


def test_registry_has_three_objectives():
    assert set(OBJECTIVES) == {"otsu", "kapur", "tsallis"}


def test_sanitise_rounds_clips_and_sorts():
    assert sanitise_thresholds([250.6, -3, 10.2, 300]) == [0, 10, 251, 255]


def test_class_masses_partition_is_complete():
    hist = bimodal_hist()
    masses = class_masses(hist, [80, 150])
    assert len(masses) == 3  # K+1 classes for K=2 thresholds
    assert np.isclose(sum(float(m.sum()) for m in masses), 1.0)


def test_otsu_prefers_the_valley():
    """Between-class variance should be maximal near the histogram valley (~125)."""
    hist = bimodal_hist(60, 190)
    candidates = range(70, 181, 5)
    best = max(candidates, key=lambda t: otsu(hist, [t]))
    assert 110 <= best <= 140


@pytest.mark.parametrize("objective", [otsu, kapur, tsallis])
def test_no_nan_or_inf_on_random_thresholds(objective):
    rng = np.random.default_rng(0)
    hist = bimodal_hist()
    for _ in range(50):
        k = int(rng.integers(3, 13))
        thresholds = rng.integers(0, GRAY_LEVELS, size=k)
        value = objective(hist, thresholds)
        assert np.isfinite(value)


@pytest.mark.parametrize("objective", [otsu, kapur, tsallis])
def test_degenerate_duplicate_thresholds_are_safe(objective):
    """Coincident thresholds create empty classes -- must not raise or NaN."""
    hist = bimodal_hist()
    assert np.isfinite(objective(hist, [100, 100, 100]))


def test_kapur_entropy_non_negative():
    hist = bimodal_hist()
    assert kapur(hist, [125]) >= 0.0
