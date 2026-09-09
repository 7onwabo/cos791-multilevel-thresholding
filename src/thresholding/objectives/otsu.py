"""Otsu between-class variance objective (Assignment 1.2, f_Otsu).

Maximising the weighted between-class variance is equivalent to minimising the
within-class variance, giving the classic Otsu criterion generalised to K
thresholds. Fitness is higher when the chosen thresholds separate the intensity
classes most cleanly.

    sigma_B^2 = sum_k  omega_k * (mu_k - mu_T)^2

where omega_k is the probability mass of class k, mu_k its mean intensity and
mu_T the global mean intensity.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import EPSILON
from ._base import class_masses


def otsu(hist: np.ndarray, thresholds: Sequence[float]) -> float:
    """Between-class variance for ``thresholds`` on histogram ``hist`` (maximise)."""
    levels = np.arange(hist.shape[0])
    global_mean = float(np.dot(levels, hist))

    variance = 0.0
    offset = 0
    for cls in class_masses(hist, thresholds):
        omega = float(cls.sum())
        if omega < EPSILON:
            offset += cls.shape[0]
            continue
        class_levels = levels[offset : offset + cls.shape[0]]
        mean = float(np.dot(class_levels, cls) / omega)
        variance += omega * (mean - global_mean) ** 2
        offset += cls.shape[0]
    return variance
