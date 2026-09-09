"""Kapur information-entropy objective (Assignment 1.2, f_Kapur).

Treats each class as an independent probability source and maximises the total
Shannon entropy across all K+1 classes. The optimum thresholds are those that
make the class-conditional intensity distributions as uniform (informative) as
possible.

    H = sum_k  ( - sum_{i in C_k} (p_i / omega_k) * ln(p_i / omega_k) )
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import EPSILON
from ._base import class_masses


def kapur(hist: np.ndarray, thresholds: Sequence[float]) -> float:
    """Sum of per-class Shannon entropies (maximise)."""
    total = 0.0
    for cls in class_masses(hist, thresholds):
        omega = float(cls.sum())
        if omega < EPSILON:
            continue
        p = cls[cls > EPSILON] / omega
        total += float(-np.sum(p * np.log(p)))
    return total
