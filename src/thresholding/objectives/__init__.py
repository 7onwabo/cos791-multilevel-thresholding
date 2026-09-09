"""Objective functions for multilevel thresholding (Assignment 1.2).

All three share one contract so optimizers and the experiment runner stay
objective-agnostic::

    fitness = objective(hist, thresholds)   # higher is better (maximise)

``hist``       -- normalised 256-bin histogram (see thresholding.histogram)
``thresholds`` -- length-K sequence of threshold values (continuous or int; the
                  objective rounds + sorts internally via _base.sanitise_thresholds)

Look objectives up by name through the ``OBJECTIVES`` registry.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .kapur import kapur
from .otsu import otsu
from .tsallis import tsallis

Objective = Callable[[np.ndarray, Sequence[float]], float]

OBJECTIVES: dict[str, Objective] = {
    "otsu": otsu,
    "kapur": kapur,
    "tsallis": tsallis,
}

__all__ = ["OBJECTIVES", "Objective", "otsu", "kapur", "tsallis"]
