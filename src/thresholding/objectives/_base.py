"""Shared machinery for the three objective functions (Assignment 1.2).

Every objective maximises a fitness computed from the normalised histogram and a
threshold vector. They all need the same first step -- carve the histogram into
K+1 contiguous intensity classes and get each class's probability mass -- so that
lives here once.

Class convention: for sorted thresholds t_1 < ... < t_K the image is split into
    C_0 = [0, t_1],  C_1 = (t_1, t_2],  ...,  C_K = (t_K, L-1]
i.e. class c covers intensity indices ``edges[c] + 1 .. edges[c+1]`` with
``edges = [-1, t_1, ..., t_K, L-1]``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import GRAY_LEVELS


def sanitise_thresholds(thresholds: Sequence[float], levels: int = GRAY_LEVELS) -> list[int]:
    """Round, clip to [0, levels-1] and sort a candidate threshold vector.

    DE proposes continuous, unordered vectors; objectives need integer, sorted
    thresholds. Duplicates are kept (they simply yield an empty class, which the
    objectives handle via their epsilon guards) so the vector length -- and hence
    the number of classes -- is preserved.
    """
    t = np.clip(np.rint(np.asarray(thresholds, dtype=float)), 0, levels - 1)
    return sorted(int(x) for x in t)


def class_masses(hist: np.ndarray, thresholds: Sequence[float]) -> list[np.ndarray]:
    """Split ``hist`` into K+1 per-class probability slices.

    Returns a list of length ``len(thresholds) + 1`` where element c is the slice
    of ``hist`` belonging to class c. Slices may be empty when thresholds coincide.
    """
    levels = hist.shape[0]
    t = sanitise_thresholds(thresholds, levels)
    edges = [-1, *t, levels - 1]
    return [hist[edges[c] + 1 : edges[c + 1] + 1] for c in range(len(t) + 1)]
