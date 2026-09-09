"""Tsallis non-extensive entropy objective (Assignment 1.2, f_Tsallis).

Generalises Kapur's entropy with a non-extensivity parameter q. Individual class
entropies combine via the pseudo-additive rule rather than simple summation:

    S_q^k = ( 1 - sum_{i in C_k} (p_i / omega_k)^q ) / (q - 1)

    S_q = sum_k S_q^k + (1 - q) * prod_k S_q^k

As q -> 1 this reduces to Kapur's Shannon entropy. q is read from
``config.TSALLIS_Q`` (Table 1 reports q = 0.8).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import EPSILON, TSALLIS_Q
from ._base import class_masses


def tsallis(hist: np.ndarray, thresholds: Sequence[float], q: float = TSALLIS_Q) -> float:
    """Tsallis non-extensive entropy with pseudo-additive coupling (maximise)."""
    if abs(q - 1.0) < EPSILON:  # degenerates to Shannon; avoid divide-by-zero
        q = 1.0 + EPSILON

    class_entropies: list[float] = []
    for cls in class_masses(hist, thresholds):
        omega = float(cls.sum())
        if omega < EPSILON:
            continue
        p = cls[cls > EPSILON] / omega
        s_q = (1.0 - float(np.sum(p ** q))) / (q - 1.0)
        class_entropies.append(s_q)

    if not class_entropies:
        return 0.0

    additive = float(np.sum(class_entropies))
    coupling = (1.0 - q) * float(np.prod(class_entropies))
    return additive + coupling
