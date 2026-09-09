"""LADE -- Late Acceptance Differential Evolution (Assignment 1.3).

Replaces greedy selection with a Late Acceptance criterion (adopted from Late
Acceptance Hill Climbing): a trial is accepted if it beats the fitness recorded
L generations ago. Owner: Dev C.
"""

from __future__ import annotations

from .base import BaseOptimizer, OptResult


class LateAcceptanceDE(BaseOptimizer):
    name = "LADE"

    def optimize(self) -> OptResult:  # noqa: D401
        raise NotImplementedError("LateAcceptanceDE.optimize -- Assignment 1.3 (Dev C)")
