"""Standard Differential Evolution -- DE/rand/1/bin (Assignment 1.3).

Baseline classical DE: rand/1 mutation + binomial crossover, fixed F and CR.
Owner: Dev A.
"""

from __future__ import annotations

from .base import BaseOptimizer, OptResult


class StandardDE(BaseOptimizer):
    name = "DE"

    def optimize(self) -> OptResult:  # noqa: D401
        raise NotImplementedError("StandardDE.optimize -- Assignment 1.3 (Dev A)")
