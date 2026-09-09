"""L-SHADE -- SHADE with Linear Population Size Reduction over the FE budget
(Assignment 1.3). Owner: Dev B.
"""

from __future__ import annotations

from .base import BaseOptimizer, OptResult


class LSHADE(BaseOptimizer):
    name = "L-SHADE"

    def optimize(self) -> OptResult:  # noqa: D401
        raise NotImplementedError("LSHADE.optimize -- Assignment 1.3 (Dev B)")
