"""Common interface every DE variant implements (Assignment 1.3).

Keeping one base class means the experiment runner can swap optimizers without
caring which variant it is. Each team member subclasses ``BaseOptimizer`` and
implements ``optimize`` for their assigned algorithm(s).

Contract:
  * The optimizer MAXIMISES ``fitness`` (all objectives in this project return a
    "higher is better" score).
  * A candidate solution is a length-K real vector within ``bounds``; the
    objective sanitises it (round + sort) so the optimizer may work in continuous
    space and need not enforce integrality/ordering itself.
  * Termination is by function-evaluation budget (``max_fes``) for a fair
    cross-algorithm comparison (Assignment section 3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..config import MAX_FES, RANDOM_SEED


@dataclass
class OptResult:
    """Outcome of a single optimisation run."""

    best_solution: np.ndarray            # best threshold vector found (length K)
    best_fitness: float                  # its objective value
    history: list[float] = field(default_factory=list)  # best-so-far per generation
    n_evals: int = 0                     # function evaluations consumed


class BaseOptimizer(ABC):
    """Base class for Differential Evolution variants.

    Parameters
    ----------
    fitness : callable(np.ndarray) -> float
        Wraps ``objective(hist, x)`` for a fixed histogram; higher is better.
    bounds : (lower, upper)
        Per-dimension search bounds (see ``config.threshold_bounds``).
    max_fes, seed, pop_size : experiment budget and reproducibility knobs.
    """

    name: str = "base"

    def __init__(
        self,
        fitness: Callable[[np.ndarray], float],
        bounds: tuple[Sequence[float], Sequence[float]],
        *,
        max_fes: int = MAX_FES,
        seed: int = RANDOM_SEED,
        pop_size: int | None = None,
    ) -> None:
        self.fitness = fitness
        self.lower = np.asarray(bounds[0], dtype=float)
        self.upper = np.asarray(bounds[1], dtype=float)
        self.dim = self.lower.shape[0]
        self.max_fes = max_fes
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size if pop_size is not None else 10 * self.dim

    @abstractmethod
    def optimize(self) -> OptResult:
        """Run the search and return the best threshold vector found."""
        raise NotImplementedError
