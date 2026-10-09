"""Standard Differential Evolution -- DE/rand/1/bin (Assignment 1.3).

Baseline classical DE: rand/1 mutation + binomial crossover, fixed F and CR.
Owner: Dev A.

References
----------
Storn, R., & Price, K. (1997). Differential Evolution -- A Simple and
Efficient Heuristic for Global Optimization over Continuous Spaces.
Journal of Global Optimization, 11(4), 341--359.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOptimizer, OptResult


class StandardDE(BaseOptimizer):
    """Classical DE/rand/1/bin with fixed control parameters.

    Parameters
    ----------
    F : float
        Scale factor (mutation amplitude). Default 0.5.
    CR : float
        Crossover rate. Default 0.9.
    """

    name = "DE"

    def __init__(self, *args, F: float = 0.5, CR: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.F = F
        self.CR = CR

    def optimize(self) -> OptResult:  # noqa: D401
        N, D = self.pop_size, self.dim
        F, CR = self.F, self.CR

        # --- Initialise population uniformly within bounds ---------------------
        pop = self.lower + self.rng.random((N, D)) * (self.upper - self.lower)
        fit = np.array([self.fitness(x) for x in pop])
        evals = N
        history = [float(fit.max())]
        evals_history = [evals]

        # --- Main loop --------------------------------------------------------
        while evals < self.max_fes:
            new_pop, new_fit = pop.copy(), fit.copy()

            for i in range(N):
                if evals >= self.max_fes:
                    break

                # Select three mutually exclusive random indices != i
                candidates = [j for j in range(N) if j != i]
                r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)

                # Mutation: DE/rand/1
                v = pop[r1] + F * (pop[r2] - pop[r3])

                # Bound handling (midpoint bounce -- matches SHADE / L-SHADE)
                lo = v < self.lower
                hi = v > self.upper
                v[lo] = (self.lower[lo] + pop[i][lo]) / 2
                v[hi] = (self.upper[hi] + pop[i][hi]) / 2

                # Binomial crossover
                mask = self.rng.random(D) < CR
                mask[self.rng.integers(0, D)] = True  # ensure at least one donor dim
                u = np.where(mask, v, pop[i])

                # Greedy selection (maximise)
                fu = self.fitness(u)
                evals += 1
                if fu >= fit[i]:
                    new_pop[i], new_fit[i] = u, fu

            pop, fit = new_pop, new_fit
            history.append(float(fit.max()))
            evals_history.append(evals)

        best = int(np.argmax(fit))
        return OptResult(pop[best].copy(), float(fit[best]), history, evals, evals_history)
