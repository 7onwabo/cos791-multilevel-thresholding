"""LADE -- Late Acceptance Differential Evolution (Assignment 1.3)."""

from __future__ import annotations

import numpy as np

from .base import BaseOptimizer, OptResult


class LateAcceptanceDE(BaseOptimizer):
    name = "LADE"

    def __init__(
        self,
        *args,
        L: int = 10,
        CR: float = 0.9,
        F_lo: float = 0.5,
        F_hi: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if kwargs.get("pop_size") is None:
            # Late acceptance already supplies the diversity that a large
            # population normally buys, so a small population -- which gets far
            # more generations out of the same FE budget -- searches better.
            # Swept over N in {8..20} x L in {5..50} on sphere/Rastrigin (D=10)
            # and on Otsu/Kapur/Tsallis for K in {3,5,12}: N ~ D, L ~ 10 ranked
            # best overall, and large N (>= 16) clearly lost at K = 12.
            self.pop_size = max(10, self.dim)
        if L < 1:
            raise ValueError("L (late-acceptance memory length) must be >= 1")
        if self.pop_size < 4:
            raise ValueError("DE/rand/1 needs pop_size >= 4")
        self.L = L
        self.CR = CR
        self.F_lo = F_lo
        self.F_hi = F_hi

    def _distinct_indices(self, N: int, count: int) -> np.ndarray:
        """``(N, count)`` matrix of mutually distinct indices, none equal to the row.

        Random keys + partial sort; cheaper to write (and to reason about) than
        rejection sampling, and vectorised over the whole population.
        """
        keys = self.rng.random((N, N))
        keys[np.arange(N), np.arange(N)] = np.inf     # never pick self
        return np.argpartition(keys, count - 1, axis=1)[:, :count]

    def optimize(self) -> OptResult:  # noqa: D401
        N, D = self.pop_size, self.dim
        span = self.upper - self.lower

        # --- Initialisation -------------------------------------------------
        pop = self.lower + self.rng.random((N, D)) * span
        fit = np.empty(N)
        evals = 0
        for i in range(N):
            fit[i] = self.fitness(pop[i])
            evals += 1

        # Late-acceptance memory: L past fitness values per individual, seeded
        # with the individual's own starting fitness.
        mem = np.repeat(fit[:, None], self.L, axis=1)

        b = int(np.argmax(fit))
        best_x, best_f = pop[b].copy(), float(fit[b])
        history = [best_f]
        evals_history = [evals]
        self.trace: list[tuple[int, int]] = []   # (evals, accepted this generation)

        g = 0
        while evals < self.max_fes:
            v = g % self.L

            # --- Mutation: DE/rand/1 with dither --------------------------
            r = self._distinct_indices(N, 3)
            F = self.rng.uniform(self.F_lo, self.F_hi, (N, 1))
            mutant = pop[r[:, 0]] + F * (pop[r[:, 1]] - pop[r[:, 2]])

            # Bound handling: midpoint between the violated bound and the parent.
            lo, hi = mutant < self.lower, mutant > self.upper
            parent_lo = np.broadcast_to(self.lower, (N, D))
            parent_hi = np.broadcast_to(self.upper, (N, D))
            mutant = np.where(lo, (parent_lo + pop) / 2, mutant)
            mutant = np.where(hi, (parent_hi + pop) / 2, mutant)

            # --- Binomial crossover ---------------------------------------
            mask = self.rng.random((N, D)) < self.CR
            mask[np.arange(N), self.rng.integers(0, D, N)] = True
            trial = np.where(mask, mutant, pop)

            # --- Evaluate + late-acceptance selection ---------------------
            accepted = 0
            for i in range(N):
                if evals >= self.max_fes:
                    break

                fu = self.fitness(trial[i])
                evals += 1

                if fu >= fit[i] or fu >= mem[i, v]:
                    pop[i], fit[i] = trial[i], fu
                    accepted += 1
                    if fu > best_f:
                        best_f, best_x = float(fu), trial[i].copy()

                # Burke & Bykov: the memory slot only ever improves.
                if fit[i] > mem[i, v]:
                    mem[i, v] = fit[i]

            history.append(best_f)
            evals_history.append(evals)
            self.trace.append((evals, accepted))
            g += 1

        return OptResult(best_x, best_f, history, evals, evals_history)
