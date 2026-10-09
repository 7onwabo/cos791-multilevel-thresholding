"""JADE -- adaptive DE with optional external archive and DE/current-to-pbest/1
mutation (Assignment 1.3).

Adapts F and CR from successful values each generation using an exponential
moving average:
    mu_CR ← (1-c)*mu_CR + c * mean_A(S_CR)      (arithmetic mean)
    mu_F  ← (1-c)*mu_F  + c * mean_L(S_F)        (Lehmer mean)

Owner: Dev A.

References
----------
Zhang, J., & Sanderson, A. C. (2009). JADE: Adaptive Differential Evolution
with Optional External Archive. IEEE Transactions on Evolutionary Computation,
13(5), 945--958.  DOI: 10.1109/TEVC.2009.2014613
"""

from __future__ import annotations

import numpy as np

from .base import BaseOptimizer, OptResult


class JADE(BaseOptimizer):
    """JADE with DE/current-to-pbest/1 mutation and external archive.

    Parameters
    ----------
    c : float
        Adaptation rate for mu_CR and mu_F.  Default 0.1.
    p : float
        Fraction of population used as "pbest" pool.  Default 0.1.
    """

    name = "JADE"

    def __init__(self, *args, c: float = 0.1, p: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.p = p

    def optimize(self) -> OptResult:  # noqa: D401
        N, D = self.pop_size, self.dim
        c, p = self.c, self.p
        mu_CR = 0.5
        mu_F = 0.5

        # --- Initialise population uniformly within bounds --------------------
        pop = self.lower + self.rng.random((N, D)) * (self.upper - self.lower)
        fit = np.array([self.fitness(x) for x in pop])
        evals = N

        archive: list[np.ndarray] = []
        history = [float(fit.max())]
        evals_history = [evals]

        # --- Main loop --------------------------------------------------------
        while evals < self.max_fes:
            # Generate per-individual CR_i ~ N(mu_CR, 0.1) and F_i ~ Cauchy(mu_F, 0.1)
            CR = np.clip(self.rng.normal(mu_CR, 0.1, N), 0.0, 1.0)

            F_vals = mu_F + 0.1 * self.rng.standard_cauchy(N)
            while np.any(F_vals <= 0):
                bad = F_vals <= 0
                F_vals[bad] = mu_F + 0.1 * self.rng.standard_cauchy(int(bad.sum()))
            F_vals = np.minimum(F_vals, 1.0)

            order = np.argsort(-fit)  # best first
            new_pop, new_fit = pop.copy(), fit.copy()
            S_CR: list[float] = []
            S_F: list[float] = []

            union = np.vstack([pop] + ([np.array(archive)] if archive else []))

            for i in range(N):
                if evals >= self.max_fes:
                    break

                # Select random pbest from top-p fraction
                p_num = max(2, int(round(p * N)))
                pbest = pop[order[self.rng.integers(0, p_num)]]

                # r1 from current population, != i
                r1 = self.rng.integers(0, N - 1)
                r1 += r1 >= i

                # r2 from pop ∪ archive, != i and != r1
                while True:
                    r2 = self.rng.integers(0, len(union))
                    if r2 != i and r2 != r1:
                        break

                # Mutation: DE/current-to-pbest/1
                v = pop[i] + F_vals[i] * (pbest - pop[i]) + F_vals[i] * (pop[r1] - union[r2])

                # Bound handling (midpoint bounce -- matches SHADE / L-SHADE)
                lo = v < self.lower
                hi = v > self.upper
                v[lo] = (self.lower[lo] + pop[i][lo]) / 2
                v[hi] = (self.upper[hi] + pop[i][hi]) / 2

                # Binomial crossover
                mask = self.rng.random(D) < CR[i]
                mask[self.rng.integers(0, D)] = True
                u = np.where(mask, v, pop[i])

                # Greedy selection (maximise)
                fu = self.fitness(u)
                evals += 1
                if fu >= fit[i]:
                    if fu > fit[i]:
                        # Record successful parameters (strict improvement only)
                        S_CR.append(CR[i])
                        S_F.append(F_vals[i])
                        archive.append(pop[i].copy())
                    new_pop[i], new_fit[i] = u, fu

            pop, fit = new_pop, new_fit

            # Trim archive to population size
            while len(archive) > N:
                archive.pop(self.rng.integers(0, len(archive)))

            # Parameter adaptation (unweighted -- per JADE paper, not SHADE)
            if S_CR:
                S_CR_arr = np.array(S_CR)
                S_F_arr = np.array(S_F)
                mu_CR = (1 - c) * mu_CR + c * float(np.mean(S_CR_arr))
                mu_F = (1 - c) * mu_F + c * float(np.sum(S_F_arr ** 2) / np.sum(S_F_arr))

            history.append(float(fit.max()))
            evals_history.append(evals)

        best = int(np.argmax(fit))
        return OptResult(pop[best].copy(), float(fit[best]), history, evals, evals_history)
