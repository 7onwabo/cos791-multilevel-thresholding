"""Sanity test for shade optimizer."""

import numpy as np
import pytest
from thresholding.optimizers.shade import SHADE

def test_shade():
    f = lambda x: -float(np.sum(x**2))
    lo, hi = -5 * np.ones(10), 5 * np.ones(10)
    res = SHADE(f, (lo, hi), max_fes=20000, seed=1).optimize()

    assert res.n_evals <= 20000, f"Exceeded max evaluations: {res.n_evals} > {20000}"
    is_monotonic = all(b >= a for a, b in zip(res.history, res.history[1:]))
    assert is_monotonic, "Fitness history is not monotonic"
    assert res.best_fitness > -1e-3, f"Failed to converge. Best fitness was {res.best_fitness}"