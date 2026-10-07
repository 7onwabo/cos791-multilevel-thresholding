"""Sanity tests for the LADE (Late Acceptance DE) optimizer."""

import numpy as np
import pytest
from thresholding.optimizers.lade import LateAcceptanceDE

DIM = 10
MAX_FES = 20000


def sphere(x):
    return -float(np.sum(x ** 2))


def make(fitness=sphere, seed=1, **kw):
    lo, hi = -5 * np.ones(DIM), 5 * np.ones(DIM)
    return LateAcceptanceDE(fitness, (lo, hi), max_fes=MAX_FES, seed=seed, **kw)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_lade(seed):
    res = make(seed=seed).optimize()

    assert res.n_evals <= MAX_FES, f"Exceeded max evaluations: {res.n_evals} > {MAX_FES}"
    is_monotonic = all(b >= a for a, b in zip(res.history, res.history[1:]))
    assert is_monotonic, "Fitness history is not monotonic"
    assert res.best_fitness > -1e-3, f"Failed to converge. Best fitness was {res.best_fitness}"


def test_lade_eval_count_matches_calls():
    calls = 0

    def counted(x):
        nonlocal calls
        calls += 1
        return sphere(x)

    res = make(fitness=counted).optimize()
    assert calls == res.n_evals, f"n_evals={res.n_evals} but fitness called {calls} times"
    assert calls <= MAX_FES


def test_lade_candidates_stay_in_bounds():
    def checked(x):
        assert np.all(x >= -5) and np.all(x <= 5), f"Out-of-bounds candidate: {x}"
        return sphere(x)

    make(fitness=checked).optimize()


def test_lade_reported_best_matches_evaluated_best():
    seen = []

    def recorded(x):
        f = sphere(x)
        seen.append(f)
        return f

    res = make(fitness=recorded).optimize()
    assert res.best_fitness == pytest.approx(max(seen))
    assert sphere(res.best_solution) == pytest.approx(res.best_fitness)


def test_lade_reproducible():
    a = make(seed=7).optimize()
    b = make(seed=7).optimize()
    assert a.best_fitness == b.best_fitness
    assert np.array_equal(a.best_solution, b.best_solution)


def test_lade_accepts_worsening_moves_early():
    """Late acceptance must be non-greedy, otherwise it is just DE/rand/1/bin."""
    opt = make(L=20)
    opt.optimize()
    assert sum(a for _, a in opt.trace) > 0, "No trial was ever accepted"

    greedy = make(L=1)          # L=1 collapses the memory onto the current value
    greedy.optimize()
    assert sum(a for _, a in opt.trace) > sum(a for _, a in greedy.trace), \
        "Longer memory should accept strictly more moves than the near-greedy case"


def test_lade_rejects_bad_memory_length():
    with pytest.raises(ValueError):
        make(L=0)


# --- Experiment 2 (CHAOS MRI): Class Separability eta -------------------------

def test_lade_maximises_class_separability_on_chaos():
    """LADE must drive eta up on a real MRI slice, and more thresholds must help.

    eta = sigma_B^2 / sigma_T^2 is bounded by 1 and is monotone non-decreasing in
    K, so a working optimizer should show both.
    """
    from thresholding.config import threshold_bounds
    from thresholding.datasets import load_chaos
    from thresholding.histogram import normalised_histogram
    from thresholding.metrics import class_separability
    from thresholding.objectives import OBJECTIVES

    image = sorted(load_chaos().items())[0][1]
    hist = normalised_histogram(image)

    etas = []
    for k in (3, 5, 7):
        res = LateAcceptanceDE(
            lambda x: OBJECTIVES["otsu"](hist, x),
            threshold_bounds(k), max_fes=5000, seed=42,
        ).optimize()
        eta = class_separability(image, res.best_solution)
        assert 0.0 <= eta <= 1.0, f"eta out of range at K={k}: {eta}"
        etas.append(eta)

    assert etas[0] > 0.9, f"eta too low even at K=3: {etas[0]}"
    assert all(b >= a - 1e-9 for a, b in zip(etas, etas[1:])), \
        f"eta must not drop as K grows: {etas}"
