"""Sanity tests for the L-SHADE optimizer."""

import numpy as np
import pytest
from thresholding.optimizers.lshade import LSHADE

DIM = 10
MAX_FES = 20000

def sphere(x):
    return -float(np.sum(x ** 2))


def make(fitness=sphere, seed=1, **kw):
    lo, hi = -5 * np.ones(DIM), 5 * np.ones(DIM)
    return LSHADE(fitness, (lo, hi), max_fes=MAX_FES, seed=seed, **kw)

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_lshade(seed):
    res = make(seed=seed).optimize()

    assert res.n_evals <= MAX_FES, f"Exceeded max evaluations: {res.n_evals} > {MAX_FES}"
    is_monotonic = all(b >= a for a, b in zip(res.history, res.history[1:]))
    assert is_monotonic, "Fitness history is not monotonic"
    assert res.best_fitness > -1e-3, f"Failed to converge. Best fitness was {res.best_fitness}"

def test_lshade_eval_count_matches_calls():
    calls = 0

    def counted(x):
        nonlocal calls
        calls += 1
        return sphere(x)

    res = make(fitness=counted).optimize()
    assert calls == res.n_evals, f"n_evals={res.n_evals} but fitness called {calls} times"
    assert calls <= MAX_FES

def test_lshade_candidates_stay_in_bounds():
    def checked(x):
        assert np.all(x >= -5) and np.all(x <= 5), f"Out-of-bounds candidate: {x}"
        return sphere(x)

    make(fitness=checked).optimize()

def test_lshade_population_reduction():
    opt = make()
    opt.optimize()
    sizes = [n for _, n, _ in opt.trace]

    assert sizes[0] <= opt.pop_size
    assert opt.pop_size == 18 * DIM, "Default N_init should be 18*D"
    assert all(b <= a for a, b in zip(sizes, sizes[1:])), "Population size must never grow"
    assert sizes[-1] >= opt.N_min, "Population fell below N_min"
    assert sizes[-1] <= opt.N_min + 2, f"Population should end near N_min, got {sizes[-1]}"
    assert sizes[0] > sizes[-1], "Population never shrank"

def test_lshade_population_follows_linear_schedule():
    opt = make()
    opt.optimize()
    for evals, n, _ in opt.trace:
        expected = max(opt.N_min,
                       int(round((opt.N_min - opt.pop_size) / MAX_FES * evals + opt.pop_size)))
        assert n == expected, f"At evals={evals}: N={n}, expected {expected}"


def test_lshade_archive_within_capacity():
    opt = make()
    opt.optimize()
    for evals, n, arc in opt.trace:
        assert arc <= int(round(opt.r_arc * n)), \
            f"Archive {arc} exceeds capacity {round(opt.r_arc * n)} at N={n}"

def test_lshade_reproducible():
    a = make(seed=7).optimize()
    b = make(seed=7).optimize()
    assert a.best_fitness == b.best_fitness
    assert np.array_equal(a.best_solution, b.best_solution)