"""The convergence x-axis must be function evaluations, not generation index.

The variants are compared under an equal-FE budget but spend it over very
different generation counts, so a generation axis would misrepresent them.
These tests pin the data that makes an FE axis possible.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from thresholding.config import threshold_bounds
from thresholding.datasets import load_chaos
from thresholding.histogram import normalised_histogram
from thresholding.objectives import OBJECTIVES
from thresholding.optimizers import OPTIMIZERS

IMPLEMENTED = tuple(OPTIMIZERS)
MAX_FES = 3000


def _run(name, k=12):
    hist = normalised_histogram(sorted(load_chaos().items())[0][1])
    return OPTIMIZERS[name](
        lambda x: OBJECTIVES["otsu"](hist, x),
        threshold_bounds(k), max_fes=MAX_FES, seed=42,
    ).optimize()


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_evals_history_aligns_with_fitness_history(name):
    res = _run(name)
    assert len(res.evals_history) == len(res.history), \
        f"{name}: {len(res.evals_history)} eval points vs {len(res.history)} fitness points"
    assert res.evals_history[-1] == res.n_evals
    assert all(b >= a for a, b in zip(res.evals_history, res.evals_history[1:])), \
        f"{name}: evals_history must be non-decreasing"
    assert res.evals_history[-1] <= MAX_FES


def test_variants_differ_in_generations_so_axis_choice_matters():
    """Guards the reason the FE axis exists: generation counts are not comparable."""
    gens = {name: len(_run(name).history) for name in IMPLEMENTED}
    assert max(gens.values()) > 2 * min(gens.values()), (
        "Variants no longer differ much in generation count; if this is "
        f"intentional, revisit the convergence x-axis rationale. Got {gens}"
    )


def test_lshade_fes_per_generation_shrink_with_population():
    """L-SHADE reduces its population, so FEs per generation must fall too.

    This is why a run's FE axis cannot be reconstructed as linspace(0, n_evals)
    after the fact -- the spacing is genuinely uneven.
    """
    res = _run("L-SHADE")
    per_gen = np.diff(res.evals_history)
    assert per_gen[0] > per_gen[-1], \
        f"expected shrinking FEs/generation, got first={per_gen[0]} last={per_gen[-1]}"


def test_convergence_plot_resample_uses_recorded_fe_axis():
    from convergence_plot import resample

    runs = [{"history": [0.0, 10.0], "evals_history": [0, 1000], "n_evals": 1000}]
    grid = np.array([0.0, 500.0, 1000.0])
    got = resample(runs, grid)[0]
    assert got == pytest.approx([0.0, 5.0, 10.0])

    # Without evals_history the axis falls back to even spacing over n_evals.
    legacy = [{"history": [0.0, 10.0], "n_evals": 1000}]
    assert resample(legacy, grid)[0] == pytest.approx([0.0, 5.0, 10.0])


def test_relative_error_uses_per_image_best_known():
    from convergence_plot import relative_error_curves

    run = lambda h: {"history": h, "evals_history": [0, 100], "n_evals": 100}
    per_opt = {
        "A": {"img1": [run([50.0, 100.0])], "img2": [run([1.0, 2.0])]},
        "B": {"img1": [run([50.0, 90.0])], "img2": [run([1.0, 1.0])]},
    }
    E = relative_error_curves(per_opt, np.array([0.0, 100.0]))
    assert E["A"][:, -1] == pytest.approx([1e-9, 1e-9])
    assert E["B"][:, -1] == pytest.approx([0.1, 0.5])
