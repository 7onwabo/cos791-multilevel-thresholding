"""
Helper function which runs one optimisation run.
Builds a fitness function for one image, create an optimiser with a specific seed and calls optimize() and returns the result
"""

import time
import numpy as np

from thresholding.config import RANDOM_SEED, threshold_bounds
from thresholding.objectives import OBJECTIVES
from thresholding.objectives._base import sanitise_thresholds
from thresholding.optimizers import OPTIMIZERS

def run_one(hist, obj_name, opt_name, k, run_idx, max_fes):
    objective = OBJECTIVES[obj_name]
    fitness = lambda x: objective(hist, x)

    opt = OPTIMIZERS[opt_name](
        fitness, threshold_bounds(k),
        max_fes=max_fes,
        seed=RANDOM_SEED + run_idx
    )

    t0 = time.perf_counter()
    res = opt.optimize()
    elapsed = time.perf_counter() - t0

    return {
        "thresholds": sanitise_thresholds(res.best_solution),
        "fitness": res.best_fitness,
        "history": res.history,
        "n_evals": res.n_evals,
        "time": elapsed,
    }