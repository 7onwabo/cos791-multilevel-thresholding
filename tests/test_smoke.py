import json, glob
import numpy as np
from thresholding.histogram import normalised_histogram
from thresholding.datasets import load_bds500
from thresholding.objectives import OBJECTIVES
import pytest

def test_smoke():
    img = next(iter(load_bds500().values()))
    hist = normalised_histogram(img)
    rng = np.random.default_rng(0)
    rand_fit = np.mean([OBJECTIVES["otsu"](hist, rng.integers(1, 255, 3)) for _ in range(300)])

    for f in glob.glob("results/smoke/*.json"):
        for r in json.load(open(f)):
            t = r["thresholds"]
            assert r["n_evals"] <= 2000, f"budget overshoot"
            assert t == sorted(t) and 0 <= t[0] and t[-1] <= 255, f"bad thresholds in {f}"
            assert len(set(t)) == len(t), f"duplicate thresholds in {f}"
            assert all(b >= a for a, b in zip(r["history"], r["history"][1:])), f"history decreased in {f}"
            assert r["fitness"] > rand_fit, f"no better than random in {f}"