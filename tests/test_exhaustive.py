import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from thresholding.datasets import load_chaos
from thresholding.histogram import normalised_histogram
from thresholding.objectives import OBJECTIVES


def test_vectorised_objectives_match_objective_functions():
    from exhaustive_check import all_objective_values

    hist = normalised_histogram(next(iter(load_chaos().values())))
    t = np.array([[1, 2, 3], [10, 80, 200], [50, 51, 254], [100, 150, 151]])
    values = all_objective_values(hist, t)
    for name, v in values.items():
        expected = [OBJECTIVES[name](hist, row) for row in t]
        assert np.allclose(v, expected, rtol=1e-9, atol=1e-9), name
