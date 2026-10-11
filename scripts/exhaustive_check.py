"""Exhaustive-search check at small K: are the DE results globally optimal?

For every image and objective, evaluates all C(254, K) sorted threshold
vectors with thresholds in [1, 254] (vectorised over histogram prefix sums,
same class convention as the objectives) and compares the global optimum with
the best objective value found by each algorithm.

Outputs in <results-dir>:
    exhaustive_K<k>.csv           one row per (image, objective, algorithm)
    exhaustive_K<k>_summary.csv   share of runs and images reaching the optimum

Usage:
    python scripts/exhaustive_check.py --results-dir results/bds500 --dataset bds500
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from thresholding.config import EPSILON, GRAY_LEVELS, TSALLIS_Q
from thresholding.datasets import load_bds500, load_chaos
from thresholding.histogram import normalised_histogram
from thresholding.objectives import OBJECTIVES
from thresholding.optimizers import OPTIMIZERS

LOADERS = {"bds500": load_bds500, "chaos": load_chaos}
REL_TOL = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exhaustive-search optimality check")
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    p.add_argument("--k", type=int, default=3, help="number of thresholds (3 is ~2.7M vectors)")
    return p.parse_args()


def all_objective_values(hist: np.ndarray, t: np.ndarray, q: float = TSALLIS_Q) -> dict[str, np.ndarray]:
    """Objective values for every row of the sorted threshold matrix ``t``."""
    levels = np.arange(GRAY_LEVELS)
    p = hist.astype(float)
    nz = p > EPSILON
    prefix = lambda x: np.concatenate([[0.0], np.cumsum(x)])
    P, M = prefix(p), prefix(levels * p)
    E = prefix(np.where(nz, p * np.log(np.where(nz, p, 1.0)), 0.0))
    Q = prefix(np.where(nz, p ** q, 0.0))
    mu_t = M[-1]

    n = t.shape[0]
    edges = np.column_stack([np.full(n, -1), t, np.full(n, GRAY_LEVELS - 1)]) + 1
    otsu = np.zeros(n)
    kapur = np.zeros(n)
    s_sum = np.zeros(n)
    s_prod = np.ones(n)
    any_class = np.zeros(n, dtype=bool)
    for c in range(edges.shape[1] - 1):
        a, b = edges[:, c], edges[:, c + 1]
        w = P[b] - P[a]
        ok = w >= EPSILON
        ws = np.where(ok, w, 1.0)
        otsu += np.where(ok, w * ((M[b] - M[a]) / ws - mu_t) ** 2, 0.0)
        kapur += np.where(ok, -(E[b] - E[a]) / ws + np.log(ws), 0.0)
        s = (1.0 - (Q[b] - Q[a]) / ws ** q) / (q - 1.0)
        s_sum += np.where(ok, s, 0.0)
        s_prod *= np.where(ok, s, 1.0)
        any_class |= ok
    tsallis = np.where(any_class, s_sum + (1.0 - q) * s_prod, 0.0)
    return {"otsu": otsu, "kapur": kapur, "tsallis": tsallis}


def main() -> None:
    args = parse_args()
    t = np.array(list(combinations(range(1, GRAY_LEVELS - 1), args.k)), dtype=np.int64)
    rng = np.random.default_rng(0)
    rows = []
    for img_name, img in LOADERS[args.dataset]().items():
        hist = normalised_histogram(img)
        values = all_objective_values(hist, t)
        for obj, v in values.items():
            # Guard: the vectorised values must match the real objective function.
            for i in rng.integers(0, len(t), 5):
                assert np.isclose(v[i], OBJECTIVES[obj](hist, t[i]), rtol=1e-9, atol=1e-9), (img_name, obj)
            opt_idx = int(np.argmax(v))
            optimum = float(v[opt_idx])
            for alg in OPTIMIZERS:
                path = args.results_dir / f"{args.dataset}__{img_name}__{obj}__K{args.k}__{alg}.json"
                if not path.is_file():
                    continue
                fits = np.array([r["fitness"] for r in json.loads(path.read_text())])
                hit = fits >= optimum - REL_TOL * max(abs(optimum), 1.0)
                rows.append({"image": img_name, "objective": obj, "optimizer": alg,
                             "optimum": optimum, "optimal_thresholds": t[opt_idx].tolist(),
                             "best_found": float(fits.max()), "mean_found": float(fits.mean()),
                             "best_gap_rel": (optimum - fits.max()) / max(abs(optimum), 1e-12),
                             "success_rate": float(hit.mean())})
        print(f"  {img_name}: done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.results_dir / f"exhaustive_K{args.k}.csv", index=False)
    summary = (df.assign(image_hit=df["best_gap_rel"] <= REL_TOL)
               .groupby(["objective", "optimizer"], sort=False)
               .agg(success_rate=("success_rate", "mean"), images_optimal=("image_hit", "sum"),
                    n_images=("image", "nunique"), max_gap_rel=("best_gap_rel", "max"))
               .reset_index())
    summary.to_csv(args.results_dir / f"exhaustive_K{args.k}_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
