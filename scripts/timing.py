"""Single-worker timing of every optimizer x objective x K.

The main experiment runs on all cores in parallel, so its per-run times include
CPU contention. This re-times a small subset sequentially in one process, which
gives clean absolute numbers for the execution-time and scalability analysis.

Outputs in --out (default results/<dataset>/timing):
    timing_runs.csv   one row per timed run
    time_vs_k.csv     mean ± std ms over runs, ms per 1000 FEs, growth vs smallest K
    time_vs_k.png     one panel per objective, one line per optimizer

Usage:
    python scripts/timing.py --dataset bds500
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from run_one import run_one                                   # noqa: E402
from scalability import plot, time_table                      # noqa: E402
from thresholding.config import K_LEVELS, MAX_FES, RESULTS_DIR  # noqa: E402
from thresholding.datasets import load_bds500, load_chaos     # noqa: E402
from thresholding.histogram import normalised_histogram       # noqa: E402
from thresholding.objectives import OBJECTIVES                # noqa: E402
from thresholding.optimizers import OPTIMIZERS                # noqa: E402

LOADERS = {"bds500": load_bds500, "chaos": load_chaos}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-worker optimizer timing")
    p.add_argument("--dataset", choices=sorted(LOADERS), default="bds500")
    p.add_argument("--n-images", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-fes", type=int, default=MAX_FES)
    p.add_argument("--k-levels", nargs="+", type=int, default=list(K_LEVELS))
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out or RESULTS_DIR / args.dataset / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list(LOADERS[args.dataset]().items())[: args.n_images]

    rows = []
    total = len(images) * len(OBJECTIVES) * len(args.k_levels) * len(OPTIMIZERS)
    done = 0
    for img_name, img in images:
        hist = normalised_histogram(img)
        for obj in OBJECTIVES:
            for k in args.k_levels:
                for opt in OPTIMIZERS:
                    for r in range(args.runs):
                        res = run_one(img, hist, obj, opt, k, r, args.max_fes)
                        rows.append({"image": img_name, "objective": obj, "K": k,
                                     "optimizer": opt, "run": r, "time_ms": res["time"] * 1000,
                                     "n_evals": res["n_evals"]})
                    done += 1
                    if done % 15 == 0 or done == total:
                        print(f"  [{done}/{total}]", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "timing_runs.csv", index=False)
    # Each timed run is its own sample, so the std reflects run-to-run spread.
    samples = runs.assign(image=runs["image"] + "#" + runs["run"].astype(str),
                          mean_time_ms=runs["time_ms"])
    table = time_table(samples)
    evals = runs.groupby(["objective", "optimizer", "K"])["n_evals"].mean()
    table["ms_per_1000_fes"] = [
        t / evals[(o, opt, k)] * 1000
        for t, o, opt, k in zip(table["time_ms_mean"], table["objective"], table["optimizer"], table["K"])
    ]
    table.to_csv(out_dir / "time_vs_k.csv", index=False)
    plot(table, out_dir / "time_vs_k.png",
         f"Single-worker time vs K -- {args.dataset} ({len(images)} image(s) x {args.runs} runs)")
    print(table.pivot_table(index=["objective", "optimizer"], columns="K",
                            values="time_ms_mean").round(1).to_string())
    print(f"Saved timing results in {out_dir}")


if __name__ == "__main__":
    main()
