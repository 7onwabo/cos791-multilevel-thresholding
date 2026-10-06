"""Summary table of fitness, runtime, and segmentation metrics.

Usage:
    python results/summary_table.py --results-dir results/phase1_shade_vs_lshade
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

METRICS = ("psnr", "ssim", "uniformity", "class_separability")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise optimisation results")
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--dataset", default="bds500")
    p.add_argument("--objectives", nargs="+", default=["otsu", "kapur", "tsallis"])
    p.add_argument("--optimizers", nargs="+", default=["shade", "lshade"])
    p.add_argument("--k-levels", nargs="+", type=int, default=[3, 5, 7, 9, 11, 12])
    p.add_argument("--out", default=None, help="output CSV path (default: <results-dir>/summary.csv)")
    return p.parse_args()


def load_runs(results_dir: Path, dataset: str, obj: str, k: int, opt: str) -> list[dict]:
    pattern = str(results_dir / f"{dataset}__*__{obj}__K{k}__{opt}.json")
    runs = []
    for f in glob.glob(pattern):
        runs.extend(json.load(open(f)))
    return runs


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else args.results_dir / "summary.csv"

    rows = []
    for obj in args.objectives:
        for k in args.k_levels:
            for opt in args.optimizers:
                runs = load_runs(args.results_dir, args.dataset, obj, k, opt)
                if not runs:
                    continue
                fits = [r["fitness"] for r in runs]
                times = [r["time"] for r in runs]
                evals = [r["n_evals"] for r in runs]
                row = {
                    "objective": obj,
                    "K": k,
                    "optimizer": opt,
                    "n_runs": len(runs),
                    "mean_fitness": np.mean(fits),
                    "std_fitness": np.std(fits),
                    "mean_time_s": np.mean(times),
                    "mean_n_evals": np.mean(evals),
                }
                for metric in METRICS:
                    values = [r[metric] for r in runs if metric in r]
                    if values:
                        row[f"mean_{metric}"] = np.mean(values)
                        row[f"std_{metric}"] = np.std(values)
                rows.append(row)

    if not rows:
        raise SystemExit(f"No result files found under {args.results_dir}. "
                          f"Check --dataset/--objectives/--optimizers/--k-levels match your run.")

    with open(out_path, "w", newline="") as f:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"{r['objective']:8s} K={r['K']:2d} {r['optimizer']:6s} "
              f"{r['mean_fitness']:.5f} \u00b1 {r['std_fitness']:.5f}  "
              f"({r['mean_time_s']:.2f}s/run, n={r['n_runs']})")
        metrics = "  ".join(
            f"{metric}={r[f'mean_{metric}']:.4f} \u00b1 {r[f'std_{metric}']:.4f}"
            for metric in METRICS
            if f"mean_{metric}" in r
        )
        if metrics:
            print(f"  {metrics}")

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
