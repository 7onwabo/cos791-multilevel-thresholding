"""Plot SHADE vs L-SHADE convergence curves for Phase 1 (BSD500).

Usage:
    python results/convergence_plot.py --results-dir results/phase1_shade_vs_lshade
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot convergence curves")
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--dataset", default="bds500")
    p.add_argument("--objectives", nargs="+", default=["otsu", "kapur", "tsallis"])
    p.add_argument("--optimizers", nargs="+", default=["shade", "lshade"])
    p.add_argument("--k-levels", nargs="+", type=int, default=[3, 5, 7, 9, 11, 12])
    p.add_argument("--out", default=None, help="output PNG path (default: <results-dir>/convergence.png)")
    return p.parse_args()


def load_runs(results_dir: Path, dataset: str, obj: str, k: int, opt: str) -> list[dict]:
    pattern = str(results_dir / f"{dataset}__*__{obj}__K{k}__{opt}.json")
    runs = []
    for f in glob.glob(pattern):
        runs.extend(json.load(open(f)))
    return runs


def stacked_history(runs: list[dict]) -> np.ndarray | None:
    if not runs:
        return None
    n = min(len(r["history"]) for r in runs)   # truncate to shortest run
    return np.array([r["history"][:n] for r in runs])


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else args.results_dir / "convergence.png"

    fig, axes = plt.subplots(
        len(args.objectives), len(args.k_levels),
        figsize=(3.2 * len(args.k_levels), 2.8 * len(args.objectives)),
        squeeze=False,
    )

    any_data = False
    for i, obj in enumerate(args.objectives):
        for j, k in enumerate(args.k_levels):
            ax = axes[i][j]
            for opt in args.optimizers:
                H = stacked_history(load_runs(args.results_dir, args.dataset, obj, k, opt))
                if H is None:
                    continue
                any_data = True
                m, s = H.mean(0), H.std(0)
                x = np.arange(len(m))
                ax.plot(x, m, label=opt, linewidth=1.5)
                ax.fill_between(x, m - s, m + s, alpha=0.2)
            if i == 0:
                ax.set_title(f"K={k}")
            if j == 0:
                ax.set_ylabel(obj)
            if i == len(args.objectives) - 1:
                ax.set_xlabel("Generation")
            ax.legend(fontsize=7)

    if not any_data:
        raise SystemExit(f"No result files found under {args.results_dir}. "
                          f"Check --dataset/--objectives/--optimizers/--k-levels match your run.")

    fig.suptitle(f"{' vs '.join(args.optimizers)} convergence -- {args.dataset}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
