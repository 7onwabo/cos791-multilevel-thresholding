"""Master experiment script (Assignment section 4 deliverable) -- SKELETON.

Goal: one command reproduces every result. Full wiring lands once the DE
variants (Assignment 1.3) and metrics (section 2) are implemented. For now it
loads data, enumerates the experiment grid, and reports what *would* run so the
CLI and structure can be reviewed early.

Usage:
    python scripts/run_experiments.py --dataset bds500 --objective otsu --k 3
    python scripts/run_experiments.py --dataset all --objective all --k all
"""

from __future__ import annotations

import argparse

from thresholding.config import K_LEVELS, N_RUNS
from thresholding.datasets import load_bds500, load_chaos
from thresholding.objectives import OBJECTIVES
from thresholding.optimizers import OPTIMIZERS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COS791 multilevel thresholding experiments")
    p.add_argument("--dataset", choices=["bds500", "chaos", "all"], default="all")
    p.add_argument("--objective", choices=[*OBJECTIVES, "all"], default="all")
    p.add_argument("--optimizer", choices=[*OPTIMIZERS, "all"], default="all")
    p.add_argument("--k", default="all", help="single K value or 'all'")
    p.add_argument("--runs", type=int, default=N_RUNS)
    return p.parse_args()


def _selected(name: str, mapping: dict) -> list[str]:
    return list(mapping) if name == "all" else [name]


def main() -> None:
    args = parse_args()

    datasets: dict[str, dict] = {}
    if args.dataset in ("bds500", "all"):
        datasets["bds500"] = load_bds500()
    if args.dataset in ("chaos", "all"):
        datasets["chaos"] = load_chaos()

    ks = list(K_LEVELS) if args.k == "all" else [int(args.k)]
    objectives = _selected(args.objective, OBJECTIVES)
    optimizers = _selected(args.optimizer, OPTIMIZERS)

    print("=== Experiment grid ===")
    for ds_name, images in datasets.items():
        print(f"  dataset={ds_name}: {len(images)} images")
    print(f"  objectives={objectives}")
    print(f"  optimizers={optimizers}")
    print(f"  K levels={ks}")
    print(f"  runs per (image, objective, K)={args.runs}")

    total = sum(len(imgs) for imgs in datasets.values())
    total *= len(objectives) * len(optimizers) * len(ks) * args.runs
    print(f"  -> {total:,} total optimisation runs when fully wired")
    print("\n[skeleton] DE variants + metrics not yet implemented (Assignment 1.3 / section 2).")


if __name__ == "__main__":
    main()
