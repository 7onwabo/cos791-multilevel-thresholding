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
import json
from pathlib import Path
import numpy as np

from run_one import run_one
from thresholding.config import K_LEVELS, N_RUNS
from thresholding.datasets import load_bds500, load_chaos
from thresholding.objectives import OBJECTIVES
from thresholding.optimizers import OPTIMIZERS
from thresholding.config import MAX_FES
from thresholding.histogram import normalised_histogram


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COS791 multilevel thresholding experiments")
    p.add_argument("--dataset", choices=["bds500", "chaos", "all"], default="all")
    p.add_argument("--objective", choices=[*OBJECTIVES, "all"], default="all")
    p.add_argument("--optimizer", choices=[*OPTIMIZERS, "all"], default="all")
    p.add_argument("--k", default="all", help="single K value or 'all'")
    p.add_argument("--runs", type=int, default=N_RUNS)
    p.add_argument("--max-fes", type=int, default=MAX_FES)
    p.add_argument("--n-images", type=int, default=None, help="use only the first N images")
    p.add_argument("--out", default="results/raw")

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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Experiment grid ===")
    for ds_name, images in datasets.items():
        print(f"  dataset={ds_name}: {len(images)} images")
    print(f"  objectives={objectives}")
    print(f"  optimizers={optimizers}")
    print(f"  K levels={ks}")
    print(f"  runs per (image, objective, K)={args.runs}")

    total = sum(len(imgs) for imgs in datasets.values())
    total *= len(objectives) * len(optimizers) * len(ks) * args.runs
    print(f"  -> {total:,} total optimisation runs")

    for ds_name, images in datasets.items():
            items = list(images.items())[: args.n_images]
            for img_name, img in items:
                hist = normalised_histogram(img)
                for obj in objectives:
                    for k in ks:
                        for opt in optimizers:
                            path = out_dir / f"{ds_name}__{img_name}__{obj}__K{k}__{opt}.json"
                            if path.exists():
                                continue
                            runs = [run_one(hist, obj, opt, k, r, args.max_fes)
                                for r in range(args.runs)]
    
                            path.write_text(json.dumps(runs))
                            fits = [r["fitness"] for r in runs]
                            print(f"{img_name} {obj} K={k} {opt}: " f"{np.mean(fits):.5f} ± {np.std(fits):.5f}")


if __name__ == "__main__":
    main()
