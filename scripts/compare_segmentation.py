"""Side-by-side segmentation comparison: all optimizers on one image at one K.

One figure per (image, K): rows = objectives, columns = original + one panel
per optimizer showing its best run (highest fitness of the independent runs),
annotated with that run's PSNR / SSIM.

Outputs: <results-dir>/comparison_figures/<image>__K<k>.png

Usage:
    python scripts/compare_segmentation.py --results-dir results/bds500 \
        --dataset bds500 --images img1 img2 --k-levels 3 7 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thresholding.datasets import load_bds500, load_chaos
from thresholding.metrics import _segment
from thresholding.objectives import OBJECTIVES
from thresholding.optimizers import OPTIMIZERS

LOADERS = {"bds500": load_bds500, "chaos": load_chaos}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimizer comparison figures")
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    p.add_argument("--images", nargs="+", default=None, help="image ids (default: first 5)")
    p.add_argument("--k-levels", nargs="+", type=int, default=[3, 7, 12])
    p.add_argument("--objectives", nargs="+", default=list(OBJECTIVES))
    p.add_argument("--optimizers", nargs="+", default=list(OPTIMIZERS))
    p.add_argument("--style", choices=["classes", "recon"], default="classes",
                   help="classes: colour-coded class map; recon: class-mean greyscale image")
    return p.parse_args()


def best_run(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return max(json.loads(path.read_text()), key=lambda r: r["fitness"])


def make_figure(args, img_name: str, img: np.ndarray, k: int, out_path: Path) -> bool:
    n_rows, n_cols = len(args.objectives), len(args.optimizers) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.1 * n_rows + 0.4),
                             squeeze=False)
    found = False
    for i, obj in enumerate(args.objectives):
        axes[i][0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[i][0].set_ylabel(obj, fontsize=10)
        if i == 0:
            axes[i][0].set_title("Original", fontsize=9)
        for j, opt in enumerate(args.optimizers, start=1):
            ax = axes[i][j]
            run = best_run(args.results_dir / f"{args.dataset}__{img_name}__{obj}__K{k}__{opt}.json")
            if run is None:
                ax.text(0.5, 0.5, "no result", ha="center", va="center", fontsize=8)
            else:
                found = True
                recon, classes = _segment(img, run["thresholds"])
                if args.style == "recon":
                    ax.imshow(recon, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(classes, cmap="viridis", vmin=0, vmax=k, interpolation="nearest")
                ax.set_xlabel(f"PSNR {run['psnr']:.2f}  SSIM {run['ssim']:.3f}", fontsize=7)
            if i == 0:
                ax.set_title(opt, fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{args.dataset} {img_name}, K = {k} (best of runs per optimizer)", fontsize=10)
    fig.tight_layout()
    if found:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return found


def main() -> None:
    args = parse_args()
    images = LOADERS[args.dataset]()
    names = args.images or list(images)[:5]
    out_dir = args.results_dir / "comparison_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in images:
            print(f"  skipping {name}: not in {args.dataset}")
            continue
        img = images[name].astype(np.float64)
        for k in args.k_levels:
            out_path = out_dir / f"{name}__K{k}.png"
            if make_figure(args, name, img, k, out_path):
                print(f"saved: {out_path}")
    print(f"Comparison figures in {out_dir}")


if __name__ == "__main__":
    main()
