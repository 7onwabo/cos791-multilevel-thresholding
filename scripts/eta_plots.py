"""Class-separability figures: 1 - eta against K, and per-image eta spread.

eta saturates near 1, so differences show up in the fourth decimal of a table.
Plotting 1 - eta (the share of intensity variance left within classes) on a
log axis makes them visible. The box plot shows how eta varies across images
(per-image mean over runs), which is the robustness view for Phase 2.

Outputs in --results-dir:
    eta_vs_k.png          1 - eta (mean over images and runs) vs K, one panel per objective
    eta_per_image_K<k>.png  per-image eta at one K, one panel per objective
    eta_per_image_K<k>.csv  min / median / max eta across images per (objective, algorithm)

Usage:
    python scripts/eta_plots.py --results-dir results/chaos --dataset chaos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thresholding.objectives import OBJECTIVES
from thresholding.optimizers import OPTIMIZERS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Class separability figures")
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--dataset", default=None, help="label for titles")
    p.add_argument("--k", type=int, default=12, help="K for the per-image box plot")
    return p.parse_args()


def eta_vs_k(summary: pd.DataFrame, out: Path, title: str) -> None:
    objs = [o for o in OBJECTIVES if o in set(summary["objective"])]
    fig, axes = plt.subplots(1, len(objs), figsize=(4.2 * len(objs), 3.2), squeeze=False)
    for ax, obj in zip(axes[0], objs):
        sub = summary[summary["objective"] == obj]
        for opt in OPTIMIZERS:
            g = sub[sub["optimizer"] == opt].sort_values("K")
            if not g.empty:
                ax.plot(g["K"], 1 - g["class_separability_mu"], marker="o", markersize=3, label=opt)
        ax.set_yscale("log")
        ax.set_title(obj)
        ax.set_xlabel("K (number of thresholds)")
        ax.set_xticks(sorted(sub["K"].unique()))
    axes[0][0].set_ylabel(r"$1 - \eta$ (lower is better)")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def eta_per_image(per_img: pd.DataFrame, k: int, out_png: Path, out_csv: Path, title: str) -> None:
    sub = per_img[per_img["K"] == k]
    objs = [o for o in OBJECTIVES if o in set(sub["objective"])]
    opts = [o for o in OPTIMIZERS if o in set(sub["optimizer"])]
    fig, axes = plt.subplots(1, len(objs), figsize=(4.2 * len(objs), 3.4), squeeze=False)
    for ax, obj in zip(axes[0], objs):
        data = [sub[(sub["objective"] == obj) & (sub["optimizer"] == o)]["mean_class_separability"]
                for o in opts]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(opts) + 1), opts, rotation=30, fontsize=8)
        ax.set_title(obj)
    axes[0][0].set_ylabel(r"$\eta$ per image (mean over runs)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    stats = (sub.groupby(["objective", "optimizer"])["mean_class_separability"]
             .agg(["min", "median", "max", "std"]).reset_index())
    worst = sub.loc[sub.groupby(["objective", "optimizer"])["mean_class_separability"].idxmin(),
                    ["objective", "optimizer", "image"]].rename(columns={"image": "worst_image"})
    best = sub.loc[sub.groupby(["objective", "optimizer"])["mean_class_separability"].idxmax(),
                   ["objective", "optimizer", "image"]].rename(columns={"image": "best_image"})
    stats.merge(worst).merge(best).to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()
    label = args.dataset or args.results_dir.name
    summary = pd.read_csv(args.results_dir / "summary.csv")
    per_img = pd.read_csv(args.results_dir / "summary_per_image.csv")
    eta_vs_k(summary, args.results_dir / "eta_vs_k.png", f"Class separability vs K -- {label}")
    eta_per_image(per_img, args.k, args.results_dir / f"eta_per_image_K{args.k}.png",
                  args.results_dir / f"eta_per_image_K{args.k}.csv",
                  f"Per-image class separability at K = {args.k} -- {label}")
    print(f"Saved eta figures in {args.results_dir}")


if __name__ == "__main__":
    main()
