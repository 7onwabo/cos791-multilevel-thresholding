#!/usr/bin/env python3
"""visualize_segmentation.py -- "original vs segmented" figures.

Produces the "Visual segmentation outputs displaying original vs. segmented
images"
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; figures are saved to disk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v3 as iio
import numpy as np
from skimage.filters import threshold_multiotsu

from thresholding.histogram import to_grayscale
from thresholding.metrics import _segment


# --------------------------------------------------------------------------
# Threshold sourcing
# --------------------------------------------------------------------------

def thresholds_from_results_json(path: str, maximize: bool) -> list[float]:
    """Pick the best run's thresholds from a run_one/run_experiments output file.

    Expects a JSON list of dicts, each with at least "fitness" and
    "thresholds" keys. Adjust the key names here if your `run_one`
    implementation uses different ones.
    """
    runs = json.loads(Path(path).read_text())
    best = max(runs, key=lambda r: r["fitness"]) if maximize else min(runs, key=lambda r: r["fitness"])
    return list(best["thresholds"])


def thresholds_auto(image: np.ndarray, k: int) -> list[float]:
    if threshold_multiotsu is None:
        raise ImportError(
            "scikit-image is required for --auto thresholding; "
            "install with `pip install scikit-image`."
        )
    # threshold_multiotsu returns k-1 thresholds for k classes -> we want K
    # thresholds (K+1 classes), so ask for K+1 classes.
    return list(threshold_multiotsu(image, classes=k + 1))


# --------------------------------------------------------------------------
# Figure building
# --------------------------------------------------------------------------

def load_grayscale(path: str) -> np.ndarray:
    # Same conversion as thresholding.datasets, so the figure shows the exact
    # image the thresholds were optimised on.
    return to_grayscale(iio.imread(path)).astype(np.float64)


def build_row(ax_orig, ax_hist, ax_seg, img: np.ndarray, thresholds: list[float], row_label: str) -> None:
    _, class_map = _segment(img, thresholds)
    ax_orig.imshow(img, cmap="gray", vmin=img.min(), vmax=img.max())
    ax_orig.set_ylabel(row_label, fontsize=10, rotation=90)
    ax_orig.set_xticks([]); ax_orig.set_yticks([])
    if row_label == "K=?":
        ax_orig.set_title("Original")

    intensity_counts = np.bincount(img.astype(np.uint8).ravel(), minlength=256)
    foreground_max = int(intensity_counts[1:].max()) if intensity_counts[1:].size else 0
    ax_hist.hist(
        img.ravel(),
        bins=np.arange(257) - 0.5,
        color="steelblue",
        alpha=0.8,
    )
    ax_hist.set_ylim(0, foreground_max * 1.05 if foreground_max else 1.0)
    for t in thresholds:
        ax_hist.axvline(t, color="red", linestyle="--", linewidth=1)
    ax_hist.set_yticks([])
    ax_hist.set_xlim(-0.5, 255.5)

    # Display the discrete threshold classes directly so neighbouring classes
    # remain visually distinct even when their mean intensities are similar.
    colors = plt.get_cmap("tab20", len(thresholds) + 1)
    ax_seg.imshow(
        class_map,
        cmap=ListedColormap(colors(np.arange(len(thresholds) + 1))),
        vmin=-0.5,
        vmax=len(thresholds) + 0.5,
        interpolation="nearest",
    )
    ax_seg.set_xticks([]); ax_seg.set_yticks([])
    ax_seg.set_title("Threshold classes")


def make_figure(image_path: str, rows: list[tuple[str, list[float]]], out_path: str, suptitle: str | None = None) -> None:
    """rows: list of (row_label, thresholds) pairs, one row per K value."""
    img = load_grayscale(image_path)

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows), squeeze=False)

    for i, (label, thresholds) in enumerate(rows):
        ax_orig, ax_hist, ax_seg = axes[i]
        build_row(ax_orig, ax_hist, ax_seg, img, thresholds, label)
        if i == 0:
            ax_hist.set_title("Histogram + thresholds")
            ax_seg.set_title(ax_seg.get_title() or "Segmented")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.96 if suptitle else 1.0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Original vs segmented visualisation")
    p.add_argument("--image", required=True, help="path to the grayscale image")
    p.add_argument("--out", default=None, help="output PNG path (default: <image>_segmentation.png)")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--thresholds", help="comma-separated threshold values, e.g. 60,120,180")
    src.add_argument("--auto", action="store_true", help="use skimage multi-Otsu for --k thresholds")
    src.add_argument("--results", help="single results JSON to read thresholds from")
    src.add_argument("--results-glob", help="glob of results JSON files, one row per match, sorted by K")

    p.add_argument("--k", type=int, default=None, help="number of thresholds (required with --auto)")
    p.add_argument("--k-levels", default=None, help="comma-separated K values, one row each, for --auto batch mode")
    p.add_argument("--maximize", action="store_true", help="pick best run by max fitness instead of min")
    p.add_argument("--title", default=None, help="figure suptitle")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or str(Path(args.image).with_suffix("").as_posix()) + "_segmentation.png"

    rows: list[tuple[str, list[float]]] = []

    if args.thresholds:
        t = [float(x) for x in args.thresholds.split(",")]
        rows.append((f"K={len(t)}", t))

    elif args.results:
        t = thresholds_from_results_json(args.results, args.maximize)
        rows.append((f"K={len(t)}", t))

    elif args.results_glob:
        def _k_of(path: str) -> int:
            # Sort numerically by the K in the filename (..._K12_...), not
            # lexicographically -- otherwise "K11"/"K12" sort before "K3".
            m = re.search(r"K(\d+)", Path(path).stem)
            return int(m.group(1)) if m else 0

        for path in sorted(glob.glob(args.results_glob), key=_k_of):
            t = thresholds_from_results_json(path, args.maximize)
            rows.append((f"K={len(t)} ({Path(path).stem})", t))

    elif args.auto:
        img = load_grayscale(args.image)
        if args.k_levels:
            ks = [int(x) for x in args.k_levels.split(",")]
        elif args.k:
            ks = [args.k]
        else:
            raise SystemExit("--auto requires --k or --k-levels")
        for k in ks:
            rows.append((f"K={k}", thresholds_auto(img, k)))

    if not rows:
        raise SystemExit("no thresholds resolved -- check your arguments")

    make_figure(args.image, rows, out, suptitle=args.title)


if __name__ == "__main__":
    main()
