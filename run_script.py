#!/usr/bin/env python3
"""
Master script: run experiments, then generate plots, summary tables, statistics
and visual segmentation figures for one dataset.

Examples:
    # Phase 1 (BSD500, 10 images), everything
    python run_script.py --dataset bds500 --img-dir data/BDS500

    # Phase 2 (CHAOS MRI, 15 slices), everything
    python run_script.py --dataset chaos --img-dir data/CHAOS

    # Re-generate tables/plots/figures only (results already on disk)
    python run_script.py --dataset bds500 --skip-experiments

Stages can be skipped with --skip-experiments / --skip-plots /
--skip-summary / --skip-figures.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OBJECTIVES = ["otsu", "kapur", "tsallis"]
# Names must match the optimizer part of the result filenames exactly
# (filenames are case-sensitive): <dataset>__<image>__<obj>__K<k>__<optimizer>.json
OPTIMIZERS = ["SHADE", "L-SHADE"]
K_LEVELS = [3, 5, 7, 9, 11, 12]
RUNS = 30
MAX_FES = 30000          # identical budget for every algorithm

# Per-dataset defaults: number of images, default image folder and extension
DATASETS = {
    "bds500": {"n_images": 10, "img_dir": Path("data/BDS500"), "img_ext": "png"},
    "chaos":  {"n_images": 15, "img_dir": Path("data/CHAOS"),  "img_ext": "png"},
}

SCRIPTS = Path("scripts")
STATS_METRICS = ["psnr", "ssim", "uniformity"]


def run(cmd: list[str]) -> None:
    """Run a subprocess, streaming output, raising on failure."""
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run experiments and generate reports")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="bds500")
    p.add_argument("--out", type=Path, default=None,
                   help="results directory (default: results/<dataset>)")
    p.add_argument("--img-dir", type=Path, default=None,
                   help="folder with the source image files (default depends on dataset)")
    p.add_argument("--img-ext", default=None, help="image file extension (default: png)")
    p.add_argument("--n-images", type=int, default=None,
                   help="number of images to run (default: 10 for bds500, 15 for chaos)")
    p.add_argument("--runs", type=int, default=RUNS)
    p.add_argument("--max-fes", type=int, default=MAX_FES)
    p.add_argument("--n-report-images", type=int, default=5,
                   help="number of images to render segmentation figures for (default: 5)")
    p.add_argument("--force", action="store_true",
                   help="re-run combinations even if their result files already exist")
    p.add_argument("--skip-experiments", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--skip-summary", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args()


def result_files(out_dir: Path, dataset: str, obj: str, k: int, opt: str) -> list[Path]:
    return sorted(out_dir.glob(f"{dataset}__*__{obj}__K{k}__{opt}.json"))


def run_experiments(args, out_dir: Path, n_images: int) -> None:
    total = len(OBJECTIVES) * len(K_LEVELS) * len(OPTIMIZERS)
    count = 0
    t_start = time.perf_counter()
    for obj in OBJECTIVES:
        for k in K_LEVELS:
            for opt in OPTIMIZERS:
                count += 1
                tag = f"[{count}/{total}] dataset={args.dataset} objective={obj} K={k} optimizer={opt}"
                if not args.force and len(result_files(out_dir, args.dataset, obj, k, opt)) >= n_images:
                    print(f"{tag} -> already complete, skipping (use --force to re-run)")
                    continue
                print(tag)
                t0 = time.perf_counter()
                run([
                    sys.executable, str(SCRIPTS / "run_experiments.py"),
                    "--dataset", args.dataset,
                    "--objective", obj,
                    "--optimizer", opt,
                    "--k", str(k),
                    "--n-images", str(n_images),
                    "--runs", str(args.runs),
                    "--max-fes", str(args.max_fes),
                    "--out", str(out_dir),
                ])
                print(f"  -> {time.perf_counter() - t0:.1f}s")
    print(f"All experiments finished in {(time.perf_counter() - t_start) / 60:.1f} min")


def run_plots(args, out_dir: Path) -> None:
    run([
        sys.executable, str(SCRIPTS / "convergence_plot.py"),
        "--results-dir", str(out_dir),
        "--dataset", args.dataset,
        "--objectives", *OBJECTIVES,
        "--optimizers", *OPTIMIZERS,
        "--k-levels", *[str(k) for k in K_LEVELS],
    ])


def run_summary(args, out_dir: Path, n_images: int) -> None:
    base = [
        sys.executable, str(SCRIPTS / "summary_table.py"),
        "--results-dir", str(out_dir),
        "--dataset", args.dataset,
        "--objectives", *OBJECTIVES,
        "--optimizers", *OPTIMIZERS,
        "--k-levels", *[str(k) for k in K_LEVELS],
        "--n-images", str(n_images),
    ]
    # First call writes summary.csv + summary_per_image.csv and the PSNR stats
    run(base + ["--stats", "--stats-metric", STATS_METRICS[0]])
    # Remaining metrics only need the Friedman/Wilcoxon files (summary is rewritten identically)
    for metric in STATS_METRICS[1:]:
        run(base + ["--stats", "--stats-metric", metric])


def run_figures(args, out_dir: Path, img_dir: Path, img_ext: str) -> None:
    # Result filename: <dataset>__<image>__<objective>__K<k>__<optimizer>.json
    result_re = re.compile(rf"^{re.escape(args.dataset)}__(.+)__([^_]+)__K(\d+)__(.+)\.json$")
    vis_out = out_dir / "segmentation_figures"
    vis_out.mkdir(parents=True, exist_ok=True)

    # Discover image names from the result filenames themselves.
    image_names: list[str] = []
    for path in sorted(out_dir.glob(f"{args.dataset}__*__K*__*.json")):
        m = result_re.match(path.name)
        if m and m.group(1) not in image_names:
            image_names.append(m.group(1))
        if len(image_names) >= args.n_report_images:
            break

    if not image_names:
        print(f"  no result files found under {out_dir} -- skipping figures")
        return

    for img_name in image_names:
        image_path = img_dir / f"{img_name}.{img_ext}"
        if not image_path.is_file():
            print(f"  skipping {img_name}: image not found at {image_path} "
                  f"(check --img-dir / --img-ext)")
            continue
        for obj in OBJECTIVES:
            for opt in OPTIMIZERS:
                pattern = f"{args.dataset}__{img_name}__{obj}__K*__{opt}.json"
                if not list(out_dir.glob(pattern)):
                    continue
                out_png = vis_out / f"{img_name}__{obj}__{opt}.png"
                print(f"  figure: {img_name} / {obj} / {opt} -> {out_png}")
                run([
                    sys.executable, str(SCRIPTS / "visualize_segmentation.py"),
                    "--image", str(image_path),
                    "--results-glob", str(out_dir / pattern),
                    "--out", str(out_png),
                    "--title", f"{img_name} -- {obj} -- {opt}",
                ])
    print(f"Figures in {vis_out}")


def main() -> None:
    args = parse_args()
    cfg = DATASETS[args.dataset]
    out_dir = args.out or Path("results") / args.dataset
    img_dir = args.img_dir or cfg["img_dir"]
    img_ext = (args.img_ext or cfg["img_ext"]).lstrip(".")
    n_images = args.n_images or cfg["n_images"]

    if args.n_report_images < 1:
        raise SystemExit("--n-report-images must be at least 1")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_experiments:
        run_experiments(args, out_dir, n_images)
    if not args.skip_plots:
        print("Generating convergence plots...")
        run_plots(args, out_dir)
    if not args.skip_summary:
        print("Generating summary tables and statistical tests...")
        run_summary(args, out_dir, n_images)
    if not args.skip_figures:
        print("Generating visual segmentation outputs (original vs. segmented)...")
        run_figures(args, out_dir, img_dir, img_ext)

    print(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()