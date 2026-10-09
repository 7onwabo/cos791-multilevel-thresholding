#!/usr/bin/env python3
"""
Master script: reproduces every experimental result in the report.

For each dataset it runs the experiment grid, then writes convergence plots,
summary and report tables, statistical tests, time-vs-K scalability,
segmentation figures, single-worker timing and the Tsallis q sweep. The
algorithm parameter table is written once to results/.

Examples:
    # Everything, both phases (BSD500 then CHAOS)
    python run_script.py

    # One phase only
    python run_script.py --dataset chaos

    # Re-generate tables/plots/figures only (results already on disk)
    python run_script.py --skip-experiments --skip-q-sweep --skip-timing

Stages can be skipped with --skip-experiments / --skip-plots / --skip-summary /
--skip-figures / --skip-timing / --skip-q-sweep.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from thresholding import config
from thresholding.objectives import OBJECTIVES as _OBJECTIVE_REGISTRY
from thresholding.optimizers import OPTIMIZERS as _OPTIMIZER_REGISTRY

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Registry names are what the result filenames use:
# <dataset>__<image>__<obj>__K<k>__<optimizer>.json
OBJECTIVES = list(_OBJECTIVE_REGISTRY)
OPTIMIZERS = list(_OPTIMIZER_REGISTRY)
K_LEVELS = list(config.K_LEVELS)
RUNS = config.N_RUNS
MAX_FES = config.MAX_FES          # identical budget for every algorithm

# Per-dataset defaults: number of images, default image folder and extension
DATASETS = {
    "bds500": {"n_images": 10, "img_dir": config.BDS500_DIR, "img_ext": "png"},
    "chaos":  {"n_images": 15, "img_dir": config.CHAOS_DIR,  "img_ext": "png"},
}

SCRIPTS = config.REPO_ROOT / "scripts"


def run(cmd: list[str]) -> None:
    """Run a subprocess, streaming output, raising on failure."""
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run experiments and generate reports")
    p.add_argument("--dataset", choices=[*sorted(DATASETS), "all"], default="all")
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR,
                   help="results root; each dataset goes in <out>/<dataset> (default: results)")
    p.add_argument("--img-dir", type=Path, default=None,
                   help="folder with the source image files (single dataset only)")
    p.add_argument("--img-ext", default=None, help="image file extension (default: png)")
    p.add_argument("--n-images", type=int, default=None,
                   help="number of images to run (default: 10 for bds500, 15 for chaos)")
    p.add_argument("--runs", type=int, default=RUNS)
    p.add_argument("--max-fes", type=int, default=MAX_FES)
    p.add_argument("--n-report-images", type=int, default=5,
                   help="number of images to render segmentation figures for (default: 5)")
    p.add_argument("--figure-k", nargs="+", type=int, default=[3, 7, 12],
                   help="K levels for the side-by-side optimizer comparison figures")
    p.add_argument("--table1-k", type=int, default=12,
                   help="K for the algorithms x objectives table (brief's Table 1 uses 12)")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel workers for the experiment grid (default: all cores)")
    p.add_argument("--force", action="store_true",
                   help="re-run combinations even if their result files already exist")
    p.add_argument("--skip-experiments", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--skip-summary", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-timing", action="store_true")
    p.add_argument("--skip-q-sweep", action="store_true")
    args = p.parse_args()
    if args.dataset == "all" and args.img_dir is not None:
        p.error("--img-dir needs a single --dataset")
    if args.n_report_images < 1:
        p.error("--n-report-images must be at least 1")
    return args


def run_experiments(args, out_dir: Path, n_images: int) -> None:
    # One call for the whole grid keeps every worker busy; run_experiments.py
    # skips (image, objective, K, optimizer) files that already exist.
    if args.force:
        for path in out_dir.glob(f"{args.dataset}__*.json"):
            path.unlink()
    cmd = [
        sys.executable, str(SCRIPTS / "run_experiments.py"),
        "--dataset", args.dataset,
        "--objective", "all",
        "--optimizer", "all",
        "--k", "all",
        "--n-images", str(n_images),
        "--runs", str(args.runs),
        "--max-fes", str(args.max_fes),
        "--out", str(out_dir),
    ]
    if args.workers:
        cmd += ["--workers", str(args.workers)]
    run(cmd)


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
    run([
        sys.executable, str(SCRIPTS / "summary_table.py"),
        "--results-dir", str(out_dir),
        "--dataset", args.dataset,
        "--objectives", *OBJECTIVES,
        "--optimizers", *OPTIMIZERS,
        "--k-levels", *[str(k) for k in K_LEVELS],
        "--n-images", str(n_images),
        "--stats",
    ])
    run([
        sys.executable, str(SCRIPTS / "scalability.py"),
        "--results-dir", str(out_dir),
        "--dataset", args.dataset,
    ])
    run([
        sys.executable, str(SCRIPTS / "report_tables.py"),
        "--results-dir", str(out_dir),
        "--dataset", args.dataset,
        "--table1-k", str(args.table1_k),
    ])


def run_timing(args, out_dir: Path) -> None:
    run([
        sys.executable, str(SCRIPTS / "timing.py"),
        "--dataset", args.dataset,
        "--max-fes", str(args.max_fes),
        "--out", str(out_dir / "timing"),
    ])


def run_q_sweep(args, out_dir: Path, n_images: int) -> None:
    run([
        sys.executable, str(SCRIPTS / "q_sweep.py"),
        "--dataset", args.dataset,
        "--n-images", str(n_images),
        "--max-fes", str(args.max_fes),
        "--out", str(out_dir / "q_sweep"),
    ])


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
                    "--maximize",
                ])
    print(f"Figures in {vis_out}")
    run([
        sys.executable, str(SCRIPTS / "compare_segmentation.py"),
        "--results-dir", str(out_dir),
        "--dataset", args.dataset,
        "--images", *image_names,
        "--k-levels", *[str(k) for k in args.figure_k],
    ])


def run_dataset(args) -> None:
    cfg = DATASETS[args.dataset]
    out_dir = args.out / args.dataset
    img_dir = args.img_dir or cfg["img_dir"]
    img_ext = (args.img_ext or cfg["img_ext"]).lstrip(".")
    n_images = args.n_images or cfg["n_images"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"===== dataset={args.dataset} -> {out_dir} =====")

    if not args.skip_experiments:
        run_experiments(args, out_dir, n_images)
    if not args.skip_plots:
        print("Generating convergence plots...")
        run_plots(args, out_dir)
    if not args.skip_summary:
        print("Generating summary tables, report tables and statistical tests...")
        run_summary(args, out_dir, n_images)
    if not args.skip_figures:
        print("Generating visual segmentation outputs (original vs. segmented)...")
        run_figures(args, out_dir, img_dir, img_ext)
    if not args.skip_timing:
        print("Timing optimizers on a single worker...")
        run_timing(args, out_dir)
    if not args.skip_q_sweep:
        print("Running Tsallis q-sensitivity sweep...")
        run_q_sweep(args, out_dir, n_images)


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(SCRIPTS / "parameters_table.py"), "--out", str(args.out)])
    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in datasets:
        run_dataset(argparse.Namespace(**{**vars(args), "dataset": name}))
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min. Results in {args.out}")


if __name__ == "__main__":
    main()