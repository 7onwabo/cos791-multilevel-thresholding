#!/usr/bin/env python3
"""
Run the full set of experiments and generate plots and summary tables.

Run using: python run_script.py --out results/my_run --img-dir data/CHAOS --n-report-images 5
"""

from __future__ import annotations

import re
import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration (mirrors the bash script's variables)
# ---------------------------------------------------------------------------

OBJECTIVES = ["otsu", "kapur", "tsallis"]
OPTIMIZERS = ["SHADE", "L-SHADE"]
K_LEVELS = [3, 5, 7, 9, 11, 12]
N_IMAGES = 15
RUNS = 30
MAX_FES = 30000
OUT = Path("results/phase2_shade_vs_lshade")

# -- visual segmentation output settings -------------------------------------
# Point this at the folder holding the actual BSD500 image files (run_one
# works off in-memory arrays from load_bds500(), but visualize_segmentation.py
# needs a real file on disk to read via PIL).
IMG_DIR = Path("data/CHAOS")
IMG_EXT = "png"

# Result filename pattern written by run_experiments.py:
#   bds500__<image>__<objective>__K<k>__<optimizer>.json
_RESULT_RE = re.compile(r"^(?:bds500|chaos)__(.+)__([^_]+)__K(\d+)__(.+)\.json$")


def run(cmd: list[str]) -> None:
    """Run a subprocess, streaming output, raising on failure."""
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiments and generate reports")
    parser.add_argument("--out", type=Path, default=OUT,
                        help=f"results output directory (default: {OUT})")
    parser.add_argument("--img-dir", type=Path, default=IMG_DIR,
                        help=f"directory containing source images (default: {IMG_DIR})")
    parser.add_argument("--n-report-images", type=int, default=15,
                        help="number of representative images to render figures for (default: 15)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    img_dir = args.img_dir
    n_report_images = args.n_report_images
    vis_out = out_dir / "segmentation_figures"

    if n_report_images < 1:
        raise SystemExit("--n-report-images must be at least 1")

    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(OBJECTIVES) * len(K_LEVELS) * len(OPTIMIZERS)
    count = 0

    for obj in OBJECTIVES:
        for k in K_LEVELS:
            for opt in OPTIMIZERS:
                count += 1
                print(f"[{count}/{total}] objective={obj} K={k} optimizer={opt}")
                t0 = time.perf_counter()

                run([
                    sys.executable, "scripts/run_experiments.py",
                    "--dataset", "chaos",
                    "--objective", obj,
                    "--optimizer", opt,
                    "--k", str(k),
                    "--n-images", str(N_IMAGES),
                    "--runs", str(RUNS),
                    "--max-fes", str(MAX_FES),
                    "--out", str(out_dir),
                ])

                elapsed = time.perf_counter() - t0
                print(f"  -> {elapsed:.1f}s")

    print("Experiments done. Generating plots and summary table...")

    run([
        sys.executable, "scripts/convergence_plot.py",
        "--results-dir", str(out_dir),
        "--dataset", "chaos",
        "--objectives", *OBJECTIVES,
        "--optimizers", *OPTIMIZERS,
        "--k-levels", *[str(k) for k in K_LEVELS],
    ])

    run([
        sys.executable, "scripts/summary_table.py",
        "--results-dir", str(out_dir),
        "--dataset", "chaos",
        "--objectives", *OBJECTIVES,
        "--optimizers", *OPTIMIZERS,
        "--k-levels", *[str(k) for k in K_LEVELS],
    ])

    print("Generating visual segmentation outputs (original vs. segmented)...")
    vis_out.mkdir(parents=True, exist_ok=True)

    # Discover image names from the result filenames themselves.
    image_names: list[str] = []
    seen = set()
    for path in sorted(out_dir.glob("chaos*__*__K*__*.json")):
        m = _RESULT_RE.match(path.name)
        if not m:
            continue
        img_name = m.group(1)
        if img_name not in seen:
            seen.add(img_name)
            image_names.append(img_name)
        if len(image_names) >= n_report_images:
            break

    if not image_names:
        print(f"  no result files found under {out_dir} -- skipping figures")
        return

    for img_name in image_names:
        image_path = img_dir / f"{img_name}.{IMG_EXT}"
        if not image_path.is_file():
            print(f"  skipping {img_name}: image not found at {image_path} "
                  f"(check --img-dir; expected extension: {IMG_EXT})")
            continue

        for obj in OBJECTIVES:
            for opt in OPTIMIZERS:
                pattern = f"chaos__{img_name}__{obj}__K*__{opt}.json"
                matches = list(out_dir.glob(pattern))
                if not matches:
                    continue

                out_png = vis_out / f"{img_name}__{obj}__{opt}.png"
                print(f"  figure: {img_name} / {obj} / {opt} -> {out_png}")

                run([
                    sys.executable, "scripts/visualize_segmentation.py",
                    "--image", str(image_path),
                    "--results-glob", str(out_dir / pattern),
                    "--out", str(out_png),
                    "--title", f"{img_name} -- {obj} -- {opt}",
                ])

    print(f"Done. Results in {out_dir}, figures in {vis_out}")


if __name__ == "__main__":
    main()
