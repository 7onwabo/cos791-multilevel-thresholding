#!/usr/bin/env python3
"""compare_shade_lshade_phase1.py

Baseline convergence validation: SHADE vs L-SHADE on BSD500, all 3 objectives.

Python port of compare_shade_lshade_phase1.sh so it runs the same way on
Windows/PowerShell as on Linux/macOS (no bash-only syntax: `find`, `sed`,
`mapfile`, `compgen`). Run it with:

    python compare_shade_lshade_phase1.py

from the project root (same place you'd have run the .sh version from).
"""

from __future__ import annotations

import re
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
N_REPORT_IMAGES = 15  # how many representative images to render figures for
VIS_OUT = OUT / "segmentation_figures"

# Result filename pattern written by run_experiments.py:
#   bds500__<image>__<objective>__K<k>__<optimizer>.json
_RESULT_RE = re.compile(r"^(?:bds500|chaos)__(.+)__([^_]+)__K(\d+)__(.+)\.json$")


def run(cmd: list[str]) -> None:
    """Run a subprocess, streaming output, raising on failure."""
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # total = len(OBJECTIVES) * len(K_LEVELS) * len(OPTIMIZERS)
    # count = 0

    # for obj in OBJECTIVES:
    #     for k in K_LEVELS:
    #         for opt in OPTIMIZERS:
    #             count += 1
    #             print(f"[{count}/{total}] objective={obj} K={k} optimizer={opt}")
    #             t0 = time.perf_counter()

    #             run([
    #                 sys.executable, "scripts/run_experiments.py",
    #                 "--dataset", "chaos",
    #                 "--objective", obj,
    #                 "--optimizer", opt,
    #                 "--k", str(k),
    #                 "--n-images", str(N_IMAGES),
    #                 "--runs", str(RUNS),
    #                 "--max-fes", str(MAX_FES),
    #                 "--out", str(OUT),
    #             ])

    #             elapsed = time.perf_counter() - t0
    #             print(f"  -> {elapsed:.1f}s")

    # print("Experiments done. Generating plots and summary table...")

    # run([
    #     sys.executable, "scripts/convergence_plot.py",
    #     "--results-dir", str(OUT),
    #     "--dataset", "chaos",
    #     "--objectives", *OBJECTIVES,
    #     "--optimizers", *OPTIMIZERS,
    #     "--k-levels", *[str(k) for k in K_LEVELS],
    # ])

    # run([
    #     sys.executable, "scripts/summary_table.py",
    #     "--results-dir", str(OUT),
    #     "--dataset", "chaos",
    #     "--objectives", *OBJECTIVES,
    #     "--optimizers", *OPTIMIZERS,
    #     "--k-levels", *[str(k) for k in K_LEVELS],
    # ])

    print("Generating visual segmentation outputs (original vs. segmented)...")
    VIS_OUT.mkdir(parents=True, exist_ok=True)

    # Discover image names from the result filenames themselves.
    image_names: list[str] = []
    seen = set()
    for path in sorted(OUT.glob("chaos*__*__K*__*.json")):
        m = _RESULT_RE.match(path.name)
        if not m:
            continue
        img_name = m.group(1)
        if img_name not in seen:
            seen.add(img_name)
            image_names.append(img_name)
        if len(image_names) >= N_REPORT_IMAGES:
            break

    if not image_names:
        print(f"  no result files found under {OUT} -- skipping figures")
        return

    for img_name in image_names:
        image_path = IMG_DIR / f"{img_name}.{IMG_EXT}"
        if not image_path.is_file():
            print(f"  skipping {img_name}: image not found at {image_path} "
                  f"(check IMG_DIR/IMG_EXT)")
            continue

        for obj in OBJECTIVES:
            for opt in OPTIMIZERS:
                pattern = f"chaos__{img_name}__{obj}__K*__{opt}.json"
                matches = list(OUT.glob(pattern))
                if not matches:
                    continue

                out_png = VIS_OUT / f"{img_name}__{obj}__{opt}.png"
                print(f"  figure: {img_name} / {obj} / {opt} -> {out_png}")

                run([
                    sys.executable, "scripts/visualize_segmentation.py",
                    "--image", str(image_path),
                    "--results-glob", str(OUT / pattern),
                    "--out", str(out_png),
                    "--title", f"{img_name} -- {obj} -- {opt}",
                ])

    print(f"Done. Results in {OUT}, figures in {VIS_OUT}")


if __name__ == "__main__":
    main()