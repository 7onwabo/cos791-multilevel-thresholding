"""Master experiment script (Assignment section 4 deliverable).

Runs every combination of (dataset, image, objective, K, optimizer) for N_RUNS
statistical repeats.  Results include optimiser output (thresholds, fitness,
convergence history) and Phase-1 reconstruction metrics (PSNR, SSIM, U).

Uses multiprocessing to parallelise across CPU cores.

Usage:
    python scripts/run_experiments.py --dataset bds500 --objective otsu --k 3
    python scripts/run_experiments.py --dataset all --objective all --k all
    python scripts/run_experiments.py --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Ensure scripts/ is importable for run_one in spawned subprocesses (Windows).
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from run_one import run_one                          # noqa: E402
from thresholding.config import K_LEVELS, MAX_FES, N_RUNS  # noqa: E402
from thresholding.datasets import load_bds500, load_chaos   # noqa: E402
from thresholding.histogram import normalised_histogram      # noqa: E402
from thresholding.metrics import psnr, ssim, uniformity      # noqa: E402
from thresholding.objectives import OBJECTIVES               # noqa: E402
from thresholding.optimizers import OPTIMIZERS               # noqa: E402


# ---------------------------------------------------------------------------
# Worker function (must be at module level for Windows multiprocessing)
# ---------------------------------------------------------------------------

def _run_task(task: dict) -> dict:
    """Run all repeats for one (dataset, image, objective, K, optimizer) combo.

    Each task dict contains:
        hist, image, obj, opt, k, n_runs, max_fes, out_path, label

    Returns a summary dict for progress reporting.
    """
    runs = []
    for r in range(task["n_runs"]):
        result = run_one(task["hist"], task["obj"], task["opt"],
                         task["k"], r, task["max_fes"])

        # Compute Phase-1 reconstruction metrics on the original image
        t = result["thresholds"]
        p = psnr(task["image"], t)
        result["psnr"] = float(p) if np.isfinite(p) else 999.0
        result["ssim"] = float(ssim(task["image"], t))
        result["uniformity"] = float(uniformity(task["image"], t))
        runs.append(result)

    Path(task["out_path"]).write_text(json.dumps(runs))

    fits = [r["fitness"] for r in runs]
    return {
        "label": task["label"],
        "mean_fitness": float(np.mean(fits)),
        "std_fitness":  float(np.std(fits)),
        "mean_psnr":    float(np.mean([r["psnr"] for r in runs])),
        "mean_ssim":    float(np.mean([r["ssim"] for r in runs])),
        "mean_u":       float(np.mean([r["uniformity"] for r in runs])),
    }


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="COS791 multilevel thresholding experiments")
    p.add_argument("--dataset",
                   choices=["bds500", "chaos", "all"], default="all")
    p.add_argument("--objective",
                   choices=[*OBJECTIVES, "all"], default="all")
    p.add_argument("--optimizer",
                   choices=[*OPTIMIZERS, "all"], default="all")
    p.add_argument("--k", default="all",
                   help="single K value or 'all'")
    p.add_argument("--runs", type=int, default=N_RUNS)
    p.add_argument("--max-fes", type=int, default=MAX_FES)
    p.add_argument("--n-images", type=int, default=None,
                   help="use only the first N images (for quick tests)")
    p.add_argument("--out", default="results/raw")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel workers (default: number of CPU cores)")
    return p.parse_args()


def _selected(name: str, mapping: dict) -> list[str]:
    return list(mapping) if name == "all" else [name]


def main() -> None:
    args = parse_args()

    # --- Load datasets --------------------------------------------------------
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

    # --- Build task list (skip already-completed JSON files) -------------------
    tasks: list[dict] = []
    skipped = 0
    for ds_name, images in datasets.items():
        items = list(images.items())[: args.n_images]
        for img_name, img in items:
            hist = normalised_histogram(img)
            for obj in objectives:
                for k in ks:
                    for opt in optimizers:
                        path = out_dir / (
                            f"{ds_name}__{img_name}__{obj}__K{k}__{opt}.json")
                        if path.exists():
                            skipped += 1
                            continue
                        tasks.append({
                            "hist":     hist,
                            "image":    img,
                            "obj":      obj,
                            "opt":      opt,
                            "k":        k,
                            "n_runs":   args.runs,
                            "max_fes":  args.max_fes,
                            "out_path": str(path),
                            "label":    f"{ds_name}/{img_name} {obj} K={k} {opt}",
                        })

    n_workers = args.workers or os.cpu_count() or 1

    # --- Print grid summary ---------------------------------------------------
    print("=== Experiment grid ===")
    for ds_name, images in datasets.items():
        print(f"  dataset={ds_name}: {len(images)} images")
    print(f"  objectives = {objectives}")
    print(f"  optimizers = {optimizers}")
    print(f"  K levels   = {ks}")
    print(f"  runs/combo = {args.runs}")
    print(f"  -> {len(tasks)} tasks to run, {skipped} already done "
          f"({n_workers} workers)")
    print()

    if not tasks:
        print("All tasks already completed (JSON files exist).")
        return

    # --- Execute with multiprocessing -----------------------------------------
    wall_t0 = time.perf_counter()
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_task, t): t["label"] for t in tasks}
        for future in as_completed(futures):
            completed += 1
            try:
                s = future.result()
                print(
                    f"[{completed}/{len(tasks)}] {s['label']}:  "
                    f"fitness={s['mean_fitness']:.5f} +/- {s['std_fitness']:.5f}  "
                    f"PSNR={s['mean_psnr']:.2f}  "
                    f"SSIM={s['mean_ssim']:.4f}  "
                    f"U={s['mean_u']:.4f}"
                )
            except Exception as exc:
                failed += 1
                print(f"[{completed}/{len(tasks)}] FAILED "
                      f"{futures[future]}: {exc}")

    elapsed = time.perf_counter() - wall_t0
    m, sec = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\nDone. {completed - failed}/{len(tasks)} succeeded, "
          f"{failed} failed.  Wall time: {h}h {m}m {sec}s")
    print(f"Results saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
