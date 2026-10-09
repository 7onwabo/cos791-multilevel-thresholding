"""Plot DE-variant convergence curves, one figure per objective.

Curves are plotted against function evaluations, not generations: the variants
share an equal FE budget but spend it over very different generation counts
(LADE's small population runs ~10x more generations than SHADE), so a
generation axis would misrepresent their speed.

Raw fitness is image-dependent, so each run is expressed as relative error to
the best fitness any optimizer found on that image, (f* - f) / |f*|, and
averaged over images and runs on a log axis.

Usage:
    python scripts/convergence_plot.py --results-dir results/bds500
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thresholding.config import K_LEVELS
from thresholding.objectives import OBJECTIVES
from thresholding.optimizers import OPTIMIZERS

ERROR_FLOOR = 1e-9   # runs that reach f* exactly would otherwise vanish on a log axis
GRID_POINTS = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot convergence curves")
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--dataset", default="bds500")
    p.add_argument("--objectives", nargs="+", default=list(OBJECTIVES))
    p.add_argument("--optimizers", nargs="+", default=list(OPTIMIZERS))
    p.add_argument("--k-levels", nargs="+", type=int, default=list(K_LEVELS))
    p.add_argument(
        "--out",
        default=None,
        help="output PNG path prefix (default: <results-dir>/convergence_<objective>.png)",
    )
    return p.parse_args()


def load_runs_by_image(results_dir: Path, dataset: str, obj: str, k: int,
                       opt: str) -> dict[str, list[dict]]:
    prefix = f"{dataset}__"
    suffix = f"__{obj}__K{k}__{opt}.json"
    out: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(str(results_dir / f"{prefix}*{suffix}"))):
        img = Path(f).name[len(prefix):-len(suffix)]
        with open(f) as fh:
            out.setdefault(img, []).extend(json.load(fh))
    return out


def resample(runs: list[dict], grid: np.ndarray) -> np.ndarray:
    """Best-so-far curves interpolated onto a shared FE grid."""
    curves = []
    for r in runs:
        y = np.asarray(r["history"], dtype=float)
        x = r.get("evals_history")
        if x and len(x) == len(y):
            x = np.asarray(x, dtype=float)
        else:
            # Older result files lack evals_history; even spacing is exact only
            # for fixed-population variants.
            x = np.linspace(0, r["n_evals"], len(y))
        curves.append(np.interp(grid, x, y))
    return np.asarray(curves)


def relative_error_curves(per_opt: dict[str, dict[str, list[dict]]],
                          grid: np.ndarray) -> dict[str, np.ndarray]:
    """Relative error to each image's best-known fitness, rows = (image, run)."""
    resampled = {opt: {img: resample(runs, grid) for img, runs in by_img.items() if runs}
                 for opt, by_img in per_opt.items()}
    best_known: dict[str, float] = {}
    for by_img in resampled.values():
        for img, C in by_img.items():
            best_known[img] = max(best_known.get(img, -np.inf), float(C.max()))

    out = {}
    for opt, by_img in resampled.items():
        if not by_img:
            continue
        errs = [(best_known[img] - C) / max(abs(best_known[img]), 1e-12)
                for img, C in by_img.items()]
        out[opt] = np.maximum(np.vstack(errs), ERROR_FLOOR)
    return out


def main() -> None:
    args = parse_args()
    any_data = False
    for obj in args.objectives:
        fig, axes = plt.subplots(
            1,
            len(args.k_levels),
            figsize=(3.2 * len(args.k_levels), 3.0),
            squeeze=False,
        )
        objective_has_data = False
        for j, k in enumerate(args.k_levels):
            ax = axes[0][j]
            per_opt = {opt: load_runs_by_image(args.results_dir, args.dataset, obj, k, opt)
                       for opt in args.optimizers}
            pooled = [r for by_img in per_opt.values() for runs in by_img.values() for r in runs]
            if not pooled:
                continue
            grid = np.linspace(0, max(r["n_evals"] for r in pooled), GRID_POINTS)
            for opt, E in relative_error_curves(per_opt, grid).items():
                any_data = objective_has_data = True
                q25, q75 = np.percentile(E, [25, 75], axis=0)
                ax.plot(grid, E.mean(0), label=opt, linewidth=1.5)
                ax.fill_between(grid, q25, q75, alpha=0.15)
            ax.set_yscale("log")
            ax.set_title(f"K={k}")
            ax.set_xlabel("Function evaluations")
            ax.legend(fontsize=7)

        if not objective_has_data:
            plt.close(fig)
            continue
        axes[0][0].set_ylabel("Relative error to best known")
        fig.suptitle(f"Convergence at equal FE budget -- {args.dataset} -- {obj} "
                     f"(mean, IQR band)")
        fig.tight_layout()
        if args.out:
            requested_path = Path(args.out)
            out_path = requested_path.with_name(
                f"{requested_path.stem}_{obj}{requested_path.suffix or '.png'}"
            )
        else:
            out_path = args.results_dir / f"convergence_{obj}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    if not any_data:
        raise SystemExit(f"No result files found under {args.results_dir}. "
                         f"Check --dataset/--objectives/--optimizers/--k-levels match your run.")


if __name__ == "__main__":
    main()
