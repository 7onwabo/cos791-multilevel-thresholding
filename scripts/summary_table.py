"""Summary tables for multilevel thresholding experiments.

Aggregation (per objective, K, optimizer):
    1. For each image: mean and std over the independent runs  -> *_per_image.csv
    2. Across images: mean of the per-image means, and the std of the
       per-image means (ddof=1)                               -> summary.csv
       Also the average run-to-run std (stability)            -> *_runstd

Optionally (--stats): Friedman average ranks and Wilcoxon signed-rank tests
(best-ranked optimizer vs each other), paired over images.

Usage:
    python summary_table.py --results-dir results/phase1 \
        --optimizers de jade shade lshade lade --stats
"""
from __future__ import annotations

import argparse
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("psnr", "ssim", "uniformity", "class_separability")
GROUP = ["objective", "K", "optimizer"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise optimisation results")
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--dataset", default="bds500")
    p.add_argument("--objectives", nargs="+", default=["otsu", "kapur", "tsallis"])
    p.add_argument("--optimizers", nargs="+",
                   default=["shade", "lshade"])
    p.add_argument("--k-levels", nargs="+", type=int, default=[3, 5, 7, 9, 11, 12])
    p.add_argument("--n-images", type=int, default=None,
                   help="expected number of images (10 for BSD500, 15 for CHAOS); "
                        "default: the maximum found")
    p.add_argument("--run-ddof", type=int, default=1,
                   help="ddof for the run-to-run std within an image (default 1)")
    p.add_argument("--stats", action="store_true",
                   help="also write Friedman ranks and Wilcoxon tests")
    p.add_argument("--stats-metric", default="psnr", choices=METRICS,
                   help="metric used for the statistical tests (higher = better)")
    p.add_argument("--out", default=None,
                   help="output CSV path (default: <results-dir>/summary.csv)")
    return p.parse_args()


def load_image_runs(results_dir: Path, dataset: str, obj: str, k: int,
                    opt: str) -> list[tuple[str, list[dict]]]:
    """Load runs grouped by image (one JSON file = one image)."""
    pattern = str(results_dir / f"{dataset}__*__{obj}__K{k}__{opt}.json")
    out = []
    for filename in sorted(glob.glob(pattern)):
        path = Path(filename)
        image = path.stem.split("__")[1]
        with path.open() as f:
            out.append((image, json.load(f)))
    return out


def build_per_image(args: argparse.Namespace) -> pd.DataFrame:
    """One row per (image, objective, K, optimizer) with stats over the runs."""
    rows = []
    for obj in args.objectives:
        for k in args.k_levels:
            for opt in args.optimizers:
                for image, runs in load_image_runs(args.results_dir, args.dataset,
                                                   obj, k, opt):
                    if not runs:
                        continue
                    row = {
                        "image": image, "objective": obj, "K": k, "optimizer": opt,
                        "n_runs": len(runs),
                        "mean_fitness": np.mean([r["fitness"] for r in runs]),
                        "std_fitness": np.std([r["fitness"] for r in runs],
                                              ddof=args.run_ddof),
                        "mean_time_ms": np.mean([r["time"] for r in runs]) * 1000,
                        "mean_n_evals": np.mean([r["n_evals"] for r in runs]),
                    }
                    for m in METRICS:
                        vals = [r[m] for r in runs if m in r]
                        if vals:
                            row[f"mean_{m}"] = np.mean(vals)
                            row[f"std_{m}"] = np.std(vals, ddof=args.run_ddof)
                    rows.append(row)
    return pd.DataFrame(rows)


def build_summary(per_img: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image means across images (NOT pooled over runs)."""
    agg = {
        "n_images": ("image", "nunique"),
        "n_runs_per_image": ("n_runs", "mean"),
        "mean_time_ms": ("mean_time_ms", "mean"),
        "std_time_ms": ("mean_time_ms", "std"),      # across images
        "mean_n_evals": ("mean_n_evals", "mean"),
    }
    for m in METRICS:
        if f"mean_{m}" in per_img:
            agg[f"{m}_mu"] = (f"mean_{m}", "mean")    # mean over images
            agg[f"{m}_sd"] = (f"mean_{m}", "std")     # std across images (ddof=1)
            agg[f"{m}_runstd"] = (f"std_{m}", "mean") # avg run-to-run std
    summary = per_img.groupby(GROUP, sort=False).agg(**agg).reset_index()

    # Ready-to-paste "mu ± sd" strings for the report tables
    for m in METRICS:
        if f"{m}_mu" in summary:
            summary[f"{m}_fmt"] = (summary[f"{m}_mu"].map("{:.4f}".format) + " ± "
                                   + summary[f"{m}_sd"].map("{:.4f}".format))
    return summary


def check_completeness(summary: pd.DataFrame, expected: int | None) -> None:
    expected = expected or int(summary["n_images"].max())
    bad = summary[summary["n_images"] != expected]
    if not bad.empty:
        warnings.warn(
            f"{len(bad)} (objective, K, optimizer) combos do not have {expected} images:\n"
            + bad[GROUP + ["n_images"]].to_string(index=False))
    # Combos that were requested but have no files at all
    print(f"Completeness: {len(summary) - len(bad)}/{len(summary)} combos have "
          f"all {expected} images.")


def friedman_and_wilcoxon(per_img: pd.DataFrame, metric: str, out_dir: Path) -> None:
    """Paired tests over images for each (objective, K). Higher metric = better."""
    from scipy.stats import friedmanchisquare, rankdata, wilcoxon

    col = f"mean_{metric}"
    rank_rows, wil_rows = [], []
    for (obj, k), sub in per_img.groupby(["objective", "K"]):
        wide = sub.pivot(index="image", columns="optimizer", values=col).dropna()
        if wide.shape[0] < 3 or wide.shape[1] < 2:
            continue
        opts = list(wide.columns)
        # rank 1 = best (highest) within each image; ties get average rank
        ranks = np.array([rankdata(-row.values) for _, row in wide.iterrows()])
        avg_rank = dict(zip(opts, ranks.mean(axis=0)))

        p_fried = np.nan
        if len(opts) >= 3:
            try:
                p_fried = friedmanchisquare(*[wide[o].values for o in opts]).pvalue
            except ValueError:  # all values identical
                p_fried = 1.0
        for o in opts:
            rank_rows.append({"objective": obj, "K": k, "optimizer": o,
                              "avg_rank": avg_rank[o], "n_images": len(wide),
                              "friedman_p": p_fried})

        best = min(avg_rank, key=avg_rank.get)
        for o in opts:
            if o == best:
                continue
            diff = wide[best].values - wide[o].values
            if np.allclose(diff, 0):
                p, outcome = 1.0, "tie"
            else:
                p = wilcoxon(wide[best].values, wide[o].values).pvalue
                outcome = ("best_wins" if p < 0.05 and diff.mean() > 0 else
                           "best_loses" if p < 0.05 else "tie")
            wil_rows.append({"objective": obj, "K": k, "best": best, "other": o,
                             "p_value": p, "outcome_at_0.05": outcome})

    pd.DataFrame(rank_rows).to_csv(out_dir / f"friedman_ranks_{metric}.csv", index=False)
    pd.DataFrame(wil_rows).to_csv(out_dir / f"wilcoxon_vs_best_{metric}.csv", index=False)
    print(f"Saved Friedman/Wilcoxon results for {metric} in {out_dir}")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else args.results_dir / "summary.csv"

    per_img = build_per_image(args)
    if per_img.empty:
        raise SystemExit(f"No result files found under {args.results_dir}. "
                         "Check --dataset/--objectives/--optimizers/--k-levels "
                         "match the JSON filenames (case-sensitive).")

    summary = build_summary(per_img)
    check_completeness(summary, args.n_images)

    per_image_path = out_path.with_name(f"{out_path.stem}_per_image{out_path.suffix}")
    summary.to_csv(out_path, index=False)
    per_img.to_csv(per_image_path, index=False)

    for _, r in summary.iterrows():
        print(f"{r['objective']:8s} K={int(r['K']):2d} {r['optimizer']:7s} "
              f"(n_images={int(r['n_images'])}, {r['mean_time_ms']:.1f} ms/run)")
        print("  " + "  ".join(f"{m}={r[f'{m}_fmt']}" for m in METRICS
                               if f"{m}_fmt" in r))

    print(f"\nSaved {out_path}")
    print(f"Saved {per_image_path}")

    if args.stats:
        friedman_and_wilcoxon(per_img, args.stats_metric, out_path.parent)


if __name__ == "__main__":
    main()