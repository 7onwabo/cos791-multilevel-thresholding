"""Summary tables and statistical tests for multilevel thresholding experiments.

Aggregation (per objective, K, optimizer):
    1. For each image: mean and std over the independent runs  -> *_per_image.csv
    2. Across images: mean of the per-image means, and the std of the
       per-image means (ddof=1)                               -> summary.csv
       Also the average run-to-run std (stability)            -> *_runstd

Optionally (--stats), for each metric (higher = better), written to <out>/stats/:
    friedman_ranks_<m>.csv    Friedman average ranks per (objective, K), paired over images
    friedman_overall_<m>.csv  Friedman over all (objective, K, image) blocks, per objective and overall
    wilcoxon_<m>.csv          Wilcoxon signed-rank for every optimizer pair per (objective, K),
                              paired over images, Holm-corrected within each (objective, K)
    wilcoxon_wtl_<m>.csv      Win/tie/loss counts from the Holm-corrected Wilcoxon tests
    ranksum_wtl_<m>.csv       Per-image Wilcoxon rank-sum over the independent runs (as in
                              Lecture 7), counted as win/tie/loss per optimizer pair

Usage:
    python scripts/summary_table.py --results-dir results/bds500 --stats
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from thresholding.config import K_LEVELS
from thresholding.objectives import OBJECTIVES
from thresholding.optimizers import OPTIMIZERS

METRICS = ("psnr", "ssim", "uniformity", "class_separability")
STATS_METRICS = ("fitness", *METRICS)
GROUP = ["objective", "K", "optimizer"]
ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise optimisation results")
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--dataset", default="bds500")
    p.add_argument("--objectives", nargs="+", default=list(OBJECTIVES))
    p.add_argument("--optimizers", nargs="+", default=list(OPTIMIZERS))
    p.add_argument("--k-levels", nargs="+", type=int, default=list(K_LEVELS))
    p.add_argument("--n-images", type=int, default=None,
                   help="expected number of images (10 for BSD500, 15 for CHAOS); "
                        "default: the maximum found")
    p.add_argument("--run-ddof", type=int, default=1,
                   help="ddof for the run-to-run std within an image (default 1)")
    p.add_argument("--stats", action="store_true",
                   help="also write Friedman, Wilcoxon and rank-sum results")
    p.add_argument("--stats-metrics", nargs="+", default=list(STATS_METRICS),
                   choices=STATS_METRICS,
                   help="metrics used for the statistical tests (default: all)")
    p.add_argument("--out", default=None,
                   help="output CSV path (default: <results-dir>/summary.csv)")
    return p.parse_args()


def load_runs(args: argparse.Namespace) -> pd.DataFrame:
    """One row per (image, objective, K, optimizer, run)."""
    rows = []
    for obj in args.objectives:
        for k in args.k_levels:
            for opt in args.optimizers:
                pattern = str(args.results_dir / f"{args.dataset}__*__{obj}__K{k}__{opt}.json")
                for filename in sorted(glob.glob(pattern)):
                    image = Path(filename).stem.split("__")[1]
                    with open(filename) as f:
                        runs = json.load(f)
                    for i, r in enumerate(runs):
                        row = {"image": image, "objective": obj, "K": k, "optimizer": opt,
                               "run": i, "fitness": r["fitness"], "time": r["time"],
                               "n_evals": r["n_evals"],
                               "duplicate_thresholds": len(set(r["thresholds"])) < len(r["thresholds"])}
                        row.update({m: r[m] for m in METRICS if m in r})
                        rows.append(row)
    return pd.DataFrame(rows)


def build_per_image(runs: pd.DataFrame, ddof: int) -> pd.DataFrame:
    """One row per (image, objective, K, optimizer) with stats over the runs."""
    g = runs.groupby(["image", *GROUP], sort=False)
    per_img = g.agg(
        n_runs=("run", "count"),
        mean_fitness=("fitness", "mean"),
        std_fitness=("fitness", lambda s: s.std(ddof=ddof)),
        mean_time_ms=("time", lambda s: s.mean() * 1000),
        mean_n_evals=("n_evals", "mean"),
        dup_rate=("duplicate_thresholds", "mean"),
    )
    for m in METRICS:
        if m in runs:
            per_img[f"mean_{m}"] = g[m].mean()
            per_img[f"std_{m}"] = g[m].std(ddof=ddof)
    return per_img.reset_index()


def build_summary(per_img: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image means across images (NOT pooled over runs)."""
    agg = {
        "n_images": ("image", "nunique"),
        "n_runs_per_image": ("n_runs", "mean"),
        "mean_time_ms": ("mean_time_ms", "mean"),
        "std_time_ms": ("mean_time_ms", "std"),      # across images
        "mean_n_evals": ("mean_n_evals", "mean"),
        "dup_rate": ("dup_rate", "mean"),   # share of runs with repeated thresholds
    }
    for m in ("fitness", *METRICS):
        if f"mean_{m}" in per_img:
            agg[f"{m}_mu"] = (f"mean_{m}", "mean")    # mean over images
            agg[f"{m}_sd"] = (f"mean_{m}", "std")     # std across images (ddof=1)
            agg[f"{m}_runstd"] = (f"std_{m}", "mean") # avg run-to-run std
    summary = per_img.groupby(GROUP, sort=False).agg(**agg).reset_index()

    # Ready-to-paste "mu ± sd" strings for the report tables
    for m in ("fitness", *METRICS):
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
    print(f"Completeness: {len(summary) - len(bad)}/{len(summary)} combos have "
          f"all {expected} images.")


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    if np.allclose(a - b, 0):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def _ranks(wide: pd.DataFrame) -> np.ndarray:
    """Rank 1 = best (highest) within each row; ties get the average rank."""
    from scipy.stats import rankdata
    return np.array([rankdata(-row) for row in wide.to_numpy()])


def _friedman_p(wide: pd.DataFrame) -> float:
    from scipy.stats import friedmanchisquare
    if wide.shape[1] < 3:
        return np.nan
    try:
        return float(friedmanchisquare(*[wide[o].to_numpy() for o in wide.columns]).pvalue)
    except ValueError:  # all values identical
        return 1.0


def _outcome(p: float, x: np.ndarray, y: np.ndarray) -> str:
    """'A_better' / 'B_better' when significant at ALPHA, else 'tie'."""
    if p >= ALPHA:
        return "tie"
    diff = float(np.median(x) - np.median(y)) or float(np.mean(x) - np.mean(y))
    return "tie" if diff == 0 else ("A_better" if diff > 0 else "B_better")


def _wtl(rows: pd.DataFrame, a_col: str, b_col: str, outcome_col: str) -> pd.DataFrame:
    """Per-optimizer win/tie/loss counts, overall and per objective."""
    records = []
    for _, r in rows.iterrows():
        o = r[outcome_col]
        records.append((r["objective"], r[a_col], r[b_col], o == "A_better", o == "tie", o == "B_better"))
        records.append((r["objective"], r[b_col], r[a_col], o == "B_better", o == "tie", o == "A_better"))
    df = pd.DataFrame(records, columns=["objective", "optimizer", "opponent", "wins", "ties", "losses"])
    per_obj = df.groupby(["objective", "optimizer"])[["wins", "ties", "losses"]].sum().reset_index()
    overall = df.groupby("optimizer")[["wins", "ties", "losses"]].sum().reset_index()
    overall.insert(0, "objective", "all")
    out = pd.concat([per_obj, overall], ignore_index=True)
    out["net"] = out["wins"] - out["losses"]
    return out


def friedman_and_wilcoxon(per_img: pd.DataFrame, metric: str, out_dir: Path) -> None:
    col = f"mean_{metric}"
    rank_rows, wil_rows, blocks = [], [], []
    for (obj, k), sub in per_img.groupby(["objective", "K"]):
        wide = sub.pivot(index="image", columns="optimizer", values=col).dropna()
        if wide.shape[0] < 3 or wide.shape[1] < 2:
            continue
        opts = list(wide.columns)
        blocks.append((obj, wide))

        avg_rank = dict(zip(opts, _ranks(wide).mean(axis=0)))
        p_fried = _friedman_p(wide)
        for o in opts:
            rank_rows.append({"objective": obj, "K": k, "optimizer": o,
                              "avg_rank": avg_rank[o], "n_images": len(wide),
                              "friedman_p": p_fried})

        pairs = list(itertools.combinations(opts, 2))
        pvals = [_paired_wilcoxon(wide[a].to_numpy(), wide[b].to_numpy()) for a, b in pairs]
        for (a, b), p, p_adj in zip(pairs, pvals, holm(pvals)):
            x, y = wide[a].to_numpy(), wide[b].to_numpy()
            wil_rows.append({"objective": obj, "K": k, "A": a, "B": b,
                             "median_diff_A_minus_B": float(np.median(x - y)),
                             "p_value": p, "p_holm": p_adj,
                             "outcome": _outcome(p_adj, x, y)})

    if not rank_rows:
        print(f"Not enough paired data for statistics on {metric}")
        return

    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(out_dir / f"friedman_ranks_{metric}.csv", index=False)

    overall_rows = []
    scopes = {obj: [w for o, w in blocks if o == obj] for obj in dict.fromkeys(o for o, _ in blocks)}
    scopes["all"] = [w for _, w in blocks]
    for scope, wides in scopes.items():
        common = sorted(set.intersection(*(set(w.columns) for w in wides)))
        stacked = pd.concat([w[common] for w in wides], ignore_index=True)
        r = _ranks(stacked).mean(axis=0)
        p = _friedman_p(stacked)
        for o, rank in zip(common, r):
            overall_rows.append({"objective": scope, "optimizer": o, "avg_rank": rank,
                                 "n_blocks": len(stacked), "friedman_p": p})
    overall = pd.DataFrame(overall_rows).sort_values(["objective", "avg_rank"])
    overall.to_csv(out_dir / f"friedman_overall_{metric}.csv", index=False)

    wil = pd.DataFrame(wil_rows)
    wil.to_csv(out_dir / f"wilcoxon_{metric}.csv", index=False)
    _wtl(wil, "A", "B", "outcome").to_csv(out_dir / f"wilcoxon_wtl_{metric}.csv", index=False)

    best = overall[overall["objective"] == "all"].iloc[0]
    print(f"  {metric:18s} overall Friedman best: {best['optimizer']} "
          f"(avg rank {best['avg_rank']:.2f}, p={best['friedman_p']:.2g})")


def ranksum_wtl(runs: pd.DataFrame, metric: str, out_dir: Path) -> None:
    """Per-image Wilcoxon rank-sum over the independent runs, as win/tie/loss."""
    from scipy.stats import ranksums

    rows = []
    for (img, obj, k), sub in runs.groupby(["image", "objective", "K"]):
        by_opt = {o: g[metric].to_numpy() for o, g in sub.groupby("optimizer")}
        for a, b in itertools.combinations(sorted(by_opt), 2):
            x, y = by_opt[a], by_opt[b]
            if len(x) < 2 or len(y) < 2:
                continue
            p = 1.0 if np.allclose(np.r_[x, y], x[0]) else float(ranksums(x, y).pvalue)
            rows.append({"image": img, "objective": obj, "K": k, "A": a, "B": b,
                         "p_value": p, "outcome": _outcome(p, x, y)})
    if rows:
        _wtl(pd.DataFrame(rows), "A", "B", "outcome").to_csv(
            out_dir / f"ranksum_wtl_{metric}.csv", index=False)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else args.results_dir / "summary.csv"

    runs = load_runs(args)
    if runs.empty:
        raise SystemExit(f"No result files found under {args.results_dir}. "
                         "Check --dataset/--objectives/--optimizers/--k-levels "
                         "match the JSON filenames (case-sensitive).")

    per_img = build_per_image(runs, args.run_ddof)
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
        stats_dir = out_path.parent / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        for metric in args.stats_metrics:
            if metric not in runs:
                continue
            friedman_and_wilcoxon(per_img, metric, stats_dir)
            ranksum_wtl(runs, metric, stats_dir)
        print(f"Saved statistical tests in {stats_dir}")


if __name__ == "__main__":
    main()
