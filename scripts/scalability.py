"""Computational time vs K: table and plot (Assignment section 4, scalability).

Reads summary_per_image.csv written by summary_table.py. Time is the
optimiser's wall time per run (``run_one`` excludes metric computation),
averaged over runs per image, then mean ± std across images.

Outputs in --results-dir:
    time_vs_k.csv   one row per (objective, optimizer, K) with mean/std ms per run
    time_vs_k.png   one panel per objective, one line per optimizer

Usage:
    python scripts/scalability.py --results-dir results/bds500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time vs K scalability table and plot")
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--dataset", default=None, help="label for the plot title")
    return p.parse_args()


def time_table(per_img: pd.DataFrame) -> pd.DataFrame:
    t = (per_img.groupby(["objective", "optimizer", "K"], sort=False)["mean_time_ms"]
         .agg(time_ms_mean="mean", time_ms_std="std", n_samples="count")
         .reset_index())
    # Growth relative to each (objective, optimizer)'s smallest K
    base = t.sort_values("K").groupby(["objective", "optimizer"])["time_ms_mean"].transform("first")
    t["ratio_vs_min_K"] = t["time_ms_mean"] / base
    t["time_fmt"] = t["time_ms_mean"].map("{:.1f}".format) + " ± " + t["time_ms_std"].map("{:.1f}".format)
    return t.sort_values(["objective", "optimizer", "K"])


def plot(table: pd.DataFrame, out_path: Path, title: str) -> None:
    objectives = list(dict.fromkeys(table["objective"]))
    fig, axes = plt.subplots(1, len(objectives), figsize=(4.2 * len(objectives), 3.2),
                             squeeze=False, sharey=True)
    for ax, obj in zip(axes[0], objectives):
        sub = table[table["objective"] == obj]
        for opt, g in sub.groupby("optimizer", sort=False):
            g = g.sort_values("K")
            ax.errorbar(g["K"], g["time_ms_mean"], yerr=g["time_ms_std"],
                        marker="o", markersize=3, capsize=2, linewidth=1.2, label=opt)
        ax.set_title(obj)
        ax.set_xlabel("K (number of thresholds)")
        ax.set_xticks(sorted(sub["K"].unique()))
    axes[0][0].set_ylabel("Time per run (ms)")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    src = args.results_dir / "summary_per_image.csv"
    if not src.is_file():
        raise SystemExit(f"{src} not found -- run summary_table.py first.")
    table = time_table(pd.read_csv(src))
    csv_path = args.results_dir / "time_vs_k.csv"
    png_path = args.results_dir / "time_vs_k.png"
    table.to_csv(csv_path, index=False)
    plot(table, png_path, f"Computational time vs K -- {args.dataset or args.results_dir.name}")
    print(table.pivot_table(index=["objective", "optimizer"], columns="K",
                            values="time_ms_mean").round(1).to_string())
    print(f"Saved {csv_path}\nSaved {png_path}")


if __name__ == "__main__":
    main()
