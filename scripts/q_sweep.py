"""Tsallis q-sensitivity sweep (Assignment section 4: impact of q vs Otsu/Kapur).

Runs one optimizer on Tsallis entropy for several q values, plus Otsu and
Kapur as q-independent references, on the same images, K levels and seeds.
Tsallis reduces to Kapur's Shannon entropy as q -> 1, so q = 1 is not run.

Outputs in --out (default results/<dataset>/q_sweep):
    raw/*.json            per-task runs (re-runs skip existing files)
    q_sweep_runs.csv      one row per run
    q_sweep_summary.csv   mean ± std over images per (objective, q, K)
    q_sweep.png           metric vs q, one line per K; dashed = Kapur, dotted = Otsu

Usage:
    python scripts/q_sweep.py --dataset bds500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from run_one import run_one                                   # noqa: E402
from thresholding.config import MAX_FES, RESULTS_DIR          # noqa: E402
from thresholding.datasets import load_bds500, load_chaos     # noqa: E402
from thresholding.histogram import normalised_histogram       # noqa: E402

METRICS = ("fitness", "psnr", "ssim", "uniformity", "class_separability")
PLOT_METRICS = ("psnr", "ssim", "uniformity", "class_separability")
LOADERS = {"bds500": load_bds500, "chaos": load_chaos}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tsallis q-sensitivity sweep")
    p.add_argument("--dataset", choices=sorted(LOADERS), default="bds500")
    p.add_argument("--qs", nargs="+", type=float, default=[0.3, 0.5, 0.8, 1.2, 1.5, 2.0])
    p.add_argument("--k-levels", nargs="+", type=int, default=[3, 7, 12])
    p.add_argument("--optimizer", default="L-SHADE")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--max-fes", type=int, default=MAX_FES)
    p.add_argument("--n-images", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _run_task(task: dict) -> str:
    runs = []
    for r in range(task["runs"]):
        res = run_one(task["image"], task["hist"], task["obj"], task["opt"],
                      task["k"], r, task["max_fes"], q=task["q"])
        runs.append({key: (float(res[key]) if np.isfinite(res[key]) else 999.0)
                     for key in ("fitness", "psnr", "ssim", "uniformity",
                                 "class_separability", "time")}
                    | {"thresholds": res["thresholds"]})
    Path(task["path"]).write_text(json.dumps(runs))
    return task["path"]


def build_tasks(args, images: dict, raw_dir: Path) -> list[dict]:
    variants = [("otsu", None), ("kapur", None)] + [("tsallis", q) for q in args.qs]
    tasks = []
    for img_name, img in images.items():
        hist = normalised_histogram(img)
        for k in args.k_levels:
            for obj, q in variants:
                tag = obj if q is None else f"tsallis_q{q:g}"
                path = raw_dir / f"{img_name}__{tag}__K{k}.json"
                if path.exists():
                    continue
                tasks.append({"image": img, "hist": hist, "obj": obj, "q": q, "k": k,
                              "opt": args.optimizer, "runs": args.runs,
                              "max_fes": args.max_fes, "path": str(path)})
    return tasks


def load_runs(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.glob("*.json")):
        img, tag, k = path.stem.split("__")
        obj, _, q = tag.partition("_q")
        for i, r in enumerate(json.loads(path.read_text())):
            rows.append({"image": img, "objective": obj, "q": float(q) if q else np.nan,
                         "K": int(k[1:]), "run": i, **{m: r[m] for m in (*METRICS, "time")}})
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["objective", "q", "K"]
    per_img = runs.groupby([*keys, "image"], dropna=False)[list(METRICS)].mean().reset_index()
    out = per_img.groupby(keys, dropna=False)[list(METRICS)].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def plot(summary: pd.DataFrame, out_path: Path, title: str) -> None:
    ts = summary[summary["objective"] == "tsallis"]
    ks = sorted(summary["K"].unique())
    colors = dict(zip(ks, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(4.0 * len(PLOT_METRICS), 3.3))
    for ax, m in zip(axes, PLOT_METRICS):
        for k in ks:
            g = ts[ts["K"] == k].sort_values("q")
            ax.plot(g["q"], g[f"{m}_mean"], marker="o", markersize=3,
                    color=colors[k], label=f"Tsallis K={k}")
            for obj, style in (("kapur", "--"), ("otsu", ":")):
                ref = summary[(summary["objective"] == obj) & (summary["K"] == k)]
                if not ref.empty:
                    ax.axhline(ref[f"{m}_mean"].iloc[0], color=colors[k],
                               linestyle=style, linewidth=1)
        ax.axvline(1.0, color="grey", linewidth=0.5)
        ax.set_xlabel("q")
        ax.set_title(m)
    axes[0].legend(fontsize=7, title="dashed: Kapur, dotted: Otsu", title_fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out or RESULTS_DIR / args.dataset / "q_sweep"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    images = dict(list(LOADERS[args.dataset]().items())[: args.n_images])
    tasks = build_tasks(args, images, raw_dir)
    print(f"q sweep: {len(images)} images, K={args.k_levels}, q={args.qs}, "
          f"{args.optimizer}, {args.runs} runs -> {len(tasks)} tasks to run")
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers or os.cpu_count()) as pool:
            futures = [pool.submit(_run_task, t) for t in tasks]
            for i, f in enumerate(as_completed(futures), 1):
                f.result()
                if i % 20 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}]", flush=True)

    runs = load_runs(raw_dir)
    summary = summarise(runs)
    runs.to_csv(out_dir / "q_sweep_runs.csv", index=False)
    summary.to_csv(out_dir / "q_sweep_summary.csv", index=False)
    plot(summary, out_dir / "q_sweep.png",
         f"Tsallis q sensitivity -- {args.dataset} -- {args.optimizer}")
    print(summary[["objective", "q", "K", "psnr_mean", "ssim_mean", "uniformity_mean"]]
          .round(4).to_string(index=False))
    print(f"Saved results in {out_dir}")


if __name__ == "__main__":
    main()
