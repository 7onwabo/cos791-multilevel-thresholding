"""Report-ready tables (CSV + LaTeX booktabs) from summary.csv and stats/.

Written to <results-dir>/tables/:
    <metric>_<objective>.{csv,tex}  mean ± std across images; rows = optimizer,
                                    columns = K; best mean per K in bold
    table1_<metric>_K<k>.{csv,tex}  rows = optimizer, columns = objective at one K
                                    (the layout of the brief's Table 1)
    stats_overview.{csv,tex}        overall Friedman rank and Holm-corrected
                                    Wilcoxon W/T/L per optimizer and metric
    duplicate_thresholds.csv        share of runs whose final thresholds repeat a
                                    value (fewer than K distinct thresholds)

Usage:
    python scripts/report_tables.py --results-dir results/chaos --dataset chaos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thresholding.optimizers import OPTIMIZERS

METRICS = ("psnr", "ssim", "uniformity", "class_separability", "fitness")
LABELS = {"psnr": "PSNR (dB)", "ssim": "SSIM", "uniformity": "Uniformity $U$",
          "class_separability": r"Class separability $\eta$", "fitness": "Fitness"}
OBJ_LABELS = {"otsu": "Otsu", "kapur": "Kapur", "tsallis": r"Tsallis ($q=0.8$)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report-ready tables")
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--dataset", default=None, help="label used in captions")
    p.add_argument("--table1-k", type=int, default=12)
    return p.parse_args()


def fmt(metric: str, mu: float, sd: float) -> tuple[str, str]:
    """(plain, latex) strings for mean ± std."""
    if metric == "psnr":
        m, s = f"{mu:.2f}", f"{sd:.2f}"
    elif metric == "fitness":
        m, s = f"{mu:.6g}", f"{sd:.2g}"
    else:
        m, s = f"{mu:.4f}", f"{sd:.4f}"
    return f"{m} ± {s}", f"{m} $\\pm$ {s}"


def order_optimizers(names) -> list[str]:
    known = [o for o in OPTIMIZERS if o in set(names)]
    return known + sorted(set(names) - set(known))


def latex_table(df: pd.DataFrame, caption: str, label: str, index_name: str) -> str:
    cols = "l" + "c" * df.shape[1]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join([index_name, *map(str, df.columns)]) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        lines.append(" & ".join([str(idx), *map(str, row.values)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def grid_table(sub: pd.DataFrame, metric: str, row_key: str, col_key: str):
    """Plain and LaTeX tables of mean ± std, best (highest) mean per column in bold."""
    rows = order_optimizers(sub[row_key]) if row_key == "optimizer" else list(dict.fromkeys(sub[row_key]))
    cols = list(dict.fromkeys(sub.sort_values(col_key)[col_key])) if col_key == "K" \
        else list(dict.fromkeys(sub[col_key]))
    plain = pd.DataFrame(index=rows, columns=cols, dtype=object)
    tex = pd.DataFrame(index=rows, columns=cols, dtype=object)
    for c in cols:
        col = sub[sub[col_key] == c]
        best = col[f"{metric}_mu"].max()
        for _, r in col.iterrows():
            p_str, t_str = fmt(metric, r[f"{metric}_mu"], r[f"{metric}_sd"])
            plain.loc[r[row_key], c] = p_str
            tex.loc[r[row_key], c] = rf"\textbf{{{t_str}}}" if r[f"{metric}_mu"] == best else t_str
    return plain, tex


def write(plain: pd.DataFrame, tex: pd.DataFrame, path: Path, caption: str, index_name: str) -> None:
    plain.rename_axis(index_name).to_csv(path.with_suffix(".csv"))
    path.with_suffix(".tex").write_text(latex_table(tex, caption, f"tab:{path.stem}", index_name))


def stats_overview(stats_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    cells = {}
    for m in METRICS:
        fr, wtl = stats_dir / f"friedman_overall_{m}.csv", stats_dir / f"wilcoxon_wtl_{m}.csv"
        if not (fr.is_file() and wtl.is_file()):
            continue
        r = pd.read_csv(fr).query("objective == 'all'").set_index("optimizer")["avg_rank"]
        w = pd.read_csv(wtl).query("objective == 'all'").set_index("optimizer")
        cells[m] = {o: (r.get(o), f"{w.loc[o, 'wins']}/{w.loc[o, 'ties']}/{w.loc[o, 'losses']}")
                    for o in r.index if o in w.index}
    if not cells:
        return None
    opts = order_optimizers({o for c in cells.values() for o in c})
    plain = pd.DataFrame(index=opts, columns=[LABELS[m] for m in cells], dtype=object)
    tex = plain.copy()
    for m, c in cells.items():
        best = min(rank for rank, _ in c.values())
        for o, (rank, wtl) in c.items():
            plain.loc[o, LABELS[m]] = f"{rank:.2f} ({wtl})"
            s = f"{rank:.2f} ({wtl})"
            tex.loc[o, LABELS[m]] = rf"\textbf{{{s}}}" if rank == best else s
    return plain, tex


def main() -> None:
    args = parse_args()
    summary_path = args.results_dir / "summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"{summary_path} not found -- run summary_table.py first.")
    summary = pd.read_csv(summary_path)
    out = args.results_dir / "tables"
    out.mkdir(exist_ok=True)
    ds = args.dataset or args.results_dir.name
    metrics = [m for m in METRICS if f"{m}_mu" in summary]

    for m in metrics:
        for obj, sub in summary.groupby("objective", sort=False):
            plain, tex = grid_table(sub, m, "optimizer", "K")
            write(plain, tex, out / f"{m}_{obj}",
                  f"{LABELS[m]} (mean $\\pm$ std across images) on {ds}, "
                  f"{OBJ_LABELS.get(obj, obj)} objective. Best per $K$ in bold.", "Algorithm")

        at_k = summary[summary["K"] == args.table1_k]
        if not at_k.empty:
            plain, tex = grid_table(at_k, m, "optimizer", "objective")
            plain.columns = [OBJ_LABELS.get(c, c) for c in plain.columns]
            tex.columns = plain.columns
            write(plain, tex, out / f"table1_{m}_K{args.table1_k}",
                  f"{LABELS[m]} ($\\mu \\pm \\sigma$) on {ds} at $K = {args.table1_k}$.", "Algorithm")

    overview = stats_overview(args.results_dir / "stats")
    if overview:
        write(*overview, out / "stats_overview",
              f"Overall Friedman average rank (lower is better) and Holm-corrected Wilcoxon "
              f"wins/ties/losses on {ds}. Best rank per metric in bold.", "Algorithm")

    if "dup_rate" in summary:
        dup = summary.pivot_table(index="optimizer", columns="K", values="dup_rate", aggfunc="mean")
        dup.reindex(order_optimizers(dup.index)).round(4).to_csv(out / "duplicate_thresholds.csv")

    print(f"Saved report tables in {out}")


if __name__ == "__main__":
    main()
