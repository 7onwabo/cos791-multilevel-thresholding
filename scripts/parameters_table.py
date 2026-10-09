"""Algorithm parameter settings table, read from the optimizer classes themselves.

Instantiates every registered optimizer at every K and records its population
size and control parameters, so the report's settings table always matches the
code that produced the results.

Outputs: <out>/parameters.{csv,tex}

Usage:
    python scripts/parameters_table.py --out results
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from thresholding.config import K_LEVELS, MAX_FES, N_RUNS, RANDOM_SEED, threshold_bounds
from thresholding.optimizers import OPTIMIZERS

STRATEGY = {
    "DE": "DE/rand/1/bin, fixed $F$, $CR$, greedy selection",
    "JADE": "DE/current-to-pbest/1/bin, external archive, adaptive $\\mu_F$, $\\mu_{CR}$",
    "SHADE": "DE/current-to-pbest/1/bin, archive, success-history memory",
    "L-SHADE": "SHADE + linear population size reduction",
    "LADE": "DE/rand/1/bin, dithered $F$, late-acceptance selection",
}
COMMON = {"fitness", "lower", "upper", "dim", "max_fes", "rng", "pop_size"}


def describe(name: str, cls) -> dict[str, str]:
    instances = {k: cls(lambda x: 0.0, threshold_bounds(k)) for k in K_LEVELS}
    sizes = [inst.pop_size for inst in instances.values()]
    params = {}
    for attr, value in vars(instances[K_LEVELS[0]]).items():
        if attr in COMMON or attr.startswith("_") or not np.isscalar(value):
            continue
        per_k = [getattr(inst, attr) for inst in instances.values()]
        if all(v == s for v, s in zip(per_k, sizes)) and len(set(per_k)) > 1:
            params[attr] = "$N$"
        elif len(set(per_k)) > 1:
            params[attr] = "/".join(str(v) for v in per_k)
        else:
            params[attr] = f"{value:g}" if isinstance(value, float) else str(value)
    return {
        "Algorithm": name,
        "Strategy": STRATEGY.get(name, ""),
        f"$N$ (K={'/'.join(map(str, K_LEVELS))})": "/".join(map(str, sizes)),
        "Parameters": ", ".join(f"{k}={v}" for k, v in params.items()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Algorithm parameter settings table")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [describe(name, cls) for name, cls in OPTIMIZERS.items()]
    header = list(rows[0])

    with open(args.out / "parameters.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    tex = [r"\begin{table}[htbp]", r"\centering", r"\small",
           rf"\caption{{Algorithm parameter settings. All algorithms: {MAX_FES} function "
           rf"evaluations, {N_RUNS} independent runs (seed {RANDOM_SEED} + run index), search "
           r"bounds $[1, 254]$, midpoint bound repair.}",
           r"\label{tab:parameters}", r"\begin{tabular}{lp{5.2cm}lp{4.2cm}}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        cells = [str(r[h]) for h in header]
        cells[-1] = cells[-1].replace("_", r"\_")
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (args.out / "parameters.tex").write_text("\n".join(tex))

    for r in rows:
        print(f"{r['Algorithm']:8s} N={r[header[2]]:24s} {r['Parameters']}")
    print(f"Saved {args.out / 'parameters.csv'} and parameters.tex")


if __name__ == "__main__":
    main()
