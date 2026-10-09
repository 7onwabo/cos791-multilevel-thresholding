# COS791 — Multilevel Image Thresholding via Differential Evolution

Group assignment: implement, analyse and empirically evaluate Differential
Evolution (DE) and its variants for multilevel image thresholding, using **Otsu**,
**Kapur** and **Tsallis** entropy as objective functions.

- **Phase 1 (BDS500):** 10 standard test images — reconstruction metrics (PSNR, SSIM, U).
- **Phase 2 (CHAOS):** 15 abdominal MRI slices — unsupervised class separability η = σ²_B / σ²_T.

---

## Quick start

**Prerequisite:** Python **3.12** (pinned in `.python-version`; 3.11+ works).

### macOS / Linux (zsh, bash)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `import thresholding` work
pytest                    # full suite, ~15 s
```

### Windows (PowerShell)
```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest
```

> First time on Windows PowerShell? If activation is blocked, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

---

## Reproducing all results

One command runs both phases and writes every table and figure used in the report:

```bash
python run_script.py
```

Runtime is roughly **2–2.5 h on 12 cores** (experiment grid ≈ 2 h; timing and the
q sweep add ≈ 10 min per dataset). Results are deterministic: run *r* uses seed
`42 + r` for every optimizer, image, objective and K.

Useful variants:

```bash
python run_script.py --dataset chaos                    # one phase only
python run_script.py --n-images 2 --runs 3 --max-fes 1000   # quick smoke run (~3 min)
python run_script.py --skip-experiments --skip-timing --skip-q-sweep   # rebuild tables/plots
python run_script.py --force                            # discard cached results, re-run all
```

Finished (image, objective, K, optimizer) result files are reused on re-runs, so an
interrupted run resumes where it stopped. Result files written by older versions of
the code are detected and refused; re-run with `--force` (or delete `results/`).

### Experimental protocol
| Setting | Value |
|---|---|
| Threshold levels | K ∈ {3, 5, 7, 9, 11, 12} |
| Objectives | Otsu, Kapur, Tsallis (q = 0.8) |
| Algorithms | DE (rand/1/bin), JADE, SHADE, L-SHADE, LADE |
| Runs | 30 independent runs per (image, objective, K, algorithm) |
| Stopping criterion | 30 000 function evaluations, identical for all algorithms |
| Statistics | Friedman ranks; Wilcoxon signed-rank (paired over images, Holm-corrected, α = 0.05); per-image Wilcoxon rank-sum over runs |

Algorithm parameters are written to `results/parameters.{csv,tex}` straight from the
optimizer classes.

### Outputs (`results/<dataset>/`)
```
summary.csv, summary_per_image.csv   mean ± std of fitness, PSNR, SSIM, U, η, time per (objective, K, algorithm)
tables/          report-ready CSV + LaTeX: <metric>_<objective> (algorithms x K),
                 table1_<metric>_K12 (algorithms x objectives), stats_overview,
                 duplicate_thresholds
stats/           Friedman ranks, Wilcoxon pairs + W/T/L, rank-sum W/T/L, per metric
convergence_<objective>.png          relative error vs function evaluations
time_vs_k.{csv,png}                  time vs K from the parallel run
timing/          single-worker time vs K (use these for absolute times)
comparison_figures/<image>__K<k>.png all algorithms side by side, best run each
segmentation_figures/                original vs segmented, one algorithm per figure
q_sweep/         Tsallis q sensitivity vs Otsu/Kapur (L-SHADE, K ∈ {3, 7, 12})
<dataset>__<image>__<objective>__K<k>__<algorithm>.json   raw runs
```

Each stage is also a standalone script in `scripts/` (`run_experiments.py`,
`summary_table.py`, `report_tables.py`, `convergence_plot.py`, `scalability.py`,
`timing.py`, `compare_segmentation.py`, `visualize_segmentation.py`, `q_sweep.py`,
`parameters_table.py`); run any with `--help`.

---

## Repository layout
```
data/            BDS500 (imgN.png + imgN_gt.png) and CHAOS MRI slices
docs/            assignment brief + DE background
src/thresholding/
  config.py      K levels, runs, FE budget, seeds, paths            [§1.1, §3]
  histogram.py   image -> normalised 256-bin histogram
  datasets.py    load_bds500(), load_chaos()
  objectives/    otsu.py, kapur.py, tsallis.py + OBJECTIVES registry [§1.2]
  optimizers/    de, jade, shade, lshade, lade + OPTIMIZERS registry [§1.3]
  metrics/       PSNR, SSIM, uniformity U, class separability η      [§2]
run_script.py    master script: reproduces every result               [§4]
scripts/         experiment, statistics, table and figure stages
tests/           pytest suite
results/         generated outputs (gitignored)
```

## Implementation notes
- **Objectives** maximise; thresholds are rounded and sorted, and a pixel equal to
  t_k belongs to the lower class (as in the Lecture 7 worked examples). Verified
  against the Lecture 7 Otsu and Kapur examples.
- **Tsallis** combines classes pseudo-additively, S = Σ S_k + (1 − q) Π S_k. For
  q < 1 the product term dominates at high K; see `q_sweep/`.
- **PSNR/SSIM** use the 8-bit peak 255 (PSNR = 20 log₁₀(255/RMSE); SSIM
  c₁ = (0.01·255)², c₂ = (0.03·255)²). **U** follows Sahoo et al. (1988) with the
  2c factor, c = number of thresholds.
- **Convergence** is plotted against function evaluations, not generations: the
  variants spend the same budget over very different generation counts.
- **L-SHADE** follows Tanabe & Fukunaga (2014), including the weighted Lehmer mean
  and terminal value for the CR memory.
- **LADE** keeps a per-individual late-acceptance memory of length L whose slots only
  improve; its population size (max(10, K)) was tuned on these objectives, whereas
  the other variants use literature defaults.
- The brief's Table 1 lists CoDE and SaDE, but the mark rubric names the five
  variants implemented here.

---

## Work split (3 developers)
| Dev | DE variants | Also owns |
|-----|-------------|-----------|
| A   | Standard DE (`DE/rand/1/bin`), JADE | Phase-1 reconstruction metrics (PSNR/SSIM/U) |
| B   | SHADE, L-SHADE (LPSR)               | Phase-2 unsupervised class separability (η)  |
| C   | LADE (Late Acceptance DE)           | Stats (Wilcoxon/Friedman), convergence plots, report |

Each variant subclasses `BaseOptimizer` in `src/thresholding/optimizers/base.py`
and implements `optimize()`, returning best-so-far fitness and FEs used per
generation. Any variant runs against any objective via the registries.

## Contributing
- Branch per feature and open PRs into `main`.
- Run `pytest` before pushing.
- Never commit `venv/` or files under `results/` (both gitignored).
