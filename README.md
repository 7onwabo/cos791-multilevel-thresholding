# COS791 — Multilevel Image Thresholding via Differential Evolution

Group assignment: implement, analyse and empirically evaluate Differential
Evolution (DE) and its variants for multilevel image thresholding, using **Otsu**,
**Kapur** and **Tsallis** entropy as objective functions.
**Due: 8 October 2026.**

- **Phase 1 (BDS500):** 10 standard test images — proof of concept / benchmarks.
- **Phase 2 (CHAOS):** 15 abdominal MRI slices — medical domain application.

---

## Quick start

**Prerequisite:** Python **3.12** (pinned in `.python-version`; 3.11+ works).

### macOS / Linux (zsh, bash)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `import thresholding` work
pytest                    # objective-function tests should pass
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

### Smoke test
```bash
python scripts/run_experiments.py --dataset bds500 --objective all --k 3
```
Prints the experiment grid. (Actual DE runs land once section 1.3 is implemented.)

---

## Repository layout
```
data/            BDS500 (imgN.png + imgN_gt.png) and CHAOS MRI slices
docs/            assignment brief + background papers
src/thresholding/
  config.py      K levels {3,5,7,9,11,12}, runs, seeds, paths        [§1.1]
  histogram.py   image -> normalised 256-bin histogram
  datasets.py    load_bds500(), load_bds500_ground_truth(), load_chaos()
  objectives/    otsu.py, kapur.py, tsallis.py + OBJECTIVES registry  [§1.2]
  optimizers/    DE / JADE / SHADE / L-SHADE / LADE (stubs)           [§1.3]
  metrics/       PSNR, SSIM, uniformity, Jaccard, Dice (stubs)        [§2]
scripts/run_experiments.py   master reproducibility script (skeleton)
tests/           pytest suite (objectives covered)
results/         generated tables/plots (gitignored)
```

## Status
- [x] **§1.1** Threshold levels — `config.K_LEVELS = (3, 5, 7, 9, 11, 12)`
- [x] **§1.2** Objective functions — Otsu, Kapur, Tsallis (tested)
- [ ] **§1.3** DE variants — interface (`optimizers/base.py`) ready; bodies are stubs
- [ ] **§2** Metrics — signatures ready; bodies are stubs
- [ ] Master script wiring, statistical tests, ESWA report

---

## Work split (3 developers)
Shared core (config, histogram, objectives, datasets, runner) is pair-programmed;
each dev owns 2 DE variants plus an analysis slice.

| Dev | DE variants | Also owns |
|-----|-------------|-----------|
| A   | Standard DE (`DE/rand/1/bin`), JADE | Phase-1 reconstruction metrics (PSNR/SSIM/U) |
| B   | SHADE, L-SHADE (LPSR)               | Phase-2 metrics (Jaccard/Dice), GT matching  |
| C   | LADE (Late Acceptance DE)           | Stats (Wilcoxon/Friedman), convergence plots, report |

Each variant subclasses `BaseOptimizer` in `src/thresholding/optimizers/base.py`
and implements `optimize()`. Objectives are shared and objective-agnostic, so any
variant runs against any of the three via the `OBJECTIVES` registry.

> **⚠ Assignment inconsistency to confirm with lecturer:** §1.3 lists **5** DE
> variants (DE, JADE, SHADE, L-SHADE, LADE); Table 1 shows **6** (…CoDE, SaDE
> instead of LADE); the mark rubric says **5**. We are building the rubric's 5.
> If CoDE/SaDE are also required, add them under `optimizers/` following the same
> `BaseOptimizer` pattern.

## Contributing
- Branch per feature: `git checkout -b devA/standard-de`. Open PRs into `main`.
- Run `pytest` before pushing.
- Never commit `venv/` or files under `results/` (both gitignored).
