"""Central experiment configuration -- single source of truth.

Covers Assignment section 1.1 (threshold levels) plus the shared experimental
protocol constants from section 3 and the dataset/output paths. Import from here
rather than hardcoding numbers anywhere else so every DE variant, objective and
script stays in lockstep.
"""

from __future__ import annotations

from pathlib import Path

# --- Repository paths (resolved from this file, OS-agnostic via pathlib) -------
# config.py -> thresholding -> src -> <repo root>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
BDS500_DIR: Path = DATA_DIR / "BDS500"
CHAOS_DIR: Path = DATA_DIR / "CHAOS"
RESULTS_DIR: Path = REPO_ROOT / "results"

# --- Section 1.1: threshold levels to evaluate --------------------------------
# K = number of thresholds -> partitions the image into K + 1 intensity classes.
K_LEVELS: tuple[int, ...] = (3, 5, 7, 9, 11, 12)

# --- Section 3: experimental protocol -----------------------------------------
N_RUNS: int = 30            # independent runs per (image, objective, K)
GRAY_LEVELS: int = 256      # 8-bit intensity range 0..255
RANDOM_SEED: int = 42       # base seed; per-run seed = RANDOM_SEED + run_index

# Equal Function-Evaluation budget across all algorithms (stopping criterion).
# Tune once experiments are wired; kept here so every optimizer reads the same value.
MAX_FES: int = 30_000

# --- Objective parameters -----------------------------------------------------
# Tsallis non-extensivity parameter q (Table 1 reports results at q = 0.8).
TSALLIS_Q: float = 0.8

# Numerical floor to guard log(0) / division-by-zero on degenerate class splits.
EPSILON: float = 1e-12


def threshold_bounds(k: int) -> tuple[list[int], list[int]]:
    """Lower/upper bounds for a K-dimensional threshold vector.

    Each threshold lives in [1, GRAY_LEVELS - 2] (endpoints 0 and 255 are the
    fixed class boundaries). Optimizers should additionally keep the vector
    sorted; see ``sort`` handling in the objective helpers.
    """
    lower = [1] * k
    upper = [GRAY_LEVELS - 2] * k
    return lower, upper
