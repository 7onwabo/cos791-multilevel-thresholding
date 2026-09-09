"""COS791 multilevel image thresholding via Differential Evolution variants.

Package layout:
    config      -- experiment constants (K levels, runs, paths)   [Assignment 1.1]
    histogram   -- image -> normalised 256-bin histogram helpers
    datasets    -- loaders for the BDS500 and CHAOS image sets
    objectives  -- Otsu / Kapur / Tsallis fitness functions       [Assignment 1.2]
    optimizers  -- DE variants (stubs, filled per team split)      [Assignment 1.3]
    metrics     -- PSNR / SSIM / uniformity / Jaccard / Dice (stubs)
"""

__version__ = "0.1.0"
