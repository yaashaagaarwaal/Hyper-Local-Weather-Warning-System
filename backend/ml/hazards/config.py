"""Configuration for the flood-mapping and lightning-detection image models.

Choices were made after inspecting the data (see
docs/flood_lightning_model_report.md), not before.
"""
from pathlib import Path

from app.services.imagery.hazard_common import HAZARD_MODEL_DIR, SYNTHETIC_ROOT

RANDOM_SEED = 42
ARTIFACTS_DIR = HAZARD_MODEL_DIR
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"   # every trained run (gitignored); the promoted one is copied up
FIGURES_DIR = ARTIFACTS_DIR / "figures"
CACHE_DIR = Path(__file__).resolve().parent / "cache"  # derived, regenerable arrays (gitignored)

# ---------------------------------------------------------------- flood ----
FLOOD_MODEL_NAME = "flood_unet"
FLOOD_MODEL_VERSION = "1.0.0"
FLOOD_CHECKPOINT = ARTIFACTS_DIR / "flood_unet_v1.pt"
FLOOD_METADATA = ARTIFACTS_DIR / "flood_unet_v1_metadata.json"
FLOOD_EVALUATION = ARTIFACTS_DIR / "flood_evaluation.json"
FLOOD_SYNTHETIC_DIR = SYNTHETIC_ROOT / "india_flood"

FLOOD_BASE = 16
FLOOD_DEPTH = 4
FLOOD_CROP = 256             # native 10 m pixels; no resizing anywhere
FLOOD_BATCH = 8
FLOOD_CROPS_PER_CHIP = 3     # random crops drawn from each training chip per epoch
FLOOD_EPOCHS = 40
FLOOD_LR = 1e-3
FLOOD_WEIGHT_DECAY = 1e-4
FLOOD_SEEDS = (0, 1, 2)      # each variant is trained three times: 29 test chips make one run's score noisy
# Variants compared. Only REAL validation chips decide which one is promoted.
FLOOD_VARIANTS = ("real", "real+synthetic", "synthetic")
FLOOD_SYNTHETIC_MIX = 0.5    # share of each batch drawn from synthetic chips in the "real+synthetic" variant

# ------------------------------------------------------------ lightning ----
LIGHTNING_MODEL_NAME = "lightning_unet"
LIGHTNING_MODEL_VERSION = "1.0.0"
LIGHTNING_CHECKPOINT = ARTIFACTS_DIR / "lightning_unet_v1.pt"
LIGHTNING_METADATA = ARTIFACTS_DIR / "lightning_unet_v1_metadata.json"
LIGHTNING_EVALUATION = ARTIFACTS_DIR / "lightning_evaluation.json"
LIGHTNING_SYNTHETIC_DIR = SYNTHETIC_ROOT / "india_lightning"
LIGHTNING_CACHE = CACHE_DIR / "goes_lightning.npz"

LIGHTNING_BASE = 24
LIGHTNING_DEPTH = 3
LIGHTNING_CROP = 128         # 4 km input pixels (=> 64x64 output cells of 8 km); equals the synthetic sample size
LIGHTNING_BATCH = 16
LIGHTNING_ITERS_PER_EPOCH = 120
LIGHTNING_EPOCHS = 20
LIGHTNING_LR = 1e-3
LIGHTNING_WEIGHT_DECAY = 1e-4
LIGHTNING_POS_WEIGHT = 4.0   # ~1 % of cells hold a flash
LIGHTNING_SEEDS = (0, 1, 2)
LIGHTNING_VARIANTS = ("real", "real+synthetic", "synthetic")
LIGHTNING_SYNTHETIC_MIX = 0.5
