"""Configuration for the INSAT-3DR infrared nowcast model.

Every choice here was made after inspecting the data
(`python -m ml.imagery.inspect_dataset`), not before -- see
docs/imagery_model_report.md for the reasoning behind each one.
"""
from datetime import datetime, timezone
from pathlib import Path

from app.services.imagery import insat

RANDOM_SEED = 42

BACKEND_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BACKEND_DIR / "ml" / "artifacts" / "imagery"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
# Derived, regenerable arrays (raw counts copied out of the HDF5 files so
# training does not gunzip 21 MB per crop). Gitignored; the original HDF5
# files are never modified.
CACHE_DIR = BACKEND_DIR / "ml" / "imagery" / "cache"
INSAT_DIR = insat.DEFAULT_INSAT_DIR

MODEL_NAME = "insat_nowcast"
MODEL_VERSION = "1.0.0"
CHECKPOINT_PATH = ARTIFACTS_DIR / f"{MODEL_NAME}_v1.pt"
METADATA_PATH = ARTIFACTS_DIR / f"{MODEL_NAME}_v1_metadata.json"
HISTORY_PATH = ARTIFACTS_DIR / "training_history.json"
EVALUATION_PATH = ARTIFACTS_DIR / "evaluation_test.json"


def _utc(day: int, hour: int, minute: int) -> datetime:
    return datetime(2025, 1, day, hour, minute, tzinfo=timezone.utc)


# --- Chronological split -------------------------------------------------
# The 59 usable frames are ONE continuous 34-hour sequence, so consecutive
# frames are near-duplicates of each other. A random split would put a
# frame's neighbours on both sides of the train/test line and the test score
# would mostly measure memorisation. Instead the timeline is cut twice and a
# sample (3 input frames + up to 3 target frames) is only used if EVERY one
# of its frames lies inside the same segment -- so no frame, as input or as
# target, is ever shared between train, validation and test.
TRAIN_END = _utc(7, 13, 15)   # train:      06 Jan 13:45 -> 07 Jan 13:15
VAL_START = _utc(7, 13, 45)   # validation: 07 Jan 13:45 -> 07 Jan 17:45
VAL_END = _utc(7, 17, 45)
TEST_START = _utc(7, 18, 15)  # test:       07 Jan 18:15 -> 07 Jan 23:45


def split_of(timestamp: datetime) -> str:
    if timestamp <= TRAIN_END:
        return "train"
    if VAL_START <= timestamp <= VAL_END:
        return "val"
    if timestamp >= TEST_START:
        return "test"
    return "unused"


# --- Model / optimisation --------------------------------------------------
BASE_CHANNELS = 32
CROP_SIZE = 128          # native 4 km pixels; no resizing anywhere
LOSS_MARGIN = 16         # crop border excluded from the loss (no inflow information there)
BATCH_SIZE = 16
ITERS_PER_EPOCH = 300
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 6
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
# Half of the training crops are drawn in proportion to how much the scene
# is actually changing there. Most of the sector, most of the time, is clear
# sky or ocean where "nothing changes" is already the right answer; without
# this the network sees mostly trivial examples (the regression analogue of
# class imbalance: only ~4-6 % of pixels are cold cloud).
ACTIVITY_SAMPLING_FRACTION = 0.5
ACTIVITY_BLOCK = 32
