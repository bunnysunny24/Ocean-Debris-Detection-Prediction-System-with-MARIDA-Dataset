"""
Central configuration for the Marine Debris Detection & Drift Prediction Pipeline.
Edit paths and hyperparameters here.
"""

import os

# ─────────────────────────── PATHS ───────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "Dataset")
PATCHES_DIR     = os.path.join(DATA_DIR, "patches")
SPLITS_DIR      = os.path.join(DATA_DIR, "splits")
H5_PATH         = os.path.join(DATA_DIR, "dataset.h5")
H5_SI_PATH      = os.path.join(DATA_DIR, "dataset_si.h5")
H5_GLCM_PATH    = os.path.join(DATA_DIR, "dataset_glcm.h5")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ─────────────────────────── DATASET ─────────────────────────
NUM_CLASSES   = 2           # Binary: debris vs. not-debris
INPUT_BANDS   = 11          # Sentinel-2 bands (B10 excluded)
PATCH_SIZE    = 256

CLASS_NAMES = {
    0: "Debris",
    1: "Not Debris",
}

# Class weights (inverse frequency – tune after spectral extraction)
CLASS_WEIGHTS = [1.0, 1.0]  # Debris (rare), Not Debris

# ─────────────────────────── TRAINING ────────────────────────

BATCH_SIZE   = 16
EPOCHS       = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE     = 15           # early-stopping patience
NUM_WORKERS  = 24           # Use all 24 CPU cores for DataLoader

# ─────────────────────────── AUGMENTATION ────────────────────
AUG_PROB          = 0.5
ELASTIC_ALPHA     = 120
ELASTIC_SIGMA     = 6
SPECTRAL_NOISE    = 0.02
BRIGHTNESS_RANGE  = (-0.1, 0.1)

# ─────────────────────────── POST-PROCESSING ─────────────────
MIN_DEBRIS_PIXELS = 200     # remove blobs smaller than this (stricter)

# ─────────────────────────── DRIFT ───────────────────────────
DRIFT_HOURS       = 72      # total prediction horizon
DRIFT_DT_SECONDS  = 3600    # RK4 time step (1 hour)
DRIFT_ENSEMBLE_N  = 50      # number of perturbed particles
DRIFT_WIND_COEFF  = 0.03    # leeway: fraction of wind added to current
