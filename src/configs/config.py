"""
Central configuration for the Marine Debris Detection & Drift Prediction Pipeline.
Edit paths and hyperparameters here.
"""

import os

# ─────────────────────────── PATHS ───────────────────────────
# src/configs/config.py → src/ → project root
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
CLASS_WEIGHTS = [2.0, 1.0]  # Debris (rare, upweighted), Not Debris

# ─────────────────────────── TRAINING ────────────────────────


# ── Tuned hyperparameters for best performance ──
BATCH_SIZE   = 16
EPOCHS       = 120
LR           = 1e-4
WEIGHT_DECAY = 5e-5
PATIENCE     = 30           # early-stopping patience
NUM_WORKERS  = 4           # Use all 24 CPU cores for DataLoader

# ─────────────────────────── AUGMENTATION ────────────────────
AUG_PROB          = 0.5
ELASTIC_ALPHA     = 120
ELASTIC_SIGMA     = 6
SPECTRAL_NOISE    = 0.02
BRIGHTNESS_RANGE  = (-0.1, 0.1)

# ─────────────────────────── POST-PROCESSING ─────────────────
MIN_DEBRIS_PIXELS = 3       # keep tiny debris (real detections average ~8 px)

# ─────────────────────────── DRIFT ───────────────────────────
DRIFT_HOURS       = 72      # total prediction horizon
DRIFT_DT_SECONDS  = 900     # RK4 time step (15 min for higher accuracy)
DRIFT_ENSEMBLE_N  = 200     # number of perturbed particles (larger ensemble)
DRIFT_WIND_COEFF  = 0.035   # leeway: fraction of wind added to current
STOKES_COEFF      = 0.016   # Stokes drift coefficient (wave-induced)
