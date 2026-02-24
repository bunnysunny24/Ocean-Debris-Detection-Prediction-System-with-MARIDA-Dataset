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
NUM_CLASSES   = 15          # MARIDA semantic classes (DN 1-15)
INPUT_BANDS   = 11          # Sentinel-2 bands (B10 excluded)
PATCH_SIZE    = 256

CLASS_NAMES = {
    1:  "Marine Debris",
    2:  "Dense Sargassum",
    3:  "Sparse Sargassum",
    4:  "Natural Organic Material",
    5:  "Ship",
    6:  "Clouds",
    7:  "Marine Water",
    8:  "Sediment-Laden Water",
    9:  "Foam",
    10: "Turbid Water",
    11: "Shallow Water",
    12: "Waves",
    13: "Cloud Shadows",
    14: "Wakes",
    15: "Mixed Water",
}

# Class weights (inverse frequency – tune after spectral extraction)
CLASS_WEIGHTS = [
    10.0,  # Marine Debris  (rare)
    3.0,   # Dense Sargassum
    3.0,   # Sparse Sargassum
    4.0,   # Natural Organic Material
    5.0,   # Ship
    1.5,   # Clouds
    0.5,   # Marine Water   (dominant)
    2.0,   # Sediment-Laden Water
    4.0,   # Foam
    2.0,   # Turbid Water
    2.0,   # Shallow Water
    2.5,   # Waves
    2.0,   # Cloud Shadows
    3.0,   # Wakes
    1.5,   # Mixed Water
]

# ─────────────────────────── TRAINING ────────────────────────
BATCH_SIZE   = 8
EPOCHS       = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE     = 15           # early-stopping patience

# ─────────────────────────── AUGMENTATION ────────────────────
AUG_PROB          = 0.5
ELASTIC_ALPHA     = 120
ELASTIC_SIGMA     = 6
SPECTRAL_NOISE    = 0.02
BRIGHTNESS_RANGE  = (-0.1, 0.1)

# ─────────────────────────── POST-PROCESSING ─────────────────
MIN_DEBRIS_PIXELS = 50      # remove blobs smaller than this

# ─────────────────────────── DRIFT ───────────────────────────
DRIFT_HOURS       = 72      # total prediction horizon
DRIFT_DT_SECONDS  = 3600    # RK4 time step (1 hour)
DRIFT_ENSEMBLE_N  = 50      # number of perturbed particles
DRIFT_WIND_COEFF  = 0.03    # leeway: fraction of wind added to current
