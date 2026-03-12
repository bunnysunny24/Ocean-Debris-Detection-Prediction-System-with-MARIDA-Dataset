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
RAW_BANDS     = 11          # Sentinel-2 bands (B10 excluded)
USE_SPECTRAL_INDICES = True # Append NDVI, NDWI, FDI, PI, RNDVI as extra channels
N_SPECTRAL_INDICES   = 5    # Number of spectral indices appended
INPUT_BANDS   = RAW_BANDS + (N_SPECTRAL_INDICES if USE_SPECTRAL_INDICES else 0)  # 16 total
PATCH_SIZE    = 256

CLASS_NAMES = {
    0: "Debris",
    1: "Not Debris",
}

# Class weights — debris is only 0.45% of pixels, needs heavy upweighting
CLASS_WEIGHTS = [15.0, 1.0]  # Debris (rare, upweighted 15x), Not Debris

# ─────────────────────────── MODEL ───────────────────────────
ENCODER_NAME  = "resnext50_32x4d"   # Options: resnext50_32x4d, resnext101_32x8d, efficientnet-b4
# ENCODER_NAME  = "resnext101_32x8d"  # Uncomment for max capacity (2x params, needs ~12GB VRAM)

# ─────────────────────────── TRAINING ────────────────────────

## ── Tuned hyperparameters for best performance ──
# Maximized for 39GB RAM, 8-thread CPU
BATCH_SIZE   = 64           # Aggressive: max RAM usage for fastest CPU training
EPOCHS       = 200          # Longer training for convergence
LR           = 1e-4
WEIGHT_DECAY = 5e-5
PATIENCE     = 50           # More patience for rare-class learning
NUM_WORKERS  = 8            # Max for 8 CPU threads
GRAD_ACCUM   = 2            # Gradient accumulation steps (effective batch = BATCH_SIZE * GRAD_ACCUM)
EMA_DECAY    = 0.995        # EMA decay — lower to let rare-class signals propagate faster
LABEL_SMOOTH = 0.02         # Mild label smoothing (was 0.05, too high for rare class)
ENCODER_FREEZE_EPOCHS = 3   # Freeze encoder for first N epochs so decoder learns debris first
FOCAL_GAMMA  = 1.0          # Reduced focal gamma — too high suppresses rare-class gradients
USE_SWA      = True         # Stochastic Weight Averaging for better generalization
SWA_START_EPOCH = 30        # Start SWA after this many epochs
SWA_LR       = 5e-5         # SWA learning rate

# ─────────────────────────── AUGMENTATION ────────────────────
AUG_PROB          = 0.5
ELASTIC_ALPHA     = 120
ELASTIC_SIGMA     = 6
SPECTRAL_NOISE    = 0.02
BRIGHTNESS_RANGE  = (-0.1, 0.1)
COPY_PASTE_PROB   = 0.7     # Copy-paste debris from donor patches (increased for more debris)

# ─────────────────────────── OVERSAMPLING ────────────────────
OVERSAMPLE_HEAVY  = 15      # Patches with >=20 debris pixels (increased)
OVERSAMPLE_LIGHT  = 10      # Patches with 1-19 debris pixels (increased)

# ─────────────────────────── POST-PROCESSING ─────────────────
MIN_DEBRIS_PIXELS = 3       # keep tiny debris (real detections average ~8 px)

# ─────────────────────────── DRIFT ───────────────────────────
DRIFT_HOURS       = 72      # total prediction horizon
DRIFT_DT_SECONDS  = 900     # RK4 time step (15 min for higher accuracy)
DRIFT_ENSEMBLE_N  = 200     # number of perturbed particles (larger ensemble)
DRIFT_WIND_COEFF  = 0.035   # leeway: fraction of wind added to current
STOKES_COEFF      = 0.016   # Stokes drift coefficient (wave-induced)
