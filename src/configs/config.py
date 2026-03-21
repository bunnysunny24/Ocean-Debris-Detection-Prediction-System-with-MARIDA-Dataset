"""
Central configuration for the Marine Debris Detection & Drift Prediction Pipeline.
Edit paths and hyperparameters here.

GPU vs CPU is auto-detected at runtime. Settings below are tuned for both.
"""

import os
import torch

# ─────────────────────────── PATHS ───────────────────────────
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

# ─────────────────────────── DEVICE AUTO-DETECT ──────────────
_ON_GPU = torch.cuda.is_available()
_DEVICE_NAME = torch.cuda.get_device_name(0) if _ON_GPU else "CPU"

# ─────────────────────────── DATASET ─────────────────────────
NUM_CLASSES   = 2
RAW_BANDS     = 11          # Sentinel-2 bands (B10 excluded)
USE_SPECTRAL_INDICES = True # NDVI, NDWI, FDI, PI, RNDVI
N_SPECTRAL_INDICES   = 5
INPUT_BANDS   = RAW_BANDS + (N_SPECTRAL_INDICES if USE_SPECTRAL_INDICES else 0)  # 16
PATCH_SIZE    = 256

CLASS_NAMES = {0: "Debris", 1: "Not Debris"}

# Class weights — debris is ~0.45% of pixels
CLASS_WEIGHTS = [5.0, 1.0]

# ─────────────────────────── MODEL ───────────────────────────
# GPU: resnext101_32x8d for max capacity (~88M params, needs ≥8GB VRAM)
# CPU: resnext50_32x4d for reasonable speed (~41M params)
ENCODER_NAME = "resnext101_32x8d" if _ON_GPU else "resnext50_32x4d"

# ─────────────────────────── TRAINING (auto-scaled) ──────────
# Batch size: GPU gets larger real batch; CPU uses grad accumulation to compensate
BATCH_SIZE   = 32  if _ON_GPU else 8
NUM_WORKERS  = 8   if _ON_GPU else 4
GRAD_ACCUM   = 2   if _ON_GPU else 4    # effective batch = BATCH_SIZE * GRAD_ACCUM = 64 both ways
EPOCHS       = 200
LR           = 1e-4
WEIGHT_DECAY = 5e-5
PATIENCE     = 50

# ── EMA ──
EMA_DECAY    = 0.9998

# ── Encoder freeze warmup ──
ENCODER_FREEZE_EPOCHS = 5

# ── Loss ──
FOCAL_GAMMA    = 3.0
LABEL_SMOOTH   = 0.01
TVERSKY_ALPHA  = 0.3   # FP weight (lower = accept more false alarms)
TVERSKY_BETA   = 0.7   # FN weight (higher = strongly penalise missed debris)
TVERSKY_WEIGHT = 0.7   # Tversky fraction of total loss
AUX_LOSS_WEIGHT = 0.4  # Deep supervision auxiliary head weight

# ── SWA (legacy, replaced by EMA) ──
USE_SWA         = False
SWA_START_EPOCH = 30
SWA_LR          = 5e-5

# ─────────────────────────── AUGMENTATION ────────────────────
AUG_PROB         = 0.5
ELASTIC_ALPHA    = 120
ELASTIC_SIGMA    = 6
SPECTRAL_NOISE   = 0.02
BRIGHTNESS_RANGE = (-0.1, 0.1)
COPY_PASTE_PROB  = 0.4   # increased from 0.25
MIXUP_PROB       = 0.15
MIXUP_ALPHA      = 0.2

# ─────────────────────────── OVERSAMPLING ────────────────────
OVERSAMPLE_HEAVY = 8
OVERSAMPLE_LIGHT = 4

# ─────────────────────────── POST-PROCESSING ─────────────────
MIN_DEBRIS_PIXELS = 3

# ─────────────────────────── DRIFT ───────────────────────────
DRIFT_HOURS      = 72
DRIFT_DT_SECONDS = 900
DRIFT_ENSEMBLE_N = 200
DRIFT_WIND_COEFF = 0.035
STOKES_COEFF     = 0.016
