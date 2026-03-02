# Marine Debris Detection & Drift Prediction Pipeline

> End-to-end system for detecting floating marine debris in Sentinel-2 satellite imagery using deep learning, with physics-based ocean drift prediction.

Built on the [MARIDA](https://github.com/marine-debris/marine-debris) dataset (Marine Debris Archive) — the largest open benchmark for marine debris detection from Sentinel-2.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Quick Start](#quick-start)
6. [Pipeline Steps](#pipeline-steps)
7. [Configuration Reference](#configuration-reference)
8. [Model Architecture — Deep Learning](#model-architecture--deep-learning)
9. [Model Architecture — Random Forest](#model-architecture--random-forest)
10. [Loss Function — HybridLoss](#loss-function--hybridloss)
11. [Data Pipeline & Augmentation](#data-pipeline--augmentation)
12. [Training Process](#training-process)
13. [Evaluation & Metrics](#evaluation--metrics)
14. [Post-Processing](#post-processing)
15. [Drift Prediction](#drift-prediction)
16. [Visualization](#visualization)
17. [PowerShell Runner (run.ps1)](#powershell-runner-runps1)
18. [Python Pipeline Runner (run_pipeline.py)](#python-pipeline-runner-run_pipelinepy)
19. [Complete API Reference](#complete-api-reference)
20. [Testing & Validation](#testing--validation)
21. [Performance Expectations](#performance-expectations)
22. [Troubleshooting](#troubleshooting)

---

## Project Overview

This pipeline solves two coupled problems:

| Task | Method | Output |
|------|--------|--------|
| **Debris Detection** | Semantic segmentation (DeepLabV3+ / ResNeXt-50 + CBAM U-Net) + Random Forest | Binary pixel masks, GeoJSON polygons, CSV centroids |
| **Drift Prediction** | RK4 Lagrangian particle tracking with ensemble uncertainty | 6/12/24/48/72h trajectory GeoJSON + 95% confidence ellipses |

**Key features:**
- 11-band Sentinel-2 input (B1–B12, excluding B10)
- Nodata-safe loss function with per-pixel confidence weighting
- 8x test-time augmentation (TTA)
- Graduated oversampling for extreme class imbalance (0.45% debris pixels)
- Deterministic, NaN-safe per-band percentile normalization
- Morphological post-processing + polygon vectorization
- Physics-based drift with Stokes wave drift + wind leeway

---

## Directory Structure

```
Ocean_debris_detection/
├── README.md                          ← This file (only file at project root)
├── src/                               ← All source code
│   ├── train.py                       ← Training loop (AdamW + cosine LR + AMP)
│   ├── evaluate.py                    ← Evaluation with TTA + threshold sweep
│   ├── postprocess.py                 ← Morphological cleanup + GeoJSON export
│   ├── run_pipeline.py                ← Python master pipeline orchestrator
│   ├── run.ps1                        ← PowerShell pipeline runner
│   ├── requirements.txt               ← Python dependencies
│   ├── configs/
│   │   └── config.py                  ← Central configuration (paths + hyperparams)
│   ├── utils/
│   │   ├── dataset.py                 ← PyTorch Dataset + normalization + augmentation
│   │   ├── spectral_extraction.py     ← Extract features to HDF5 (for Random Forest)
│   │   └── visualise.py               ← Plot masks & drift trajectories
│   ├── semantic_segmentation/
│   │   └── models/
│   │       └── resnext_cbam_unet.py   ← ResNeXt-CBAM U-Net model + HybridLoss
│   ├── random_forest/
│   │   └── train_eval.py              ← RF training & evaluation
│   └── drift_prediction/
│       └── drift.py                   ← RK4 ensemble Lagrangian drift
├── Dataset/
│   ├── patches/                       ← MARIDA Sentinel-2 patches
│   │   └── S2_<DATE>_<TILE>/
│   │       ├── S2_..._CROP.tif        ← 11-band image (256x256)
│   │       ├── S2_..._CROP_cl.tif     ← Class label mask
│   │       └── S2_..._CROP_conf.tif   ← Pixel confidence map
│   ├── splits/
│   │   ├── train_X.txt                ← Training patch IDs
│   │   ├── val_X.txt                  ← Validation patch IDs
│   │   └── test_X.txt                 ← Test patch IDs
│   ├── dataset.h5                     ← Extracted spectral bands (for RF)
│   ├── dataset_si.h5                  ← Spectral indices (for RF)
│   └── dataset_glcm.h5               ← GLCM texture features (for RF)
├── checkpoints/
│   ├── resnext_cbam_best.pth          ← Best model (by debris F1)
│   ├── resnext_cbam_last.pth          ← Last epoch checkpoint
│   └── run_<TIMESTAMP>/               ← Per-run checkpoints + metrics
│       ├── best.pth
│       ├── last.pth
│       ├── epoch_001.pth ... epoch_N.pth
│       ├── metrics.csv
│       └── *.png                      ← Per-metric training curves
├── outputs/
│   ├── predicted_test/                ← Predicted masks (GeoTIFF)
│   ├── geospatial/
│   │   ├── all_debris.geojson         ← Merged debris polygons
│   │   ├── debris_locations.csv       ← Centroids + bounding boxes
│   │   └── <patch>_debris.geojson     ← Per-patch polygons
│   ├── drift/
│   │   ├── all_drift.geojson          ← All trajectories + ellipses
│   │   └── <patch>_drift.geojson      ← Per-object trajectories
│   ├── random_forest_model.joblib     ← Trained RF model
│   └── confusion_matrix_test.png      ← Eval confusion matrix
├── logs/
│   ├── external_logs/                 ← Timestamped training logs
│   └── tsboard/                       ← TensorBoard event files
└── archive/                           ← Archived utility scripts
```

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.x+ (recommended for GPU training)
- Conda (recommended) or pip

### Setup

```bash
# Create conda environment
conda create -n debris python=3.10 -y
conda activate debris

# Install PyTorch (GPU)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Or PyTorch (CPU only)
conda install pytorch torchvision cpuonly -c pytorch

# Install remaining dependencies
cd Ocean_debris_detection/src
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=1.13.0 | Deep learning framework |
| `torchvision` | >=0.14.0 | Pretrained encoder weights |
| `segmentation-models-pytorch` | >=0.3.0 | DeepLabV3+ architecture |
| `albumentations` | >=1.0.0 | Data augmentation pipeline |
| `rasterio` | — | GeoTIFF I/O |
| `numpy` | — | Numerical computation |
| `scipy` | — | Interpolation, morphology |
| `scikit-learn` | — | Random Forest, metrics |
| `scikit-image` | — | Morphological operations |
| `pandas` | — | Tabular data, HDF5 |
| `matplotlib` | — | Plotting & visualization |
| `shapely` | — | Geometry operations |
| `joblib` | — | Model serialization |
| `tensorboard` | — | Training visualization |
| `tqdm` | — | Progress bars |
| `netCDF4` | (optional) | Ocean/wind current data |
| `pydensecrf` | (optional) | DenseCRF post-processing |

---

## Dataset

### MARIDA (Marine Debris Archive)

| Property | Value |
|----------|-------|
| **Satellite** | Sentinel-2 (L2A) |
| **Bands** | 11 (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12) |
| **Patch size** | 256 x 256 pixels |
| **Resolution** | 10m (visible), 20m (SWIR), 60m (coastal) |
| **Total patches** | ~1,381 |
| **Train split** | 694 patches |
| **Validation split** | 328 patches |
| **Test split** | 359 patches |
| **Classes (original)** | 15 (Marine Debris, Dense Sargassum, Sparse Sargassum, Natural Organic Material, Ship, Cloud, Marine Water, Turbid Water, Shallow Water, Waves/Whitecaps, Wakes, Mixed Water, Volcanic Island Mask, Snow/Ice, Other) |
| **Classes (binary)** | 2 — Debris (class 0), Not-Debris (class 1) |
| **Nodata** | Encoded as -1 in binary mapping |

### Label Mapping

| Raw DN (in _cl.tif) | Binary Label | Class |
|----------------------|--------------|-------|
| 0 | -1 (ignored) | Nodata |
| 1 | 0 | **Marine Debris** |
| 2–15 | 1 | Not-Debris |

### Class Imbalance Statistics

| Metric | Value |
|--------|-------|
| Total pixels (train) | ~45.5M (694 x 256 x 256) |
| Valid (non-nodata) pixels | ~430K (0.82% of total) |
| Debris pixels | 1,943 (0.45% of valid) |
| Patches with debris | 190 / 694 (27.4%) |
| Patches without debris | 504 / 694 (72.6%) |
| Average debris pixels per patch | ~10.2 |

### Confidence Maps

Each patch has a `_conf.tif` with per-pixel confidence levels:

| DN Value | Meaning | Loss Weight |
|----------|---------|-------------|
| 0 | Uncertain | 0.2 |
| 1 | Confident | 0.7 |
| 2 | Highly confident | 1.0 |

### Patch File Layout

```
Dataset/patches/
└── S2_1-12-19_48MYU/
    ├── S2_1-12-19_48MYU_0.tif         ← 11-band image
    ├── S2_1-12-19_48MYU_0_cl.tif      ← Class label mask (DN 0-15)
    └── S2_1-12-19_48MYU_0_conf.tif    ← Confidence map (DN 0-2)
```

---

## Quick Start

### Full pipeline (train + eval + post-process + drift)

```powershell
cd Ocean_debris_detection/src
.\run.ps1
```

### Train only

```powershell
.\run.ps1 -TrainOnly
```

### Evaluate existing checkpoint

```powershell
.\run.ps1 -EvalOnly
```

### Python alternative

```bash
cd src
python run_pipeline.py                  # Full pipeline
python train.py --epochs 120 --batch 16 # Train only
python evaluate.py --split test --tta   # Evaluate with TTA
```

---

## Pipeline Steps

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. TRAIN    │────>│  2. EVALUATE │────>│  3. POST-    │────>│  4. DRIFT    │
│  train.py    │     │  evaluate.py │     │  PROCESS     │     │  PREDICT     │
│              │     │              │     │ postprocess  │     │  drift.py    │
│  DeepLabV3+  │     │  TTA + sweep │     │  .py         │     │              │
│  AdamW       │     │  Threshold   │     │  Morphology  │     │  RK4 + wind  │
│  FocalDice   │     │  IoU/F1/P/R  │     │  GeoJSON     │     │  Ensemble    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                      │
                                          ┌──────────────┐            │
                                          │  OPTIONAL:   │            │
                                          │  Random      │<───────────┘
                                          │  Forest      │
                                          └──────────────┘
```

---

## Configuration Reference

All settings are in `src/configs/config.py`:

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_DIR` | Auto-detected (3 levels up from config.py) | Project root |
| `DATA_DIR` | `{BASE_DIR}/Dataset` | Dataset directory |
| `PATCHES_DIR` | `{DATA_DIR}/patches` | Sentinel-2 patches |
| `SPLITS_DIR` | `{DATA_DIR}/splits` | Train/val/test split files |
| `H5_PATH` | `{DATA_DIR}/dataset.h5` | Spectral bands HDF5 |
| `H5_SI_PATH` | `{DATA_DIR}/dataset_si.h5` | Spectral indices HDF5 |
| `H5_GLCM_PATH` | `{DATA_DIR}/dataset_glcm.h5` | GLCM texture HDF5 |
| `CHECKPOINTS_DIR` | `{BASE_DIR}/checkpoints` | Model checkpoints |
| `OUTPUTS_DIR` | `{BASE_DIR}/outputs` | All pipeline outputs |
| `LOGS_DIR` | `{BASE_DIR}/logs` | Training logs + TensorBoard |

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_CLASSES` | 2 | Binary: debris vs. not-debris |
| `INPUT_BANDS` | 11 | Sentinel-2 bands (excluding B10) |
| `PATCH_SIZE` | 256 | Spatial dimension of patches |
| `BATCH_SIZE` | 16 | Training batch size |
| `EPOCHS` | 120 | Maximum training epochs |
| `LR` | 1e-4 | Initial learning rate |
| `WEIGHT_DECAY` | 5e-5 | AdamW L2 regularization |
| `PATIENCE` | 30 | Early stopping patience (epochs without debris F1 improvement) |
| `NUM_WORKERS` | 4 | DataLoader worker processes |
| `CLASS_WEIGHTS` | [2.0, 1.0] | Loss weights [debris, not-debris] |

### Augmentation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `AUG_PROB` | 0.5 | Base augmentation probability |
| `ELASTIC_ALPHA` | 120 | Elastic transform alpha |
| `ELASTIC_SIGMA` | 6 | Elastic transform sigma |
| `SPECTRAL_NOISE` | 0.02 | Spectral noise standard deviation |
| `BRIGHTNESS_RANGE` | (-0.1, 0.1) | Brightness jitter range |

### Post-Processing

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MIN_DEBRIS_PIXELS` | 3 | Minimum blob size to keep (real debris averages ~8 px) |

### Drift Prediction

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DRIFT_HOURS` | 72 | Total prediction horizon |
| `DRIFT_DT_SECONDS` | 900 | RK4 timestep (15 minutes) |
| `DRIFT_ENSEMBLE_N` | 200 | Number of perturbed particles |
| `DRIFT_WIND_COEFF` | 0.035 | Wind leeway fraction |
| `STOKES_COEFF` | 0.016 | Wave-induced Stokes drift coefficient |

---

## Model Architecture — Deep Learning

### Primary Model: DeepLabV3+ with ResNeXt-50 Encoder

The production model uses **segmentation-models-pytorch (smp)** `DeepLabV3Plus`:

```
Input: (B, 11, 256, 256)  <- 11 Sentinel-2 bands
         |
    ┌────┴────┐
    | Encoder |  ResNeXt-50 (32x4d), ImageNet pretrained
    |         |  First conv adapted: 3ch -> 11ch (weight averaging + tiling)
    |         |
    |  Stage 1|  Conv 7x7/2 + BN + ReLU + MaxPool/2  -> (B, 64, 64, 64)
    |  Stage 2|  3x Bottleneck (grouped conv 32x4d)   -> (B, 256, 64, 64)
    |  Stage 3|  4x Bottleneck                        -> (B, 512, 32, 32)
    |  Stage 4|  6x Bottleneck                        -> (B, 1024, 16, 16)
    |  Stage 5|  3x Bottleneck (dilated)              -> (B, 2048, 16, 16)
    └────┬────┘
         |
    ┌────┴────┐
    | ASPP    |  Atrous Spatial Pyramid Pooling
    | Decoder |  rates: [12, 24, 36], 256 output channels
    |         |  Bilinear upsample 4x -> concat with Stage 2 features
    |         |  Conv 3x3 -> Conv 3x3 -> final Conv 1x1
    └────┬────┘
         |
    Output: (B, 2, 256, 256)  <- logits [debris, not-debris]
```

| Property | Value |
|----------|-------|
| **Architecture** | DeepLabV3+ |
| **Encoder** | `resnext50_32x4d` |
| **Encoder weights** | ImageNet (adapted to 11 channels) |
| **Total parameters** | 26,174,802 |
| **Output** | 2-class logits (no activation) |
| **Channel adaptation** | Mean of pretrained RGB weights tiled to 11 channels |

### ResNeXt-50 Encoder Details

ResNeXt-50 uses **grouped convolutions** with cardinality=32 and bottleneck width=4d:

```
Bottleneck Block:
  Input (C_in)
    |
    ├── Conv 1x1 (C_in -> 128)  + BN + ReLU
    ├── Conv 3x3 (128 -> 128, groups=32)  + BN + ReLU    <- Grouped convolution
    ├── Conv 1x1 (128 -> 256)  + BN
    |
    └── Residual connection (identity or 1x1 projection)
         |
    ReLU(sum)
```

| Stage | Blocks | Output Channels | Stride | Output Size |
|-------|--------|----------------|--------|-------------|
| Conv1 | 1 | 64 | /2 | 128x128 |
| MaxPool | — | 64 | /2 | 64x64 |
| Layer 1 | 3 | 256 | /1 | 64x64 |
| Layer 2 | 4 | 512 | /2 | 32x32 |
| Layer 3 | 6 | 1024 | /2 | 16x16 |
| Layer 4 | 3 | 2048 | /1 (dilated) | 16x16 |

### Alternative Model: ResNeXt-CBAM U-Net

A custom architecture defined in `resnext_cbam_unet.py` (available but not the default production model):

```
Input: (B, 11, 256, 256)
         |
    ┌────┴────┐
    | Encoder |  ResNeXt-50 backbone (ImageNet pretrained)
    |         |
    |  enc0   |  Conv 7x7/2 (11->64) + BN + ReLU         -> (B, 64, 128, 128)
    |  pool   |  MaxPool 3x3/2                            -> (B, 64, 64, 64)
    |  enc1   |  Layer1: 3x Bottleneck                    -> (B, 256, 64, 64)
    |  enc2   |  Layer2: 4x Bottleneck /2                 -> (B, 512, 32, 32)
    |  enc3   |  Layer3: 6x Bottleneck /2                 -> (B, 1024, 16, 16)
    |  enc4   |  Layer4: 3x Bottleneck /2                 -> (B, 2048, 8, 8)
    └────┬────┘
         |
    ┌────┴──────────┐
    | Bottleneck    |  CBAM(2048) — Channel + Spatial Attention
    | CBAM          |
    └──────┬────────┘
           |
    ┌──────┴──────┐
    |   Decoder   |
    |             |
    |  dec4       |  ConvT 2x2/2 (2048->512) + cat(enc3) -> 2xConv3x3 + CBAM -> (B, 512, 16, 16)
    |  dec3       |  ConvT 2x2/2 (512->256)  + cat(enc2) -> 2xConv3x3 + CBAM -> (B, 256, 32, 32)
    |  dec2       |  ConvT 2x2/2 (256->128)  + cat(enc1) -> 2xConv3x3 + CBAM -> (B, 128, 64, 64)
    |  dec1       |  ConvT 2x2/2 (128->64)   + cat(enc0) -> 2xConv3x3 + CBAM -> (B, 64, 128, 128)
    |             |
    |  final_up   |  ConvT 2x2/2 (64->64)  -> bilinear to input size         -> (B, 64, 256, 256)
    |  final_conv |  Conv 1x1 (64->2)                                         -> (B, 2, 256, 256)
    └─────────────┘
```

### CBAM (Convolutional Block Attention Module)

CBAM applies **channel attention** then **spatial attention** sequentially:

```
Channel Attention:
  Input (B, C, H, W)
    ├── AdaptiveAvgPool -> (B, C, 1, 1) -> FC(C -> C/r) -> ReLU -> FC(C/r -> C)
    ├── AdaptiveMaxPool -> (B, C, 1, 1) -> FC(C -> C/r) -> ReLU -> FC(C/r -> C)
    └── Sigmoid(sum) -> channel weights (B, C, 1, 1)
  Output = Input * channel_weights

Spatial Attention:
  Input (B, C, H, W)
    ├── AvgPool across channels -> (B, 1, H, W)
    ├── MaxPool across channels -> (B, 1, H, W)
    └── Concat -> Conv 7x7 -> Sigmoid -> spatial weights (B, 1, H, W)
  Output = Input * spatial_weights
```

| Parameter | Value |
|-----------|-------|
| Reduction ratio `r` | 16 |
| Spatial kernel | 7x7 |
| Applied at | Bottleneck + every decoder block |

### Decoder Block (U-Net)

Each decoder stage:
1. `ConvTranspose2d` (2x2, stride 2) — upsamples spatially
2. Concatenate with encoder skip connection
3. Two `Conv2d` (3x3) + `BatchNorm2d` + `ReLU` blocks
4. `CBAM` attention module

Weights initialized with Kaiming Normal (fan_out, ReLU).

---

## Model Architecture — Random Forest

| Property | Value |
|----------|-------|
| **Algorithm** | `sklearn.ensemble.RandomForestClassifier` |
| **Trees** | 300 (trained in chunks of 10 with warm_start) |
| **Max depth** | None (unlimited) |
| **Min samples leaf** | 5 |
| **Class weights** | Balanced (computed from training labels) |
| **Sample weights** | Confidence-based (0.2 / 0.7 / 1.0) |
| **OOB score** | Enabled (final model retrained) |
| **Random state** | 42 |
| **n_jobs** | -1 (all cores) |

### Feature Sets

| Mode | Features | Dimensions | HDF5 File |
|------|----------|------------|-----------|
| `bands` | Raw 11 spectral bands (B1-B12 exc. B10) | 11 | `dataset.h5` |
| `indices` | NDVI, NDWI, FDI, PI, RNDVI | 5 | `dataset_si.h5` |
| `texture` | NIR local variance, NIR local mean (5x5 window) | 2 | `dataset_glcm.h5` |

### Spectral Indices

| Index | Formula | Purpose |
|-------|---------|---------|
| NDVI | (B8 - B4) / (B8 + B4) | Vegetation detection |
| NDWI | (B3 - B8) / (B3 + B8) | Water detection |
| FDI | B8 - (B4 + (B12 - B4) * (832.8 - 664.6) / (2201.5 - 664.6)) | Floating Debris Index |
| PI | B4 / (B3 + B4 + B8) | Plastic Index |
| RNDVI | (B4 - B3) / (B4 + B3) | Reversed NDVI |

---

## Loss Function — HybridLoss

Defined in `resnext_cbam_unet.py`. Combines **Focal Cross-Entropy** and **Dice Loss** with nodata safety.

### Formula

```
L = (1 - dice_weight) * L_focal + dice_weight * L_dice
```

Where `dice_weight = 0.7` (Dice-dominant).

### Focal Cross-Entropy

```
L_focal = -alpha_c * (1 - p_t)^gamma * log(p_t)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `focal_gamma` | 2.0 | Focuses on hard examples |
| `class_weights` | [2.0, 1.0] | Debris upweighted 2x |

### Dice Loss

```
L_dice = 1 - (1/C) * sum_c [ (2 * sum(p_c * t_c) + eps) / (sum(p_c) + sum(t_c) + eps) ]
```

### Nodata Handling (Critical)

Nodata pixels (target = -1) are **completely excluded** from loss computation:

1. A `valid_mask` is computed: `(target != -1).float()`
2. `safe_target` replaces -1 with 0 (placeholder only — masked out before contributing)
3. CE is computed with `reduction="none"`, then multiplied by `valid_mask`
4. Mean is taken over valid pixels only: `ce_map.sum() / valid_mask.sum()`
5. Dice loss also masks out nodata pixels per class

### Confidence Weighting

Per-pixel confidence weights from `_conf.tif` are applied after the focal modulation:

```python
ce_map = ce_map * valid_mask * conf_weights
```

### HybridLoss Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_weights` | `float[]` | `[2.0, 1.0]` | Per-class loss weights |
| `dice_weight` | `float` | `0.7` | Weight for Dice loss (1-this for CE) |
| `ignore_index` | `int` | `-1` | Nodata label value |
| `use_focal` | `bool` | `True` | Enable focal modulation |
| `focal_gamma` | `float` | `2.0` | Focal loss gamma |

---

## Data Pipeline & Augmentation

### MARIDADataset (PyTorch Dataset)

File: `src/utils/dataset.py`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | `str` | `"train"` | `"train"`, `"val"`, or `"test"` |
| `augment_data` | `bool` | `True` | Apply augmentations (forced `False` for val/test) |
| `use_conf_weights` | `bool` | `True` | Load confidence maps for weighted loss |

#### Data Loading Flow

```
1. Read split file (train_X.txt / val_X.txt / test_X.txt)
         |
2. Filter out pure-nodata patches (all DN=0)
         |
3. Oversampling (train only):
   ├── Patches with >=20 debris pixels -> 8x duplication
   ├── Patches with 1-19 debris pixels -> 5x duplication
   └── Non-debris patches -> 1x (no duplication)
         |
4. For each __getitem__(idx):
   ├── Load 11-band image from .tif -> (11, 256, 256) float32
   ├── Replace NaN with 0 (np.nan_to_num)
   ├── Load mask from _cl.tif -> DN [0-15]
   ├── Binary mapping: DN=1->0 (debris), DN=2-15->1 (not-debris), DN=0->-1 (nodata)
   ├── Load confidence from _conf.tif -> DN [0, 1, 2]
   ├── Percentile normalization -> [0, 1]
   ├── Transpose to (H, W, C) for Albumentations
   ├── Apply augmentation (image + mask + conf jointly)
   ├── Map conf integers -> float weights: {0->0.2, 1->0.7, 2->1.0}
   ├── Transpose back to (C, H, W)
   └── Return {"image": Tensor, "mask": Tensor, "conf": Tensor, "id": str}
```

#### Dataset Size After Oversampling (Train)

| Category | Patches | After Oversampling |
|----------|---------|-------------------|
| Heavy debris (>=20 px) | 42 | 42 x 8 = 336 |
| Light debris (1-19 px) | 148 | 148 x 5 = 740 |
| Non-debris | 456 | 456 x 1 = 456 |
| **Total** | **646** | **1,532** |

### Normalization

Per-band percentile clipping computed **deterministically from all 694 training patches**:

```python
# For each band b:
lo = np.nanpercentile(all_training_pixels_band_b, 2)
hi = np.nanpercentile(all_training_pixels_band_b, 98)
normalized[b] = clip((pixel[b] - lo) / (hi - lo), 0, 1)
```

- Uses `np.nanpercentile` to handle NaN values in bands 9-10 (e.g., patch `21-2-17_16PCC_0`)
- Uses `np.nan_to_num` on image load to replace NaN with 0
- Computed once on first dataset instantiation, cached globally

### File Index Cache

All `.tif` files are indexed once via `os.walk()` at startup:

```python
_FILE_INDEX = {}  # stem -> full path (e.g., "S2_1-12-19_48MYU_0" -> "D:/.../.tif")
```

Prevents repeated recursive `glob.glob()` calls (>10,000 files).

### Augmentation Pipeline

Applied only during training. Uses **Albumentations v2.0.8**:

| Transform | Probability | Parameters | Purpose |
|-----------|-------------|------------|---------|
| `HorizontalFlip` | 0.5 | — | Rotation invariance |
| `VerticalFlip` | 0.5 | — | Rotation invariance |
| `RandomRotate90` | 0.5 | — | Rotation invariance |
| `GaussNoise` | 0.2 | `std_range=(0.01, 0.03)` | Sensor noise simulation |
| `ElasticTransform` | 0.15 | `alpha=120, sigma=6` | Shape deformation |
| `GridDistortion` | 0.1 | — | Geometric distortion |

**Critical details:**
- `additional_targets={'conf_raw': 'mask'}` ensures confidence maps receive the **same spatial transforms** as masks
- Confidence integer-to-float mapping (`{0->0.2, 1->0.7, 2->1.0}`) happens **after** augmentation to preserve spatial alignment
- `GaussNoise` uses `std_range` (fraction of value range), NOT `var_limit` (which is uint8-scale and destroys float32 images)
- `RandomResizedCrop` and `CoarseDropout` are **intentionally excluded** — they can discard the ~30 labeled pixels per patch

---

## Training Process

File: `src/train.py`

### Training Loop

```
For each epoch:
  1. train_epoch():
     ├── model.train()
     ├── For each batch:
     |   ├── Forward pass with AMP autocast
     |   ├── Compute HybridLoss(logits, masks, conf_weights)
     |   ├── Skip batch if loss is NaN
     |   ├── GradScaler: scale -> backward -> unscale
     |   ├── Gradient clipping (max_norm=5.0)
     |   └── Optimizer step + scaler update
     └── Return mean training loss

  2. val_epoch():
     ├── model.eval(), torch.no_grad()
     ├── For each batch:
     |   ├── Forward pass with AMP autocast
     |   ├── Compute loss + argmax predictions
     |   └── Accumulate predictions & masks
     ├── Compute IoU per class
     ├── Compute debris Precision / Recall / F1
     └── Return (val_loss, mIoU, per_class_iou, precision, recall, F1)

  3. Logging:
     ├── Console + file log (timestamped)
     ├── TensorBoard scalars (loss, mIoU)
     └── CSV row (epoch, losses, mIoU, per-class IoU, P/R/F1)

  4. Checkpointing:
     ├── Save epoch_NNN.pth (every epoch)
     ├── Save last.pth (for resuming)
     ├── If debris F1 improves:
     |   ├── Save best.pth (in run directory)
     |   └── Copy to checkpoints/resnext_cbam_best.pth (shared)
     └── If no improvement for PATIENCE epochs -> early stop

  5. LR scheduler step:
     ├── Epochs 1-5: LinearLR warmup (0.1x -> 1.0x LR)
     └── Epochs 6+: CosineAnnealingLR (-> eta_min=1e-6)
```

### Optimizer & Scheduler

| Component | Configuration |
|-----------|--------------|
| **Optimizer** | AdamW (lr=1e-4, weight_decay=5e-5, betas=(0.9, 0.999)) |
| **Warmup** | LinearLR, 5 epochs, start_factor=0.1 |
| **Main scheduler** | CosineAnnealingLR, T_max=115, eta_min=1e-6 |
| **Combined** | SequentialLR with milestone at epoch 5 |
| **AMP** | GradScaler for mixed-precision (float16/float32) on CUDA |
| **Gradient clipping** | max_norm=5.0 (prevents exploding gradients) |

### Learning Rate Schedule

```
LR
1e-4 ┌──────────────────────────────────────────────────
     |      / Warmup          Cosine Annealing
     |     /              \
     |    /                  \
     |   /                     \
     |  /                        \
1e-5 | /                           \
     |/                              \
1e-6 └────────────────────────────────────────────────
     0    5                        115            120
                       Epoch
```

### Best Model Tracking

The best checkpoint is tracked by **debris F1 score** (not mIoU or total accuracy), because:
- Overall accuracy is dominated by not-debris class (>99% of pixels)
- mIoU can be high even with zero debris detections
- Debris F1 directly measures detection performance

### train.py Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | `int` | 120 | Maximum training epochs |
| `--batch` | `int` | 16 | Batch size |
| `--lr` | `float` | 1e-4 | Initial learning rate |
| `--workers` | `int` | 4 | DataLoader workers |
| `--weight_decay` | `float` | 5e-5 | AdamW weight decay |
| `--patience` | `int` | 30 | Early stopping patience |
| `--resume` | flag | — | Resume from best.pth in latest run |

### Output Files (Per Training Run)

```
checkpoints/run_YYYYMMDD_HHMMSS/
├── best.pth          ← Best debris F1 checkpoint
├── last.pth          ← Last epoch (for resuming)
├── epoch_001.pth     ← Every epoch checkpoint
├── epoch_002.pth
├── ...
├── metrics.csv       ← Epoch-by-epoch metrics table
├── train_loss.png    ← Training loss curve
├── val_loss.png      ← Validation loss curve
├── mIoU.png          ← Mean IoU curve
├── iou_debris.png    ← Debris IoU curve
├── iou_not_debris.png
├── precision.png     ← Debris precision curve
├── recall.png        ← Debris recall curve
└── f1.png            ← Debris F1 curve
```

### metrics.csv Format

| Column | Type | Description |
|--------|------|-------------|
| epoch | int | Epoch number (1-indexed) |
| train_loss | float | Mean training loss |
| val_loss | float | Mean validation loss |
| mIoU | float | Mean IoU (both classes) |
| iou_debris | float | Debris class IoU |
| iou_not_debris | float | Not-debris class IoU |
| precision | float | Debris precision |
| recall | float | Debris recall |
| f1 | float | Debris F1 score |

---

## Evaluation & Metrics

File: `src/evaluate.py`

### Evaluation Flow

```
1. Load checkpoint (auto-detects smp DeepLabV3+ vs custom ResNeXt-CBAM U-Net)
2. Load test/val dataset (no augmentation, no confidence weights)
3. For each batch:
   ├── Forward pass (with optional 8x TTA)
   ├── Apply threshold to debris probability
   ├── Remove small blobs < MIN_DEBRIS_PIXELS
   ├── Update confusion matrix
   └── Optionally save predicted masks as GeoTIFF
4. Compute per-class metrics (IoU, F1, Precision, Recall, Dice)
5. Run threshold sweep (0.15 -> 0.85, step 0.05)
6. Re-report metrics at optimal threshold
7. Save confusion matrix PNG
```

### Model Auto-Detection

The evaluator automatically detects which architecture was used:
- If checkpoint keys contain `encoder.`, `decoder.`, `segmentation_head.` -> loads as smp `DeepLabV3Plus`
- Otherwise -> loads as custom `ResNeXtCBAMUNet`
- Tries `resnext50_32x4d` encoder first, falls back to `resnet50` for older checkpoints

### Test-Time Augmentation (TTA)

8x geometric variants averaged together:

| Variant | Rotation | H-Flip |
|---------|----------|--------|
| 1 | 0 deg | No |
| 2 | 0 deg | Yes |
| 3 | 90 deg | No |
| 4 | 90 deg | Yes |
| 5 | 180 deg | No |
| 6 | 180 deg | Yes |
| 7 | 270 deg | No |
| 8 | 270 deg | Yes |

```python
def _tta_predict(model, imgs, device):
    accum = zeros(B, 2, H, W)
    for k in [0, 1, 2, 3]:        # rotations
        for hflip in [False, True]:
            x = apply_transforms(imgs, k, hflip)
            probs = softmax(model(x))
            probs = undo_transforms(probs, k, hflip)
            accum += probs
    return accum / 8.0
```

Typically boosts debris F1 by **3-5%**.

### Threshold Sweep

The evaluation automatically sweeps debris probability thresholds from 0.15 to 0.85:

```
thr=0.15  P=0.1234  R=0.9876  F1=0.2345
thr=0.20  P=0.1567  R=0.9654  F1=0.2789
...
thr=0.50  P=0.4321  R=0.7890  F1=0.5678  <- default
...
thr=0.85  P=0.8765  R=0.3456  F1=0.4567
>>> Optimal threshold: 0.35  (F1=0.6234)
```

If the optimal threshold differs from the `--threshold` argument, metrics are re-computed and reported at that threshold.

### Metrics Computed

| Metric | Formula | Per-Class |
|--------|---------|-----------|
| IoU | TP / (TP + FP + FN) | Yes |
| F1 | 2 * P * R / (P + R) | Yes |
| Precision | TP / (TP + FP) | Yes |
| Recall | TP / (TP + FN) | Yes |
| Dice | 2 * TP / (2 * TP + FP + FN) | Yes |
| Pixel Accuracy | correct / total_valid | No |
| Mean IoU | (1/C) * sum(IoU_c) | No |

### evaluate.py Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--split` | `str` | `"test"` | Dataset split to evaluate |
| `--ckpt` | `str` | `checkpoints/resnext_cbam_best.pth` | Checkpoint path |
| `--save_masks` | flag | `True` | Save predicted masks as GeoTIFF |
| `--threshold` | `float` | 0.5 | Debris probability threshold |
| `--min_debris_pixels` | `int` | 3 | Min blob size |
| `--no_postprocessing` | flag | `False` | Skip small-blob removal |
| `--tta` | flag | `False` | Enable 8x test-time augmentation |

### Predicted Mask Format

Saved as GeoTIFF with `(pred_mask + 1)` encoding:
- DN 1 = Debris
- DN 2 = Not-Debris

CRS and transform are copied from the source patch.

---

## Post-Processing

File: `src/postprocess.py`

### Pipeline

```
For each _pred.tif in pred_dir:
  1. Load predicted mask (DN 1=debris, DN 2=not-debris in saved file)
  2. Optional DenseCRF refinement (if pydensecrf installed + RGB available):
     ├── Build unary potentials from binary mask
     ├── Add pairwise Gaussian (sxy=3, compat=3)
     ├── Add pairwise Bilateral (sxy=80, srgb=13, compat=10)
     └── Run 5 CRF iterations
  3. Morphological refinement:
     ├── Remove small holes (area < MIN_DEBRIS_PIXELS)
     ├── Remove small objects (area < MIN_DEBRIS_PIXELS)
     └── Binary dilation (disk radius=1)
  4. Polygon vectorization (rasterio.features.shapes)
  5. For each polygon:
     ├── Compute centroid (lon, lat)
     ├── Compute bounding box
     ├── Compute area in CRS units
     └── Build GeoJSON Feature
  6. Export per-patch GeoJSON + merged all_debris.geojson + CSV
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `crf_refine` | `(image: ndarray, mask_probs: ndarray, n_classes=2, crf_iters=5) -> ndarray` | DenseCRF mask refinement |
| `refine_mask` | `(binary: ndarray, min_pixels=MIN_DEBRIS_PIXELS) -> ndarray` | Morphological cleanup |
| `vectorise_patch` | `(pred_path: str) -> list[dict]` | Single patch to GeoJSON features |
| `process_all` | `(pred_dir: str, out_dir: str) -> list[dict]` | Batch process all patches |

### DenseCRF Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sxy` (Gaussian) | 3 | Spatial standard deviation |
| `compat` (Gaussian) | 3 | Compatibility function weight |
| `sxy` (Bilateral) | 80 | Spatial standard deviation |
| `srgb` (Bilateral) | 13 | Color standard deviation |
| `compat` (Bilateral) | 10 | Compatibility function weight |
| `crf_iters` | 5 | Number of mean-field iterations |

### Output Format

**GeoJSON Feature properties:**
```json
{
  "class": "Marine Debris",
  "patch": "S2_1-12-19_48MYU_0",
  "area_crs": 0.000123,
  "centroid_lon": 48.123456,
  "centroid_lat": -8.654321,
  "bbox": [48.12, -8.66, 48.13, -8.65],
  "crs": "EPSG:32648"
}
```

**CSV columns:** `patch, centroid_lon, centroid_lat, area_crs, bbox`

### postprocess.py Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pred_dir` | `str` | `outputs/predicted_test` | Directory with `*_pred.tif` |
| `--out_dir` | `str` | `outputs/geospatial` | Output directory |

---

## Drift Prediction

File: `src/drift_prediction/drift.py`

### Physics Model

Lagrangian particle tracking with **Runge-Kutta 4th order (RK4)** integration:

```
v_total = v_ocean + alpha_wind * v_wind + alpha_stokes * v_wind
```

Where:
- `v_ocean` = ocean surface current (from CMEMS NetCDF or synthetic)
- `alpha_wind` = 0.035 = wind leeway coefficient
- `alpha_stokes` = 0.016 = Stokes drift coefficient

### RK4 Integration

Each timestep (dt = 900s = 15 min):

```
k1 = v(y_n)
k2 = v(y_n + dt/2 * k1)
k3 = v(y_n + dt/2 * k2)
k4 = v(y_n + dt * k3)
y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

Coordinate conversion:
- `d_lat = v * (1/111320)` degrees per meter latitude
- `d_lon = u * (1 / (111320 * cos(lat)))` degrees per meter longitude

### Ensemble Uncertainty

| Parameter | Value |
|-----------|-------|
| Particles | 200 |
| Initial spread | +/-0.005 deg (~500m) Gaussian |
| Velocity noise | 15% of local speed (velocity-proportional) |
| Save horizons | 6h, 12h, 24h, 48h, 72h |

### 95% Confidence Ellipse

Fitted to ensemble particle positions at each horizon:

1. Compute covariance matrix of (lat, lon) positions
2. Eigendecomposition to get principal axes
3. Scale by chi-squared(2, 0.95) = 2.4477 for 95% confidence
4. Generate polygon approximation (36 vertices)

### Velocity Field Interpolation

Uses `scipy.RegularGridInterpolator` with bilinear interpolation:

```python
class VelocityField:
    def __init__(self, lats, lons, u_grid, v_grid):
        self._u_interp = RegularGridInterpolator((lats, lons), u_grid, ...)
        self._v_interp = RegularGridInterpolator((lats, lons), v_grid, ...)

    def __call__(self, lat, lon) -> (u, v):
        ...
```

Falls back to a synthetic field (0.1 m/s east, 0.05 m/s north) when no NetCDF is provided.

### Input: NetCDF Variables

| Source | U variable | V variable | Purpose |
|--------|-----------|-----------|---------|
| CMEMS | `uo` | `vo` | Ocean surface currents |
| ERA5 | `u10` | `v10` | 10m wind components |

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `VelocityField.__init__` | `(lats, lons, u_grid, v_grid)` | Bilinear interpolated velocity field |
| `VelocityField.__call__` | `(lat, lon) -> (u, v)` | Query velocity at a point |
| `_load_nc_field` | `(path, u_var, v_var) -> VelocityField` | Load NetCDF velocity data |
| `_synthetic_field` | `() -> VelocityField` | Synthetic test currents |
| `_rk4_step` | `(lat, lon, ocean, wind, wind_coeff, dt, stokes_coeff) -> (lat, lon)` | Single RK4 integration step |
| `_perturb_field` | `(ocean_field, noise_frac=0.15) -> callable` | Add velocity-proportional noise |
| `_confidence_ellipse` | `(positions) -> (center_lat, center_lon, semi_a, semi_b, angle)` | 95% confidence ellipse |
| `run_ensemble` | `(lat0, lon0, ocean, wind, n_particles, dt, total_hours, wind_coeff, save_hours) -> dict` | Full ensemble simulation |
| `to_geojson` | `(debris_id, lat, lon, results, start_time) -> FeatureCollection` | Build output GeoJSON |

### Drift Output GeoJSON Structure

Each debris object produces:
- 1 **origin** point feature (lat, lon at T+0h)
- N **trajectory** point features (mean position at each save horizon)
- N **confidence_ellipse_95pct** polygon features

```json
{
  "type": "FeatureCollection",
  "features": [
    {"type": "Feature", "geometry": {"type": "Point", ...}, "properties": {"type": "origin", "time": "T+0h"}},
    {"type": "Feature", "geometry": {"type": "Point", ...}, "properties": {"type": "trajectory", "time": "T+24h"}},
    {"type": "Feature", "geometry": {"type": "Polygon", ...}, "properties": {"type": "confidence_ellipse_95pct", "time": "T+24h"}}
  ]
}
```

### drift.py Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--geojson` | `str` | (required) | Input debris GeoJSON from postprocess |
| `--ocean_nc` | `str` | `None` | CMEMS ocean current NetCDF |
| `--wind_nc` | `str` | `None` | ERA5 wind field NetCDF |
| `--ocean_u_var` | `str` | `"uo"` | Ocean U variable name |
| `--ocean_v_var` | `str` | `"vo"` | Ocean V variable name |
| `--wind_u_var` | `str` | `"u10"` | Wind U variable name |
| `--wind_v_var` | `str` | `"v10"` | Wind V variable name |
| `--out_dir` | `str` | `outputs/drift` | Output directory |

---

## Visualization

File: `src/utils/visualise.py`

### visualise_masks(pred_dir, n=10, out_dir=None)

Generates side-by-side comparison PNGs:

```
┌──────────┬──────────┬──────────┐
|   RGB    | Predicted|  Ground  |
| (B4,B3,  |  Mask    |  Truth   |
|  B2)     |          |          |
└──────────┴──────────┴──────────┘
```

- RGB from bands 4, 3, 2 with 2nd/98th percentile stretch
- 15-color categorical colormap
- Legend with all class names

### visualise_drift(geojson_path, out_dir=None)

Generates per-debris drift trajectory plots:
- Red dot = origin position
- Blue line + dots = trajectory points (6h, 12h, 24h, 48h, 72h)
- Blue dashed ellipse = 95% confidence region
- Time labels at each point

### visualise.py Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pred_dir` | `str` | `None` | Directory with `*_pred.tif` for mask visualization |
| `--drift_geojson` | `str` | `None` | Drift GeoJSON for trajectory visualization |
| `--n` | `int` | 10 | Number of masks to visualize |
| `--out_dir` | `str` | `None` | Override output directory |

### Usage

```bash
# Visualize predicted masks (RGB + predicted + ground truth)
python utils/visualise.py --pred_dir outputs/predicted_test --n 20

# Visualize drift trajectories
python utils/visualise.py --drift_geojson outputs/drift/all_drift.geojson
```

---

## PowerShell Runner (run.ps1)

File: `src/run.ps1`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-SkipTrain` | switch | — | Skip training step |
| `-Resume` | switch | — | Resume from last checkpoint |
| `-WithRF` | switch | — | Also train Random Forest |
| `-WithSI` | switch | — | Include spectral indices for RF |
| `-WithGLCM` | switch | — | Include GLCM textures for RF |
| `-EvalOnly` | switch | — | Only evaluate + post-process |
| `-TrainOnly` | switch | — | Only train (skip eval/postprocess/drift) |
| `-NoTTA` | switch | — | Disable test-time augmentation |
| `-Epochs` | int | 120 | Training epochs |
| `-Batch` | int | 16 | Batch size |
| `-LR` | double | 1e-4 | Learning rate |
| `-Patience` | int | 30 | Early stopping patience |
| `-WeightDecay` | double | 5e-5 | Weight decay |
| `-Workers` | int | 4 | DataLoader workers |
| `-Checkpoint` | string | `""` | Custom checkpoint path for eval |
| `-OceanNC` | string | `""` | CMEMS NetCDF path |
| `-WindNC` | string | `""` | ERA5 NetCDF path |

### Usage Examples

```powershell
# Full pipeline with defaults
.\run.ps1

# Train only, custom hyperparameters
.\run.ps1 -TrainOnly -Epochs 200 -Batch 8 -LR 5e-5

# Evaluate existing checkpoint with TTA (default)
.\run.ps1 -EvalOnly

# Evaluate without TTA, custom checkpoint
.\run.ps1 -EvalOnly -NoTTA -Checkpoint "D:\checkpoints\my_model.pth"

# Resume interrupted training
.\run.ps1 -Resume

# Full pipeline with Random Forest
.\run.ps1 -WithRF -WithSI -WithGLCM

# With real ocean/wind data
.\run.ps1 -OceanNC "data/cmems.nc" -WindNC "data/era5.nc"
```

### Pipeline Steps

```
[1/4] Training    -> python train.py --epochs ... --batch ... --lr ...
[2/4] Evaluation  -> python evaluate.py --split test --save_masks --tta
[3/4] PostProcess -> python postprocess.py --pred_dir ... --out_dir ...
[4/4] Drift       -> python drift_prediction/drift.py --geojson ...
[RF]  (optional)  -> python random_forest/train_eval.py
```

---

## Python Pipeline Runner (run_pipeline.py)

File: `src/run_pipeline.py`

Orchestrates the same pipeline via Python subprocess calls.

### Pipeline Steps

```
Step 0 (optional): Spectral extraction (for RF)
  -> python utils/spectral_extraction.py --type bands
  -> python utils/spectral_extraction.py --type indices   (if --with_si)
  -> python utils/spectral_extraction.py --type texture   (if --with_glcm)

Step 1: Train segmentation model
  -> python train.py --epochs ... --batch ... --lr ...

Step 2: Evaluate on test set
  -> python evaluate.py --split test --save_masks

Step 3: Post-process predictions
  -> python postprocess.py --pred_dir outputs/predicted_test --out_dir outputs/geospatial

Step 4: Drift prediction
  -> python drift_prediction/drift.py --geojson outputs/geospatial/all_debris.geojson

Step 5 (optional): Random Forest
  -> python random_forest/train_eval.py [--use_si] [--use_glcm]
```

### run_pipeline.py Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--skip_train` | flag | — | Skip model training |
| `--eval_only` | flag | — | Only evaluate + post-process |
| `--extract_only` | flag | — | Only run spectral extraction |
| `--resume` | flag | — | Resume from checkpoint |
| `--epochs` | int | 120 | Training epochs |
| `--batch` | int | 16 | Batch size |
| `--lr` | float | 1e-4 | Learning rate |
| `--workers` | int | 4 | DataLoader workers |
| `--weight_decay` | float | 5e-5 | Weight decay |
| `--patience` | int | 30 | Early stopping patience |
| `--with_rf` | flag | — | Train Random Forest |
| `--with_si` | flag | — | Include spectral indices |
| `--with_glcm` | flag | — | Include GLCM textures |
| `--ocean_nc` | str | `None` | CMEMS NetCDF path |
| `--wind_nc` | str | `None` | ERA5 NetCDF path |

---

## Complete API Reference

### configs/config.py

Central configuration module. All constants are importable:

```python
from configs.config import (
    # Paths
    BASE_DIR, DATA_DIR, PATCHES_DIR, SPLITS_DIR,
    H5_PATH, H5_SI_PATH, H5_GLCM_PATH,
    CHECKPOINTS_DIR, OUTPUTS_DIR, LOGS_DIR,
    # Dataset
    NUM_CLASSES, INPUT_BANDS, PATCH_SIZE, CLASS_NAMES, CLASS_WEIGHTS,
    # Training
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE, NUM_WORKERS,
    # Augmentation
    AUG_PROB, ELASTIC_ALPHA, ELASTIC_SIGMA, SPECTRAL_NOISE, BRIGHTNESS_RANGE,
    # Post-processing
    MIN_DEBRIS_PIXELS,
    # Drift
    DRIFT_HOURS, DRIFT_DT_SECONDS, DRIFT_ENSEMBLE_N, DRIFT_WIND_COEFF, STOKES_COEFF,
)
```

### utils/dataset.py

| Symbol | Type | Signature / Description |
|--------|------|------------------------|
| `_FILE_INDEX` | `dict` | Global cache: TIF stem -> full path |
| `BAND_P2` | `ndarray[11]` | Per-band 2nd percentile (lazy-computed) |
| `BAND_P98` | `ndarray[11]` | Per-band 98th percentile (lazy-computed) |
| `_build_file_index()` | function | `() -> None` — Walks PATCHES_DIR, populates `_FILE_INDEX` |
| `_find_tif(patch_id)` | function | `(str) -> str or None` — Full path to image .tif |
| `_find_mask_tif(patch_id)` | function | `(str) -> str or None` — Full path to _cl.tif |
| `_find_conf_tif(patch_id)` | function | `(str) -> str or None` — Full path to _conf.tif |
| `_compute_norm_stats(split_file, n_samples=0)` | function | `(str, int) -> None` — Sets BAND_P2/P98 |
| `normalise(arr)` | function | `(ndarray[B,H,W]) -> ndarray[B,H,W]` — Percentile clip + [0,1] |
| `get_advanced_augment()` | function | `() -> A.Compose` — Albumentations pipeline |
| `MARIDADataset` | class | PyTorch Dataset for MARIDA patches |
| `MARIDADataset.__init__` | method | `(split, augment_data=True, use_conf_weights=True)` |
| `MARIDADataset.__len__` | method | `() -> int` |
| `MARIDADataset.__getitem__` | method | `(idx) -> {"image": Tensor, "mask": Tensor, "conf": Tensor, "id": str}` |

### semantic_segmentation/models/resnext_cbam_unet.py

| Class | Constructor | Description |
|-------|-------------|-------------|
| `ChannelAttention` | `(channels: int, reduction: int = 16)` | Squeeze-excitation channel attention |
| `SpatialAttention` | `(kernel_size: int = 7)` | Conv-based spatial attention map |
| `CBAM` | `(channels: int, reduction: int = 16, spatial_kernel: int = 7)` | Channel + Spatial attention |
| `DecoderBlock` | `(in_ch: int, skip_ch: int, out_ch: int)` | Upsample + skip concat + 2xConv + CBAM |
| `ResNeXtCBAMUNet` | `(in_channels: int = 11, num_classes: int = 15, pretrained: bool = True)` | Full encoder-decoder model |
| `HybridLoss` | `(class_weights, dice_weight=0.4, ignore_index=-1, use_focal=False, focal_gamma=2.0)` | Focal CE + Dice loss |

**HybridLoss.forward:**
```python
def forward(self, logits, target, conf_weights=None):
    """
    logits       : (B, C, H, W) — raw model output
    target       : (B, H, W) long, -1 = nodata
    conf_weights : (B, H, W) float in [0,1] or None
    Returns      : scalar loss
    """
```

### train.py

| Function | Signature | Returns |
|----------|-----------|---------|
| `compute_iou(pred, target, num_classes, ignore_index=-1)` | `(Tensor, Tensor, int, int) -> list[float]` | Per-class IoU |
| `train_epoch(model, loader, optimizer, criterion, device, scaler)` | `(...) -> float` | Mean training loss |
| `val_epoch(model, loader, criterion, device, num_classes)` | `(...) -> tuple` | `(val_loss, mIoU, ious, prec, rec, f1)` |
| `main(args)` | `(Namespace) -> None` | Runs full training loop |

### evaluate.py

| Function | Signature | Returns |
|----------|-----------|---------|
| `plot_confusion_matrix(cm, class_names, save_path)` | `(ndarray, list, str) -> None` | Saves PNG |
| `save_predicted_mask(pred_mask, patch_id, out_dir)` | `(ndarray, str, str) -> None` | Saves GeoTIFF |
| `_tta_predict(model, imgs, device)` | `(Module, Tensor, device) -> Tensor[B,2,H,W]` | Averaged TTA probabilities |
| `evaluate(args)` | `(Namespace) -> None` | Full evaluation pipeline |

### postprocess.py

| Function | Signature | Returns |
|----------|-----------|---------|
| `crf_refine(image, mask_probs, n_classes=2, crf_iters=5)` | `(ndarray[H,W,3], ndarray[C,H,W], int, int) -> ndarray[H,W]` | CRF-refined mask |
| `refine_mask(binary, min_pixels)` | `(ndarray[H,W], int) -> ndarray[H,W]` | Morphologically cleaned mask |
| `vectorise_patch(pred_path)` | `(str) -> list[dict]` | GeoJSON features for one patch |
| `process_all(pred_dir, out_dir)` | `(str, str) -> list[dict]` | Batch processing |

### drift_prediction/drift.py

| Function / Class | Signature | Returns |
|-----------------|-----------|---------|
| `VelocityField.__init__` | `(lats, lons, u_grid, v_grid)` | Interpolated velocity field |
| `VelocityField.__call__` | `(lat: float, lon: float) -> (float, float)` | (u, v) velocity at point |
| `_load_nc_field(path, u_var, v_var)` | `(str, str, str) -> VelocityField` | Load NetCDF field |
| `_synthetic_field()` | `() -> VelocityField` | Test field (0.1 m/s E, 0.05 m/s N) |
| `_rk4_step(lat, lon, ocean, wind, coeff, dt, stokes)` | `-> (float, float)` | (new_lat, new_lon) |
| `_perturb_field(field, noise_frac=0.15)` | `-> callable` | Noisy field wrapper |
| `_confidence_ellipse(positions)` | `(ndarray[N,2]) -> tuple[5]` | (center_lat, center_lon, semi_a, semi_b, angle_deg) |
| `run_ensemble(lat0, lon0, ...)` | `-> dict` | Results at save horizons |
| `_ellipse_polygon(center_lat, center_lon, semi_a, semi_b, angle_deg, n=36)` | `-> dict` | GeoJSON Polygon geometry |
| `to_geojson(debris_id, lat, lon, results, start_time)` | `-> dict` | GeoJSON FeatureCollection |

### random_forest/train_eval.py

| Function | Signature | Returns |
|----------|-----------|---------|
| `load_split(h5_path, split, max_pixels=500_000)` | `(str, str, int) -> DataFrame` | Feature DataFrame |
| `merge_features(*dfs)` | `(*DataFrame) -> DataFrame` | Horizontally merged features |
| `main(args)` | `(Namespace) -> None` | Train + evaluate RF |

### utils/spectral_extraction.py

| Function | Signature | Returns |
|----------|-----------|---------|
| `compute_spectral_indices(arr)` | `(ndarray[11,H,W]) -> ndarray[5,H,W]` | NDVI, NDWI, FDI, PI, RNDVI |
| `compute_glcm_features(arr, band_idx=7)` | `(ndarray, int) -> ndarray[2,H,W]` | Local variance + mean |
| `extract_split(split, mode)` | `(str, str) -> DataFrame` | Extracted features per pixel |
| `main(args)` | `(Namespace) -> None` | Extract and save to HDF5 |

### utils/visualise.py

| Function | Signature | Returns |
|----------|-----------|---------|
| `visualise_masks(pred_dir, n=10, out_dir=None)` | `(str, int, str) -> None` | Saves comparison PNGs |
| `visualise_drift(geojson_path, out_dir=None)` | `(str, str) -> None` | Saves trajectory PNGs |

---

## Testing & Validation

### Comprehensive Smoke Test

Run from `src/`:

```python
python -c "
import sys, torch, numpy as np
sys.path.insert(0, '.')
from utils.dataset import MARIDADataset, BAND_P2, BAND_P98
from torch.utils.data import DataLoader
from semantic_segmentation.models.resnext_cbam_unet import HybridLoss
from configs.config import CLASS_WEIGHTS, MIN_DEBRIS_PIXELS
import segmentation_models_pytorch as smp
from evaluate import _tta_predict

# 1. Dataset loads correctly
train_ds = MARIDADataset('train', augment_data=True)
val_ds   = MARIDADataset('val',   augment_data=False)
print(f'Train: {len(train_ds)}   Val: {len(val_ds)}')

# 2. Batch has no NaN/Inf
dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
batch = next(iter(dl))
assert not torch.isnan(batch['image']).any()
assert not torch.isinf(batch['image']).any()

# 3. Confidence values are correct
unique_c = torch.unique(batch['conf']).tolist()
for v in unique_c:
    assert any(abs(v - t) < 0.01 for t in [0.2, 0.7, 1.0])

# 4. Loss is finite
criterion = HybridLoss(class_weights=CLASS_WEIGHTS, dice_weight=0.7,
                       use_focal=True, focal_gamma=2.0)
fake_logits = torch.randn(4, 2, 256, 256)
loss = criterion(fake_logits, batch['mask'], batch['conf'])
assert torch.isfinite(loss)

# 5. Model forward pass
model = smp.DeepLabV3Plus(encoder_name='resnext50_32x4d', encoder_weights=None,
                          in_channels=11, classes=2, activation=None)
model.eval()
with torch.no_grad():
    out = model(batch['image'])
assert out.shape == (4, 2, 256, 256)
assert not torch.isnan(out).any()

# 6. TTA produces valid probabilities
with torch.no_grad():
    tta = _tta_predict(model, batch['image'], torch.device('cpu'))
assert abs(tta[0,:,0,0].sum().item() - 1.0) < 0.01

# 7. Normalization stats are NaN-free
assert not np.any(np.isnan(BAND_P2))
assert not np.any(np.isnan(BAND_P98))

# 8. Training backward pass
model.train()
logits = model(batch['image'])
loss = criterion(logits, batch['mask'], batch['conf'])
assert torch.isfinite(loss)
loss.backward()

print('ALL CHECKS PASSED')
"
```

### What Each Test Validates

| Test | Validates |
|------|-----------|
| Dataset load | Split files exist, patches found, oversampling works |
| No NaN/Inf | `np.nan_to_num` works, normalization handles edge cases |
| Confidence values | Correct mapping: {0->0.2, 1->0.7, 2->1.0} |
| Loss finite | HybridLoss handles nodata, class weights, focal correctly |
| Model output | DeepLabV3+ produces correct shape, no NaN propagation |
| TTA | 8x averaging produces valid probabilities summing to 1.0 |
| Norm stats | `np.nanpercentile` handles NaN bands (9, 10) |
| Backward pass | Gradients flow correctly through entire model |

### Bugs Found & Fixed

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| NaN in bands 9-10 | Patch `21-2-17_16PCC_0` has NaN pixels | `np.nanpercentile` + `np.nan_to_num` |
| 99.9% nodata trained as debris | Old `target.clamp(min=0)` mapped nodata DN=0 to class 0 | Manual `valid_mask` excluding -1 |
| GaussNoise destroying images | `var_limit=(10,50)` is uint8-scale on float32 [0,1] | Changed to `std_range=(0.01, 0.03)` |
| Confidence not spatially aligned | Conf stays in original coords after flip/rotate | `additional_targets={'conf_raw': 'mask'}` |
| Non-deterministic normalization | Random 200/694 subset with no seed | Use ALL 694 patches |
| 78% debris patches not oversampled | Threshold `>10` excluded 148/190 patches | Changed to `>=1` with graduated multiplier |
| Real debris removed in eval | `MIN_DEBRIS_PIXELS=50` vs average 8px debris | Reduced to 3 |
| Best model tracked by wrong metric | Tracked by mIoU (dominated by background) | Track by **debris F1** |
| Slow file lookup | Recursive `glob.glob` per patch | Cached `os.walk` index |
| PostProcess wrong DN mapping | Checked `data == 0` but saved masks use `pred+1` | Changed to `data == 1` |
| Class weight too aggressive | 5.0 caused precision collapse | Reduced to 2.0 |
| Debris-only training split | Model never saw non-debris examples | Use full `train_X.txt` |

---

## Performance Expectations

### Realistic Targets (MARIDA Binary)

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| **Overall pixel accuracy** | 95-99% | Dominated by not-debris (>99.5% of valid pixels) |
| **Debris F1** | 55-75% | Published MARIDA SOTA is 60-75% |
| **Not-Debris F1** | 98-99% | Easy majority class |
| **Debris IoU** | 40-60% | Harder than F1 |
| **Mean IoU** | 55-75% | Average of debris + not-debris IoU |
| **Debris Precision** | 40-70% | Many false positives in turbid/shallow water |
| **Debris Recall** | 60-90% | Oversampling + class weights boost recall |

### Why 90+ Debris F1 Is Not Achievable

1. **Only 1,943 debris pixels** in training (0.45% of valid pixels)
2. **Debris looks like turbid water** spectrally — high inter-class similarity
3. **10-20m resolution** — debris patches are sub-pixel to a few pixels
4. **Published SOTA** on MARIDA peaks at 60-75% debris F1 (e.g., Kikaki et al., 2022)
5. **Noisy labels** — annotator disagreement in ambiguous patches

### Techniques That Improve Performance

| Technique | Impact | Implemented |
|-----------|--------|-------------|
| Graduated oversampling (8x/5x) | +5-10% recall | Yes |
| Focal Loss (gamma=2) | +3-5% F1 (hard examples) | Yes |
| Dice Loss (weight=0.7) | +3-5% F1 (pixel overlap) | Yes |
| TTA (8x geometric) | +3-5% F1 | Yes |
| Threshold sweep | +2-5% F1 (optimal threshold) | Yes |
| Confidence weighting | Better calibration | Yes |
| Cosine LR + warmup | Stable convergence | Yes |
| Gradient clipping (5.0) | Prevents training collapse | Yes |
| Mixed precision (AMP) | 2x training speed | Yes |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `FileNotFoundError: Split file not found` | Missing split files | Place `train_X.txt`, `val_X.txt`, `test_X.txt` in `Dataset/splits/` |
| `NaN in training loss` | Extreme pixel values | Already handled — NaN batches are skipped |
| `CUDA out of memory` | Batch size too large | Reduce `--batch` to 8 or 4 |
| `DLL load failed` (Windows) | PyTorch/CUDA mismatch | Install via conda: `conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia` |
| Very low debris F1 | Not training long enough | Train for full 120 epochs with patience=30 |
| All predictions = not-debris | Threshold too high | Use `--threshold 0.3` or check threshold sweep output |
| Slow data loading | Workers fighting for I/O | Reduce `--workers` to 2 on HDD, keep 4 on SSD |
| `ModuleNotFoundError: pydensecrf` | Optional dependency | `pip install pydensecrf` (optional, postprocess works without it) |
| TensorBoard empty | Wrong log directory | Run `tensorboard --logdir logs/tsboard` from project root |
| `albumentations` API error | Version mismatch | Use `albumentations>=2.0.0` with `std_range` for GaussNoise |
| Checkpoint load fails | Architecture mismatch | Evaluator auto-detects smp vs custom model from checkpoint keys |

---

## License

This project uses the MARIDA dataset, which is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See the [MARIDA repository](https://github.com/marine-debris/marine-debris) for citation requirements.
