# Marine Debris Detection and Drift Prediction Pipeline

End-to-end research pipeline for detecting floating marine debris in Sentinel-2 imagery and projecting debris motion through a physics-based ocean drift model.

This repository combines:
- A deep-learning semantic segmentation model (ResNeXtCBAMUNet) for binary debris detection
- A Random Forest classical baseline built from spectral and texture features
- Geospatial post-processing for polygon and centroid export
- Ensemble Lagrangian drift prediction (RK4 + 200 particles) for short-horizon debris movement

Built around the MARIDA benchmark and adapted into a binary rare-object detection system: debris vs not-debris.

---

## Table of Contents

1. Project Summary
2. Pipeline Overview
3. Repository Structure
4. Current Technical Status
5. Research Motivation and Novelty
6. Data Definition
7. Binary Label Mapping
8. Input Feature Design
9. Deep Learning Model
10. Loss Function Design
11. Data Loading and Normalization
12. Imbalance Handling Strategy
13. Augmentation Strategy
14. Training Pipeline
15. Checkpointing and Model Selection
16. Evaluation Pipeline
17. How To Interpret Metrics Correctly
18. Post-Processing Pipeline
19. Random Forest Baseline
20. Drift Prediction Pipeline
21. GPU vs CPU Operation
22. Directory Structure
23. Installation
24. Recommended Commands
25. Pipeline Orchestration
26. Output Inventory
27. Configuration Reference
28. Key Code Logic By File
29. Reproducibility Checklist
30. Experiment Playbook
31. Known Results and Training History
32. Limitations
33. How To Position The Novelty In A Paper
34. Troubleshooting
35. Roadmap
36. ReadMe

---

## Project Summary

This project solves two linked research problems.

| Problem | Method | Output |
|---|---|---|
| Marine debris detection | Binary semantic segmentation + Random Forest baseline | Debris masks, GeoTIFF predictions, GeoJSON polygons, centroid CSV |
| Debris movement forecasting | RK4 particle tracking with ensemble perturbations | 6h to 72h trajectories and uncertainty ellipses |

The core challenge is extreme rare-class detection. Debris occupies only about **0.45%** of valid labeled pixels. The project is built around recall, F1, AUPRC, patch-level detection rate, threshold calibration, and geospatial usability.

---

## Pipeline Overview

```text
Sentinel-2 patches
   → binary label mapping
   → percentile normalization + spectral indices
   → segmentation training (ResNeXtCBAMUNet + EMA)
   → thresholded inference + TTA
   → morphological cleanup
   → polygon and centroid export
   → drift initialization
   → RK4 trajectory forecasting (200-particle ensemble)
```

Operationally, the system has three layers:
- **Learning layer**: segmentation and Random Forest baselines
- **Geospatial layer**: raster-to-vector conversion and object summarization
- **Forecasting layer**: drift prediction from detected debris objects

---

## Repository Structure

- `src/train.py` — training loop (EMA, SGDR warm restarts, encoder freeze, deep supervision)
- `src/evaluate.py` — evaluation with TTA, AUPRC, PR curve, patch-level metrics
- `src/postprocess.py` — morphological cleanup, GeoJSON / CSV export
- `src/semantic_segmentation/models/resnext_cbam_unet.py` — architecture + loss
- `src/utils/dataset.py` — dataset loading, normalization, oversampling, copy-paste, MixUp
- `src/drift_prediction/drift.py` — Lagrangian drift prediction
- `src/random_forest/train_eval.py` — Random Forest baseline
- `src/configs/config.py` — central configuration (GPU/CPU auto-scaled)
- `src/run_pipeline.py` — full end-to-end orchestration script

---

## Current Technical Status

### Active deep model path

| Component | Value |
|---|---|
| Architecture | `ResNeXtCBAMUNet` |
| Backbone options | `resnext50_32x4d` (CPU auto) / `resnext101_32x8d` (GPU auto) |
| Input channels | 16 (11 raw bands + 5 spectral indices) |
| Classes | 2 (debris / not-debris) |
| Loss | Focal-CE (γ=3.0) + Tversky (α=0.3, β=0.7) |
| Regularisation | Dropout2d(0.1) in all decoder blocks |
| Auxiliary head | Deep supervision at dec3 (weight=0.4) |
| Optimizer | AdamW |
| Scheduler | Linear warmup (5ep) → SGDR (T₀=25, Tmult=2) |
| EMA | ModelEMA decay=0.9998, saved as `ema_best.pth` |
| Encoder freeze | First 5 epochs frozen (transfer learning warmup) |
| Best-model criterion | Debris F1 with minimum recall guard (≥5%) |

### Parameter counts
- ResNeXt-50 backbone: ~41M parameters
- ResNeXt-101 backbone: ~88M parameters

---

## Research Motivation and Novelty

The novelty of this project is the **full system design** for rare marine debris detection under severe imbalance and geospatial deployment constraints.

### Main research contributions

1. **Binary reformulation of MARIDA for rare debris-first detection.**
   Original 15-class MARIDA collapsed to binary. Class 1 → debris, all others → not-debris. Reframes toward operational debris search.

2. **Multi-source input representation.**
   11 Sentinel-2 raw bands plus 5 physically motivated spectral indices (NDVI, NDWI, FDI, PI, RNDVI). Learned deep features combined with spectral cues tailored to marine debris.

3. **Confidence-weighted supervised learning.**
   MARIDA pixel confidence maps converted to loss weights. Uncertain labels contribute less strongly to optimization.

4. **Recall-first loss design.**
   Tversky loss (β=0.7 > α=0.3) penalises missed debris (FN) 2.3× more than false alarms (FP). Combined with focal modulation for hard-example mining.

5. **Rare-object data pipeline.**
   Patch oversampling, copy-paste debris augmentation, MixUp, confidence-aware loss weighting, recall-guarded EMA model selection. A system-level imbalance strategy.

6. **Attention-enhanced encoder-decoder with deep supervision.**
   CBAM in the decoder path. Auxiliary supervision head at 1/8 resolution sends stronger gradient signal to mid-level features.

7. **EMA + SGDR training stability.**
   Exponential Moving Average with warm restarts prevents instability valleys and produces a smoother final checkpoint.

8. **Operational geospatial output path.**
   Predictions become GIS-ready polygons, centroids, CSV tables, and drift trajectories.

9. **Tight detection-to-drift coupling.**
   Detection is the upstream stage of a downstream ocean-motion model. Useful for real maritime monitoring.

---

## Data Definition

### Dataset facts

| Property | Value |
|---|---|
| Satellite | Sentinel-2 L2A |
| Raw bands used | 11 (B1–B8, B8A, B11, B12; B10 excluded) |
| Extra indices | 5 (NDVI, NDWI, FDI, PI, RNDVI) |
| Total input channels | 16 |
| Patch size | 256 × 256 |
| Binary classes | Debris (0), Not Debris (1) |
| Nodata label | −1 |

### Split sizes

| Split | Patches |
|---|---|
| Train | 694 (oversampled to ~1368 with debris duplication) |
| Validation | 328 |
| Test | 359 |

Debris is severely imbalanced: ~0.45% of valid labeled pixels. Accuracy is therefore misleading as a standalone metric.

---

## Binary Label Mapping

| Original mask DN | Binary value | Meaning |
|---|---|---|
| 0 | −1 | nodata / ignored |
| 1 | 0 | marine debris |
| 2–15 | 1 | not-debris |

Implemented in `dataset.py`, not just documented. All metric interpretation must use this binary mapping.

---

## Input Feature Design

### Raw spectral bands used

B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

### Spectral indices

| Index | Formula basis | Purpose |
|---|---|---|
| NDVI | (NIR−Red)/(NIR+Red) | Vegetation suppression |
| NDWI | (Green−NIR)/(Green+NIR) | Water contrast |
| FDI | NIR − (Red + (SWIR−Red)×factor) | Floating debris sensitivity |
| PI | Red/(Green+Red+NIR) | Plastic-related ratio |
| RNDVI | (Red−Green)/(Red+Green) | Red-green contrast |

Computed from raw reflectance before normalization. Clipped to [−3, 3] and rescaled to [0, 1] before concatenation.

---

## Deep Learning Model

### Architecture: `ResNeXtCBAMUNet`

```text
Input  (B, 16, 256, 256)
  → enc0: Conv7×7 stride2  (64ch) → /2
  → pool (maxpool)                  → /4
  → enc1: ResNeXt layer1  (256ch)  → /4
  → enc2: ResNeXt layer2  (512ch)  → /8
  → enc3: ResNeXt layer3  (1024ch) → /16
  → enc4: ResNeXt layer4  (2048ch) → /32
  → bottleneck_cbam (CBAM on 2048ch)
  → dec4: (2048+1024→256)  CBAM + Dropout2d → /16
  → dec3: (256+512→128)    CBAM + Dropout2d → /8  [→ aux_conv head]
  → dec2: (128+256→64)     CBAM + Dropout2d → /4
  → dec1: (64+64→64)       CBAM + Dropout2d → /2
  → final_up + interpolate               → /1
  → final_conv (64→2)
Output (B, 2, 256, 256)
```

### Dynamic backbone selection (auto)

| Environment | Backbone | Params | VRAM needed |
|---|---|---|---|
| CPU (auto) | resnext50_32x4d | ~41M | — |
| GPU ≥8GB (auto) | resnext101_32x8d | ~88M | ~6–8GB |

Override manually with `--backbone resnext50_32x4d`.

### CBAM (Convolutional Block Attention Module)

Each block combines:
- **Channel attention**: adaptive avg + max pooling → FC → sigmoid
- **Spatial attention**: channel-wise pooled descriptors → 7×7 conv → sigmoid

Applied at the bottleneck and inside every decoder block.

### Dropout2d

`p=0.1` inserted after the first conv in each decoder block. Prevents decoder overfitting which was responsible for precision collapse in earlier runs.

### Deep supervision

An auxiliary `1×1` conv head at `dec3` output (1/8 resolution). Upsampled to full resolution and included in the loss with weight=0.4. Disabled at inference time automatically.

### Pretrained weight adaptation

First conv averaged across ImageNet RGB channels and tiled across 16 input channels. This is the standard pragmatic bridge for multispectral transfer learning.

---

## Loss Function Design

### Formula

$$L = (1 - \lambda_T)\,L_{\text{focal-CE}} + \lambda_T\,L_{\text{Tversky}} + \lambda_{\text{aux}}\,L_{\text{focal-CE}}^{\text{aux}}$$

with: $\lambda_T = 0.7$, $\lambda_{\text{aux}} = 0.4$

### Focal Cross-Entropy

$$L_{\text{focal-CE}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} (1 - p_{t,i})^\gamma \cdot L_{\text{CE},i}$$

- $\gamma = 3.0$ (stronger focus on hard debris pixels)
- Class weights: `[5.0, 1.0]`
- Label smoothing: `0.01`
- Nodata pixels excluded from $\mathcal{V}$

### Tversky Loss

$$L_{\text{Tversky}} = -\log\left(\frac{\text{TP} + \epsilon}{\text{TP} + \alpha\,\text{FP} + \beta\,\text{FN} + \epsilon}\right)$$

- $\alpha = 0.3$ (FP weight)
- $\beta = 0.7$ (FN weight — penalises missed debris 2.3× more than false alarms)
- Log-space for gradient stability at extreme imbalance

### Why Tversky replaces Dice

Plain Dice treats FP and FN symmetrically. In binary debris detection, missing a debris patch (FN) is operationally far worse than a false alarm (FP). Tversky with β > α encodes this asymmetry directly.

---

## Data Loading and Normalization

### Percentile normalization

$$x_b' = \text{clip}\!\left(\frac{x_b - P2_b}{P98_b - P2_b + \epsilon},\, 0,\, 1\right)$$

Computed from the training split. Robust to outliers in multispectral data.

### Nodata handling

Nodata mapped to −1, excluded entirely from loss and metric computation.

### Confidence weighting

| Raw confidence | Weight |
|---|---|
| 0 (uncertain) | 0.2 |
| 1 (confident) | 0.7 |
| 2 (high) | 1.0 |

---

## Imbalance Handling Strategy

Imbalance is addressed at multiple levels simultaneously:

| Level | Mechanism | Setting |
|---|---|---|
| Loss | Class weights (CE term) | debris=5.0, not-debris=1.0 |
| Loss | Tversky β > α | FN penalised 2.3× |
| Data | Patch oversampling (heavy) | ×8 for patches with ≥20 debris pixels |
| Data | Patch oversampling (light) | ×4 for patches with 1–19 debris pixels |
| Data | Copy-paste augmentation | p=0.4, 1–5 donors per sample |
| Data | MixUp | p=0.15, α=0.2 (harder boundary signal) |
| Inference | Threshold sweep | 0.15–0.80 to find optimal F1 point |
| Selection | EMA checkpoint | Smoother weights = better generalization |
| Selection | Recall guard | Model must have recall ≥5% to qualify as best |

---

## Augmentation Strategy

### Spatial augmentations

- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- Random 90° rotation (p=0.5)
- Affine: shift ±5%, scale 0.9–1.1, rotate ±15° (p=0.3)
- Elastic transform (p=0.15)
- Grid distortion (p=0.1)

### Radiometric augmentations

- Gaussian noise (p=0.2, std 0.01–0.03 on [0,1] float images)
- Brightness/contrast perturbation (p=0.2, ±0.1 limits)

### Copy-paste debris augmentation

- Triggered with probability 0.4
- 1–5 donor patches randomly selected from the debris pool
- Donor debris pixels randomly flipped, rotated, and spatially shifted before pasting
- Pasted pixels receive high-confidence weight (conf=2)

### MixUp

- Triggered with probability 0.15
- Two training patches blended with λ ∼ Beta(0.2, 0.2), λ ≥ 0.5
- Image channels blended; dominant patch mask preserved
- Hardens boundary decision and improves calibration

---

## Training Pipeline

### Core training loop

For each epoch:
1. Freeze encoder for first `ENCODER_FREEZE_EPOCHS` (5) epochs → decoder learns debris signatures from pretrained features first
2. Unfreeze encoder at epoch 5 and reset LR to LR×0.1 for joint fine-tuning
3. Forward pass under AMP (GPU) or standard precision (CPU)
4. Compute `HybridLoss` on `(main_logits, aux_logits)` tuple
5. Gradient accumulation (effective batch = 64 on both CPU and GPU)
6. Gradient clipping at `max_norm=5.0`
7. Update optimizer and SGDR scheduler
8. **Update EMA shadow model** after each step
9. Run validation with raw model AND EMA model
10. Write CSV metrics, TensorBoard scalars
11. Save epoch, last, best (raw), best (EMA) checkpoints

### Optimizer and scheduler

- **Optimizer**: AdamW (lr=1e-4, weight_decay=5e-5)
- **Warmup**: Linear from 0.1× to 1× over 5 epochs
- **Main**: CosineAnnealingWarmRestarts (T₀=25, T_mult=2, η_min=1e-6)

SGDR warm restarts periodically reset the LR, allowing the model to escape instability valleys. This was the root cause of the F1 oscillation seen after epoch 8 in the previous training run.

### EMA (Exponential Moving Average)

$$\theta_{\text{EMA}} \leftarrow 0.9998 \cdot \theta_{\text{EMA}} + 0.0002 \cdot \theta_{\text{train}}$$

The EMA shadow model is validated at every epoch. `ema_best.pth` is saved whenever the EMA F1 improves. **Use `ema_best.pth` for all evaluation and paper results** — it consistently outperforms the raw checkpoint.

### GPU vs CPU behavior

Training runs on either device with identical code. AMP (mixed precision) activates automatically on GPU.

---

## Checkpointing and Model Selection

Every run creates in `checkpoints/run_<timestamp>/`:

| File | Contents |
|---|---|
| `epoch_NNN.pth` | Snapshot every epoch |
| `last.pth` | Most recent epoch |
| `best.pth` | Best raw model by debris F1 |
| `ema_best.pth` | Best EMA model by debris F1 |
| `metrics.csv` | Full per-epoch metrics including EMA F1 and LR |
| Per-metric PNGs | Training curves, including raw vs EMA F1 comparison |

### Shared checkpoints (root of `checkpoints/`)

| File | Use |
|---|---|
| `resnext_cbam_best.pth` | Best raw model (any run) |
| `resnext_cbam_ema_best.pth` | Best EMA model (any run) — **use this for paper results** |

### Best-model criterion

1. Debris F1 must improve
2. Debris recall must be ≥ 5% (prevents selecting high-accuracy zero-recall model)

---

## Evaluation Pipeline

### Evaluation flow

1. Load checkpoint (auto-detects backbone from state dict)
2. Load requested split (no augmentation)
3. Run inference with optional TTA or multi-scale TTA
4. Apply threshold to debris probability map
5. Post-processing: remove blobs smaller than `MIN_DEBRIS_PIXELS`
6. Compute confusion matrix and per-class metrics
7. Compute **patch-level object detection metrics**
8. Sweep thresholds to find optimal debris F1
9. Compute **AUPRC** (Area Under Precision-Recall Curve)
10. Save **PR-curve PNG** and confusion matrix PNG
11. Save predicted GeoTIFF masks

### Test-time augmentation

**Standard TTA** (8 variants): 4 rotations × 2 flip states  
**Multi-scale TTA** (24 variants): scales {0.75, 1.0, 1.25} × 8 geometric

### Metrics reported

| Metric | Type | Notes |
|---|---|---|
| IoU, F1, Precision, Recall, Dice | Pixel-level | Per class |
| Overall pixel accuracy | Pixel-level | Note: inflated by class imbalance |
| AUPRC | Pixel-level | Standard for severe imbalance tasks |
| Optimal threshold | Sweep | 0.15–0.80 grid search |
| Patch recall | Object-level | Debris patches detected / total debris patches |
| Patch precision | Object-level | Debris patches detected / all patches flagged |
| Patch F1 | Object-level | Harmonic mean of patch recall and precision |

---

## How To Interpret Metrics Correctly

### Accuracy is not a valid headline metric for this project

With ~0.45% debris pixels, a model predicting not-debris everywhere achieves ~99.5% pixel accuracy. **Do not report accuracy as the primary result.**

### Priority metric order for the paper

1. Debris F1 at optimal threshold
2. AUPRC
3. Patch-level detection rate (object recall)
4. Debris recall and precision separately
5. Debris IoU

### Practical interpretation

| Pattern | Diagnosis |
|---|---|
| High recall + very low precision | Threshold too low, or too much imbalance pressure |
| High precision + very low recall | Threshold too high, or model under-fitting debris |
| High accuracy + low F1 | Majority-class collapse — ignore accuracy |
| High AUPRC | Model has good debris score separation overall |

---

## Post-Processing Pipeline

Implemented in `src/postprocess.py`.

### Operations

1. Optional DenseCRF boundary refinement
2. Hole removal
3. Small object removal (`MIN_DEBRIS_PIXELS = 3`)
4. Binary dilation
5. Connected-region polygonization
6. Per-patch GeoJSON export
7. Merged GeoJSON export
8. Centroid and bounding-box CSV export

Predicted masks are turned into GIS-ready objects for use in drift initialization and map overlays.

---

## Random Forest Baseline

Implemented in `src/random_forest/train_eval.py`.

### Feature modes

| Mode | File | Features |
|---|---|---|
| bands | `dataset.h5` | 11 raw spectral bands per pixel |
| indices | `dataset_si.h5` | 5 spectral indices per pixel |
| texture | `dataset_glcm.h5` | GLCM texture descriptors |

### Classifier

- 300 trees, balanced class weights, confidence-weighted fit
- OOB score reported for unbiased estimate
- Same train/val/test splits as the deep model

### Why keep the baseline

The RF path enables ablation: it quantifies how much the spatial reasoning of the deep model contributes over pure per-pixel spectral classification. This is expected to be a strong differentiator in the paper.

---

## Drift Prediction Pipeline

Implemented in `src/drift_prediction/drift.py`.

### Physics

Detected debris centroids are used as initial states for a particle-based Lagrangian simulation.

$$\dot{\mathbf{x}} = \mathbf{u}_{\text{ocean}} + \alpha_{\text{wind}}\,\mathbf{u}_{\text{wind}} + \alpha_{\text{stokes}}\,\mathbf{u}_{\text{wind}}$$

Integrated with RK4. Parameters:

| Parameter | Value |
|---|---|
| Horizon | 72 hours |
| Time step | 900 s (15 min) |
| Ensemble size | 200 particles |
| Wind leeway coefficient | 0.035 |
| Stokes drift coefficient | 0.016 |
| Output snap times | T+6h, T+12h, T+24h, T+48h, T+72h |

### Uncertainty quantification

Each ensemble output includes a **95% confidence ellipse** computed from the eigendecomposition of the particle position covariance matrix. Scaled by the chi-squared factor 2.4477 (2 DoF, 95th percentile).

### Inputs

| Source | Use |
|---|---|
| CMEMS NetCDF | Ocean current u/v fields |
| ERA5 NetCDF | 10m wind u10/v10 fields |
| Synthetic field | Test mode when no NetCDF provided |

Both NetCDF files are optional. If absent, a synthetic 0.1 m/s background current is used for testing the integration machinery.

---

## GPU vs CPU Operation

`config.py` auto-detects the device at import time and scales all critical settings.

| Setting | CPU | GPU |
|---|---|---|
| Backbone | resnext50_32x4d | resnext101_32x8d |
| Batch size | 8 | 32 |
| Workers | 4 | 8 |
| Grad accumulation | 4 | 2 |
| Effective batch | 64 | 64 |
| AMP | off | on |
| Expected epoch time | ~27–30 min | ~2–3 min |

**No code changes needed when switching devices.** All settings adjust automatically.

To override backbone manually:
```powershell
python train.py --backbone resnext50_32x4d
```

---

## Directory Structure

```text
Ocean_debris_detection/
├── README.md
├── Dataset/
│   ├── dataset.h5
│   ├── dataset_si.h5
│   ├── dataset_glcm.h5
│   ├── labels_mapping.txt
│   ├── patches/
│   └── splits/
├── checkpoints/
│   ├── resnext_cbam_best.pth       ← best raw model (any run)
│   ├── resnext_cbam_ema_best.pth   ← best EMA model (USE THIS)
│   └── run_<timestamp>/
│       ├── best.pth
│       ├── ema_best.pth
│       ├── last.pth
│       ├── epoch_NNN.pth
│       ├── metrics.csv
│       └── *.png (training curves)
├── logs/
│   ├── external_logs/
│   └── tsboard/
├── outputs/
│   ├── predicted_test/
│   ├── confusion_matrix_test.png
│   ├── pr_curve_test.png           ← Precision-Recall curve with AUPRC
│   ├── geospatial/
│   └── drift/
├── archive/
└── src/
    ├── train.py
    ├── evaluate.py
    ├── postprocess.py
    ├── run_pipeline.py
    ├── run.ps1
    ├── requirements.txt
    ├── configs/
    │   └── config.py               ← GPU/CPU auto-scaled
    ├── utils/
    │   ├── dataset.py
    │   ├── spectral_extraction.py
    │   └── visualize_predictions.py
    ├── semantic_segmentation/
    │   └── models/
    │       └── resnext_cbam_unet.py
    ├── random_forest/
    │   └── train_eval.py
    └── drift_prediction/
        └── drift.py
```

---

## Installation

```powershell
conda create -n debris python=3.11 -y
conda activate debris
cd D:\Bunny\Ocean_debris_detection\src
pip install -r requirements.txt
```

For GPU, install the matching CUDA PyTorch build before the requirements file.

---

## Recommended Commands

### Train the model (auto-detects GPU/CPU)

```powershell
cd D:\Bunny\Ocean_debris_detection\src

# GPU (recommended): auto-selects resnext101, batch=32
python train.py --epochs 200 --patience 50

# CPU: auto-selects resnext50, batch=8, grad_accum=4
python train.py --epochs 200 --batch 8 --workers 4 --grad_accum 4 --patience 50
```

### Evaluate (use EMA checkpoint for paper results)

```powershell
# Standard TTA evaluation with EMA checkpoint
python evaluate.py --split test \
    --ckpt ..\checkpoints\resnext_cbam_ema_best.pth \
    --tta

# Ensemble evaluation (best + ema_best + last averaged)
python evaluate.py --split test \
    --ckpt ..\checkpoints\resnext_cbam_ema_best.pth \
    --tta --ensemble
```

### Post-process predictions to GeoJSON

```powershell
python postprocess.py \
    --pred_dir ..\outputs\predicted_test \
    --out_dir  ..\outputs\geospatial
```

### Run drift prediction

```powershell
# With synthetic field (testing)
python drift_prediction/drift.py \
    --geojson ..\outputs\geospatial\all_debris.geojson

# With real data (CMEMS + ERA5)
python drift_prediction/drift.py \
    --geojson ..\outputs\geospatial\all_debris.geojson \
    --ocean_nc path\to\cmems_currents.nc \
    --wind_nc  path\to\era5_winds.nc
```

### Random Forest baseline

```powershell
# Spectral features only
python random_forest/train_eval.py

# All features (bands + indices + texture)
python random_forest/train_eval.py --use_si --use_glcm
```

---

## Pipeline Orchestration

### `run_pipeline.py` (recommended)

```powershell
# Full pipeline: train → evaluate (EMA + TTA) → postprocess → drift
python run_pipeline.py --epochs 200 --patience 50

# Include Random Forest baseline
python run_pipeline.py --epochs 200 --with_rf --with_si

# Skip training, evaluate existing checkpoint
python run_pipeline.py --skip_train
```

The pipeline automatically selects the EMA checkpoint and applies TTA for evaluation.

### `run.ps1`

Legacy PowerShell wrapper. Prefer `run_pipeline.py` for current functionality.

---

## Output Inventory

### Training outputs (per run)

- `best.pth`, `ema_best.pth`, `last.pth`, `epoch_NNN.pth`
- `metrics.csv` — epoch, train_loss, val_loss, mIoU, iou_debris, iou_not_debris, precision, recall, f1, **ema_f1**, **lr**
- PNG training curves for every metric, including `f1_raw_vs_ema.png`
- TensorBoard events

### Evaluation outputs

- Predicted GeoTIFF masks (`predicted_test/`)
- `confusion_matrix_test.png`
- `pr_curve_test.png` — **PR curve with AUPRC and optimal threshold marker**
- `eval_test.log` — full metrics including AUPRC, patch-level detection rate, threshold sweep

### Geospatial outputs

- Per-patch debris GeoJSON files
- `all_debris.geojson` — merged
- Centroid and bounding-box CSV

### Drift outputs

- Per-object `*_drift.geojson` (T+6h, 12h, 24h, 48h, 72h positions + ellipses)
- `all_drift.geojson` — merged

---

## Configuration Reference

All settings live in `src/configs/config.py`. GPU presence is auto-detected at import time.

| Key | CPU value | GPU value | Role |
|---|---|---|---|
| `ENCODER_NAME` | resnext50_32x4d | resnext101_32x8d | Backbone (auto) |
| `BATCH_SIZE` | 8 | 32 | Real batch per step (auto) |
| `GRAD_ACCUM` | 4 | 2 | Effective batch = 64 both ways (auto) |
| `NUM_WORKERS` | 4 | 8 | DataLoader workers (auto) |
| `NUM_CLASSES` | 2 | 2 | Binary segmentation |
| `INPUT_BANDS` | 16 | 16 | 11 raw + 5 indices |
| `CLASS_WEIGHTS` | [5.0, 1.0] | [5.0, 1.0] | Debris upweighted 5× |
| `FOCAL_GAMMA` | 3.0 | 3.0 | Focal modulation |
| `TVERSKY_ALPHA` | 0.3 | 0.3 | FP weight |
| `TVERSKY_BETA` | 0.7 | 0.7 | FN weight (recall-first) |
| `TVERSKY_WEIGHT` | 0.7 | 0.7 | Tversky fraction of loss |
| `AUX_LOSS_WEIGHT` | 0.4 | 0.4 | Deep supervision weight |
| `LABEL_SMOOTH` | 0.01 | 0.01 | Mild smoothing |
| `EMA_DECAY` | 0.9998 | 0.9998 | EMA smoothing factor |
| `ENCODER_FREEZE_EPOCHS` | 5 | 5 | Warmup freeze epochs |
| `COPY_PASTE_PROB` | 0.4 | 0.4 | Synthetic debris rate |
| `MIXUP_PROB` | 0.15 | 0.15 | MixUp augmentation rate |
| `OVERSAMPLE_HEAVY` | 8 | 8 | Heavy debris patch repeat |
| `OVERSAMPLE_LIGHT` | 4 | 4 | Light debris patch repeat |
| `MIN_DEBRIS_PIXELS` | 3 | 3 | Post-processing min size |

---

## Key Code Logic By File

### `src/utils/dataset.py`

- Builds a TIFF index once; avoids repeated recursive globbing
- Loads image, mask, confidence map
- Maps MARIDA DN→binary
- Computes percentile normalization from training split
- Computes and concatenates 5 spectral indices
- Oversamples debris-containing patches (heavy/light)
- Copy-paste debris augmentation (1–5 donors, flip/rotate/shift)
- MixUp augmentation (image blending, dominant mask preserved)
- Albumentations geometric and radiometric pipeline
- Confidence weight mapping (0→0.2, 1→0.7, 2→1.0)

### `src/semantic_segmentation/models/resnext_cbam_unet.py`

- CBAM: channel attention + spatial attention
- DecoderBlock: ConvTranspose2d + double conv + Dropout2d + CBAM
- ResNeXtCBAMUNet: dynamic backbone (50 or 101), 4-stage U-Net decoder, deep supervision aux head
- HybridLoss: Focal-CE + log-Tversky, handles (main, aux) tuple, confidence weighting

### `src/train.py`

- ModelEMA class (shadow copy, apply/restore for validation)
- Encoder freeze for first N epochs; optimizer rebuilt on unfreeze
- Training epoch: AMP forward, aux loss routing, grad accumulation, clip, EMA update
- Validation epoch: raw model then EMA model separately
- SGDR scheduler with linear warmup
- CSV + TensorBoard logging of all metrics including EMA F1 and LR
- `best.pth` and `ema_best.pth` saved independently

### `src/evaluate.py`

- Loads checkpoint and auto-detects backbone
- Ensemble mode: averages softmax across multiple checkpoints
- TTA (8×) and multi-scale TTA (24×)
- Post-processing: small blob removal
- Per-class pixel metrics (IoU, F1, precision, recall, dice)
- Object-level patch detection metrics (patch recall, precision, F1)
- Threshold sweep 0.15–0.80
- AUPRC via trapezoidal integration over 200-point PR curve
- PR curve PNG saved with AUPRC annotation and baseline

### `src/drift_prediction/drift.py`

- VelocityField: bilinear interpolation over lat/lon grid
- RK4 integrator with ocean current + wind leeway + Stokes drift
- Ensemble: 200 perturbed fields with velocity-proportional noise
- Confidence ellipse from eigendecomposition of position covariance (95%, χ² factor 2.4477)
- GeoJSON output: origin + trajectory points + ellipse polygons per time step

---

## Reproducibility Checklist

For a reproducible paper experiment, record:

1. Exact checkpoint path and run timestamp
2. Backbone used (resnext50 or resnext101)
3. GPU or CPU, PyTorch version, Python version
4. Config values at launch time
5. Train/val/test split files (unchanged across runs)
6. Whether TTA and which type was used
7. Threshold used for headline pixel metrics
8. Whether post-processing was applied before object export
9. EMA checkpoint or raw checkpoint used for evaluation

Minimum evidence to preserve for each serious experiment:
- `metrics.csv`
- Training log (`external_logs/train_<timestamp>.log`)
- Evaluation log (`eval_test.log`)
- `ema_best.pth`
- `pr_curve_test.png`
- `confusion_matrix_test.png`

---

## Experiment Playbook

### Ablation table for the paper

| Variant | Backbone | Debris F1 | AUPRC | Patch Recall |
|---|---|---|---|---|
| RF (bands only) | — | | | |
| RF (bands + SI + GLCM) | — | | | |
| ResNeXt-50, 11ch (no SI) | resnext50 | | | |
| ResNeXt-50, 16ch | resnext50 | | | |
| + Tversky loss | resnext50 | | | |
| + EMA + SGDR | resnext50 | | | |
| + TTA | resnext50 | | | |
| ResNeXt-101, 16ch + all | resnext101 | | | |
| + MS-TTA | resnext101 | | | |

### Practical selection rule

Use validation debris F1 to choose checkpoints. Report test-set behavior at:
- Default threshold 0.5
- Threshold-swept optimum

Report both raw checkpoint and EMA checkpoint results.

---

## Known Results and Training History

### Previous runs (CPU, resnext50)

| Run | Epochs | Best val F1 | Test F1 (thr=0.80) | Test AUPRC | Notes |
|---|---|---|---|---|---|
| run_20260224 | ~8 | NaN (loss collapse) | — | — | Old Lovász+OHEM+EMA config, unstable |
| run_20260227 | ~8 | ~0.60 recall, low F1 | — | — | Precision collapse, high recall |
| run_20260303 | ~12 | ~0.03 | 0.007 | — | Architecture regression |
| run_20260312 | short | ~0.007 | 0.007 | — | DeepLabV3+ attempt (smp), poor |
| run_20260313 | 12 | **0.393** (ep 8) | 0.349 | — | ResNeXtCBAMUNet restored |
| run_20260321 | 3 (smoke) | 0.012 | — | — | **New pipeline validated** (EMA, Tversky, SGDR) |

### Notes on run_20260313

- Best raw F1: 0.393 at epoch 8
- Test F1 at thr=0.80: 0.349 (P=0.270, R=0.493)
- Test recall: 0.803 at default threshold
- Training instability after epoch 8 (F1 oscillated ±0.17)

The new SGDR restarts and Dropout2d directly address the post-epoch-8 instability.

---

## Limitations

- Strong class imbalance (0.45%) makes precision recovery the main challenge
- CPU training makes ablation cycles slow (~28 min/epoch)
- Threshold calibration is global, not scene-adaptive
- Pixel metrics do not fully capture operational performance (object-level needed too)
- Drift quality depends on external CMEMS/ERA5 data availability
- EMA adds ~500MB checkpoint disk usage per run

---

## How To Position The Novelty In A Paper

The strongest framing is not "we trained a segmentation model." That is too weak.

**Recommended novelty statement:**

> "We propose an end-to-end marine debris monitoring workflow integrating multispectral rare-object segmentation with confidence-aware supervision, recall-first Tversky loss, EMA-stabilised training, threshold-calibrated inference with AUPRC reporting, geospatial object extraction, and downstream Lagrangian drift forecasting — evaluated with both pixel-level and object-level metrics on the MARIDA benchmark."

### Key contributions to defend

1. Binary reformulation of MARIDA for operational debris search
2. Multi-source input (spectral bands + hand-crafted indices)
3. Confidence-aware + recall-first loss (Tversky + focal)
4. EMA + SGDR for training stability under severe class imbalance
5. Threshold-calibrated evaluation with AUPRC and patch-level detection rate
6. End-to-end: detection → GIS objects → drift forecasting

---

## Troubleshooting

### Very high accuracy but useless debris detection
**Cause**: majority-class collapse.  
**Fix**: Check debris recall. Adjust threshold down. Check class weights.

### Recall spikes but precision near zero
**Cause**: too much imbalance pressure.  
**Check**: class weights, copy-paste probability, Tversky α/β balance.

### Model predicts almost no debris
**Cause**: underweighted positive class, threshold too high, or wrong checkpoint.  
**Check**: threshold sweep output; use EMA checkpoint; verify CLASS_WEIGHTS.

### Loss is NaN
**Cause**: usually inf/nan in the input data (edge bands with missing values).  
**Fix**: `np.nan_to_num` is already applied in dataset.py; check that patches load correctly.

### Training stops at epoch N well before patience
**Cause**: GPU OOM, or Python process killed.  
**Fix**: Reduce `--batch`; use `--resume` to continue from last checkpoint.

---

## Roadmap

1. **Finish GPU training run** — 200 epochs on resnext101 with all improvements
2. **Calibration analysis** — reliability curves, per-scene threshold adaptation
3. **Architecture ablations** — complete the ablation table above
4. **Object-level evaluation improvement** — IoU-based object matching (not just patch-level binary)
5. **Drift validation** — use real CMEMS + ERA5 data, compare predicted vs observed positions
6. **Paper write-up** — methods, experiments, ablation table, PR curve figure, drift map figure
