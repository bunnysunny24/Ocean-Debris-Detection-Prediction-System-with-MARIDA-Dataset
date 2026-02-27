---

## Full Script & Function Details

### Data Handling & Preprocessing
- **utils/dataset.py:**
    - Implements `MARIDADataset` (PyTorch Dataset) for loading patches, masks, and confidence maps.
    - Robust percentile normalization per band (2nd/98th percentiles).
    - Geometric augmentations: random horizontal/vertical flips, rotations, elastic transforms, spectral noise, brightness jitter, advanced Albumentations pipeline.
    - Oversampling debris-rich and hard negative patches for balanced training.
    - Converts images/masks to tensors, applies augmentation only for training.
    - Handles confidence-weighted loss masking.
    - Helper functions: `_find_tif`, `_find_mask_tif`, `_find_conf_tif`, `normalise`, augmentation helpers.

### Feature Extraction
- **utils/spectral_extraction.py:**
    - Extracts per-pixel spectral bands, indices (NDVI, NDWI, FDI, PI, RNDVI), and GLCM texture features.
    - Saves features to HDF5 for Random Forest training.
    - Functions: `compute_spectral_indices`, `compute_glcm_features`, `extract_split` (handles modes: bands, indices, texture).

### Model Architecture & Training
- **semantic_segmentation/models/resnext_cbam_unet.py:**
    - Defines `ResNeXtCBAMUNet` (ResNeXt-50 encoder, CBAM attention, U-Net decoder).
    - CBAM: Channel and spatial attention modules focus on relevant features.
    - Decoder blocks upsample and merge skip connections.
    - `HybridLoss`: Combines weighted cross-entropy and Dice loss, supports confidence maps and focal loss.

- **train.py:**
    - Main training loop: loads dataset, builds model, optimizer, loss, and scaler.
    - Functions: `train_epoch` (forward, backward, debug prints for tensor shapes, mask counts, learning rate, memory usage), `val_epoch` (validation, metrics, debug prints).
    - Computes IoU, saves checkpoints, logs metrics, handles NaN loss and batch timing.

### Evaluation & Postprocessing
- **evaluate.py:**
    - Evaluates model on test/val set, computes per-class IoU, F1, precision, recall, Dice.
    - Saves confusion matrix as PNG, predicted masks as GeoTIFF.
    - Functions: `plot_confusion_matrix`, `save_predicted_mask`, `evaluate` (main eval loop).

- **postprocess.py:**
    - Refines predicted masks: morphological cleanup, connected-component labeling, polygon vectorization.
    - Exports debris polygons as GeoJSON, centroids/bboxes as CSV.
    - Functions: `crf_refine` (optional DenseCRF), `refine_mask`, `vectorise_patch`, `process_all` (batch processing).

### Random Forest Training
- **random_forest/train_eval.py:**
    - Loads spectral features, indices, and textures from HDF5.
    - Trains Random Forest classifier, evaluates on val/test splits.
    - Functions: `load_split`, `merge_features`, `main` (training, evaluation, confusion matrix, balanced accuracy).

### Drift Prediction
- **drift_prediction/drift.py:**
    - Physics-based Lagrangian drift prediction using debris centroids, ocean/wind fields (NetCDF).
    - RK4 integrator, ensemble simulation for uncertainty, confidence ellipses.
    - Functions: `VelocityField`, `_load_nc_field`, `_synthetic_field`, `_rk4_step`, `run_ensemble`, `_confidence_ellipse`, `to_geojson` (GeoJSON output).

### Visualization
- **utils/visualise.py:**
    - Visualizes predicted masks, overlays RGB and ground truth, plots drift trajectories.
    - Functions: `visualise_masks`, `visualise_drift` (group by debris ID, plot ensemble trajectories).

### Pipeline Orchestration
- **run_pipeline.py:**
    - Master script: runs spectral extraction, model training, evaluation, postprocessing, drift prediction.
    - Handles command-line options for Random Forest, indices, GLCM, ocean/wind data, skipping steps.
    - Functions: `run` (subprocess wrapper), `main` (stepwise orchestration).

---

## Function Arguments & Data Flow
- **Dataset:** Loads patch images (11 bands), masks (15 classes), confidence maps. Normalizes, augments, converts to tensors.
- **Feature Extraction:** Takes normalized patch arrays, computes indices/textures, outputs DataFrames for RF.
- **Model Input:** Batch tensors [batch, channels, height, width], masks [batch, height, width], confidence [batch, height, width].
- **Training:** Forward pass → logits, loss (weighted), backward pass, optimizer step, gradient clipping, debug prints.
- **Evaluation:** Predicts masks, computes metrics, saves outputs.
- **Postprocessing:** Refines masks, vectorizes debris, exports GeoJSON/CSV.
- **Drift:** Takes debris centroids, simulates trajectories using ocean/wind fields, outputs ensemble results and confidence ellipses.
- **Visualization:** Plots masks, overlays, drift trajectories for presentation.

---

This section now covers every script, function, argument, input/output, and workflow in the project, suitable for deep technical documentation and presentation.
# Marine Debris Detection & Drift Prediction Pipeline
# =====================================================
# End-to-end pipeline on top of MARIDA + Sentinel-2

## Project Layout

```
marine_debris/
├── configs/
│   └── config.py               ← All paths & hyperparameters (EDIT THIS FIRST)
├── utils/
│   ├── dataset.py              ← PyTorch Dataset, normalisation, augmentation
│   ├── spectral_extraction.py  ← Extract signatures to HDF5 (for RF)
│   └── visualise.py            ← Plot masks & drift trajectories
├── semantic_segmentation/
│   └── models/
│       └── resnext_cbam_unet.py ← ResNeXt-50 + CBAM encoder, UNet decoder, HybridLoss
├── random_forest/
│   └── train_eval.py           ← RF training & evaluation
├── drift_prediction/
│   └── drift.py                ← RK4 ensemble Lagrangian drift, GeoJSON export
├── train.py                    ← Train segmentation model
├── evaluate.py                 ← Evaluate + save predicted masks
├── postprocess.py              ← Morphological cleanup, vectorise, GeoJSON
├── run_pipeline.py             ← Master runner for the full pipeline
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
# GPU (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Download & place MARIDA data
```
data/
├── patches/   ← extracted MARIDA patches
│   └── S2_<DATE>_<TILE>/
│       ├── S2_..._CROP.tif
│       ├── S2_..._CROP_cl.tif
│       └── S2_..._CROP_conf.tif
└── splits/
    ├── train_X.txt
    ├── val_X.txt
    └── test_X.txt
```

### 3. Edit paths in `configs/config.py` if needed

### 4. Run the full pipeline
```bash
# Deep learning only:
python run_pipeline.py

# With Random Forest too:
python run_pipeline.py --with_rf --with_si

# With real ocean/wind data for drift:
python run_pipeline.py --ocean_nc data/cmems.nc --wind_nc data/era5.nc

# Resume interrupted training:
python run_pipeline.py --resume
```

### 5. Run individual steps
```bash
# Train only:
python train.py --epochs 100 --batch 8

# Evaluate only (needs checkpoint):
python evaluate.py --split test

# Post-process only:
python postprocess.py --pred_dir outputs/predicted_test

# Drift only:
python drift_prediction/drift.py --geojson outputs/geospatial/all_debris.geojson

# Visualise:
python utils/visualise.py --pred_dir outputs/predicted_test --n 20
python utils/visualise.py --drift_geojson outputs/drift/all_drift.geojson
```

## Outputs
| Path | Contents |
|------|----------|
| `checkpoints/resnext_cbam_best.pth` | Best model checkpoint |
| `outputs/predicted_test/` | Predicted masks (GeoTIFF) |
| `outputs/geospatial/all_debris.geojson` | Vectorised debris polygons |
| `outputs/geospatial/debris_locations.csv` | Centroid coordinates + bbox |
| `outputs/drift/all_drift.geojson` | 24/48/72h trajectories + 95% ellipses |
| `outputs/drift_plots/` | Trajectory visualisation PNGs |
| `outputs/confusion_matrix_test.png` | Confusion matrix |
| `logs/train.log` | Training log |

## Notes on Drift Data


# Comprehensive Project Explanation

## Project Purpose
This project automates the detection of marine debris in satellite images and predicts its drift in the ocean. It combines deep learning (semantic segmentation) and classical machine learning (Random Forest) with robust feature extraction, evaluation, GIS-ready postprocessing, and physics-based drift modeling.

## Dataset
- **MARIDA Dataset:** Sentinel-2 satellite patches, split into train, validation, and test sets.
- **Patch Structure:** Each patch includes a multispectral image (.tif), a mask (.tif), and a confidence map (.tif).
- **Splits:** Defined in `Dataset/splits` as `train_X.txt`, `val_X.txt`, `test_X.txt`.

## Configuration
- All paths, hyperparameters, augmentation, and drift settings are centralized in `configs/config.py`.
- Key parameters: batch size, epochs, learning rate, class weights, augmentation probabilities, drift prediction horizon.

## Workflow Overview

1. **Data Preparation**
    - Organize patches and splits.
    - Edit paths in `configs/config.py` if needed.

2. **Feature Extraction**
    - Run `utils/spectral_extraction.py` to extract spectral bands, indices, and GLCM textures for Random Forest.

3. **Model Training**
    - Deep learning: `train.py` trains a segmentation model (ResNeXt-50 + CBAM U-Net or DeepLabV3+).
    - Classical ML: `random_forest/train_eval.py` trains a Random Forest classifier.

4. **Evaluation**
    - `evaluate.py` computes metrics (IoU, F1, precision, recall), saves confusion matrix and predicted masks.

5. **Postprocessing**
    - `postprocess.py` refines predictions, vectorizes debris regions, exports GeoJSON and CSV.

6. **Drift Prediction**
    - `drift_prediction/drift.py` predicts debris movement using ocean/wind data or synthetic currents.

7. **Pipeline Automation**
    - `run_pipeline.py` orchestrates all steps, allowing full workflow execution with options for Random Forest, spectral indices, GLCM features, and real ocean/wind data.

## Script Details

- **train.py:** Trains the segmentation model, logs metrics, saves checkpoints, and plots graphs.
- **random_forest/train_eval.py:** Trains and evaluates Random Forest, merges features, saves model and confusion matrix.
- **utils/spectral_extraction.py:** Extracts features for Random Forest.
- **utils/dataset.py:** Handles patch loading, normalization, augmentation, confidence-weighted loss.
- **evaluate.py:** Evaluates model, computes metrics, saves outputs.
- **postprocess.py:** Refines predictions, vectorizes, exports GIS-ready files.
- **drift_prediction/drift.py:** Predicts debris drift, outputs trajectories and ellipses.
- **configs/config.py:** Central configuration for all settings.
- **run_pipeline.py:** Master script for full pipeline execution.

## Outputs
- Model checkpoints, predicted masks, vectorized debris polygons, drift trajectories, confusion matrix, logs.

## How to Run
- Install dependencies (see requirements.txt).
- Prepare data and edit config.
- Run pipeline: `python run_pipeline.py` (with options as needed).
- Check outputs and logs for results.

## Validation & Debugging
- Dataset loading is robust; normalization and augmentation are applied.
- Debug prints and logs help monitor training, data flow, and performance.
- For missing patches or errors, use the provided validation script.

---

This section is suitable for presentations and detailed project documentation. If you need a visual workflow or more details on any step, refer to the script explanations or ask for a diagram.
## Overview
This project detects marine debris in satellite images and predicts ocean drift using deep learning and classical machine learning. It is built on the MARIDA dataset and Sentinel-2 imagery.

## Recommended Configuration

---

## Model Architecture (Deep Learning)
### ResNeXt-50 + CBAM U-Net
- **ResNeXt-50 Encoder:** Uses grouped convolutions for efficient, hierarchical feature extraction from multispectral input.
- **CBAM (Convolutional Block Attention Module):** Adds spatial and channel attention, helping the model focus on relevant debris regions.
- **U-Net Decoder:** Upsamples features, merges encoder outputs via skip connections, reconstructs spatial details for pixel-wise segmentation.
- **Layer Flow:**
    - Input patch → ResNeXt-50 encoder (learns low/high-level features)
    - CBAM modules (enhance important features)
    - U-Net decoder (upsamples, combines encoder features)
    - Final layer: 1x1 convolution for class prediction (debris/background)
- **Alternative:** DeepLabV3+ (optional, for advanced segmentation)

## Model Architecture (Random Forest)
- Classical ML using scikit-learn, trained on extracted spectral, index, and texture features from patches.

## Data Augmentation
Augmentation increases robustness by simulating real-world variations:
- **Types Used:**
    - Random horizontal/vertical flips
    - Random rotations
    - Random crops/resizing
    - Color jitter (brightness, contrast, saturation)
    - Gaussian noise
    - Cutout (randomly masks regions)
    - Elastic transforms (distorts shapes)
- **Purpose:** Prevents overfitting, improves generalization, handles class imbalance.
- **Configurable:** Probabilities and types are set in `configs/config.py`.

## Input Handling & Preprocessing
- **Input:** Multispectral Sentinel-2 patch (typically 13 bands), mask, confidence map.
- **Dataset Class:** Loads patch, mask, confidence; applies normalization and augmentation.
- **Normalization:** Scales pixel values to [0, 1] or standardizes per band.
- **Augmentation:** Applied on-the-fly during training.
- **Tensor Conversion:** Data is converted to PyTorch tensors (shape: [batch, channels, height, width]).
- **Mask Handling:** Masks are binarized; confidence maps weight the loss.
- **Why:** Preprocessing ensures consistent input, reduces noise, and aligns with model expectations.

## Tensor Flow & Training Process
- **Forward Pass:**
    - Input tensor → encoder layers (extract features)
    - Features pass through CBAM (attention)
    - Decoder upsamples, merges skip connections
    - Output: predicted mask tensor
- **Loss Calculation:**
    - Dice loss, cross-entropy, or confidence-weighted loss (combines mask and confidence map)
- **Backward Pass:**
    - Gradients computed via autograd
    - Optimizer (Adam or SGD) updates weights
- **Layer-wise Learning:**
    - Early layers learn basic patterns (edges, textures)
    - Deeper layers learn complex structures (debris shapes, context)
    - CBAM focuses attention on relevant regions
    - Decoder reconstructs spatial details for accurate segmentation
- **Epochs:** Training iterates over dataset multiple times, improving predictions each epoch.
- **Validation:** Metrics computed on validation set to monitor overfitting and generalization.

## Why Preprocessing & Augmentation Matter
- **Preprocessing:** Ensures input consistency, reduces artifacts, aligns with model requirements.
- **Augmentation:** Simulates real-world variability, improves robustness, prevents overfitting.

## How Model Learns
- **Supervised Learning:** Model compares predicted mask to ground truth, adjusts weights to minimize loss.
- **Layer-wise Feature Extraction:** Each layer builds on previous, learning increasingly complex representations.
- **Attention Mechanisms:** CBAM modules help model focus on debris regions, improving accuracy.
- **Skip Connections:** U-Net structure preserves spatial information, enabling precise segmentation.

---
- **Epochs:** 80 (tune as needed)
- **Learning Rate:** 5e-5 (default)
- **Optimizer:** Adam
- **Loss:** HybridLoss (Dice + Cross-Entropy)
- **Augmentation:** Geometric + spectral (enabled for training)
- **Early Stopping:** Patience = 10 epochs

## Step-by-Step Workflow

### 1. Data Preparation
- Place Sentinel-2 patches in `Dataset/patches`.
- Ensure splits (`train_X.txt`, `val_X.txt`, `test_X.txt`) are in `Dataset/splits`.

### 2. Feature Extraction (for Random Forest)
```bash
python utils/spectral_extraction.py               # raw bands → data/dataset.h5
python utils/spectral_extraction.py --type indices # spectral indices → data/dataset_si.h5
python utils/spectral_extraction.py --type texture # GLCM texture → data/dataset_glcm.h5
```

### 3. Train Random Forest Classifier
```bash
python random_forest/train_eval.py
python random_forest/train_eval.py --use_si --use_glcm
```

### 4. Train Deep Learning Model
```bash
python train.py --epochs 80 --batch 4 --lr 5e-5
```

### 5. Evaluate Model
- Use `evaluate.py` to assess model performance.
- Use `postprocess.py` for output processing and visualization.

### 6. Drift Prediction
- Use `drift_prediction/drift.py` to predict ocean drift based on model outputs.

---

## Script Explanations

- **train.py**: Trains the ResNeXt-50 + CBAM U-Net for semantic segmentation. Loads data, applies augmentation, defines model/loss/optimizer, runs training/validation, logs metrics, saves checkpoints.
- **random_forest/train_eval.py**: Trains and evaluates a Random Forest classifier on extracted spectral features. Loads features, merges indices/textures, trains RF, evaluates, saves model and confusion matrix.
- **utils/spectral_extraction.py**: Extracts spectral features from patches and saves to HDF5. Supports raw bands, spectral indices, and GLCM texture features.
- **utils/dataset.py**: Defines `MARIDADataset` for PyTorch. Handles normalization, augmentation, and confidence-weighted loss.
- **semantic_segmentation/models/resnext_cbam_unet.py**: Defines the ResNeXt-50 + CBAM U-Net model and HybridLoss.
- **configs/config.py**: Central configuration for paths, hyperparameters, and class names.
- **evaluate.py**: Evaluates the trained model and saves predicted masks.
- **postprocess.py**: Cleans up predictions, vectorizes, and exports to GeoJSON.
- **drift_prediction/drift.py**: Predicts ocean drift using model outputs and ocean/wind data.
- **utils/visualise.py**: Visualizes masks and drift trajectories.

---

## Dataset Validation
The dataset pipeline ensures:
- Patch IDs are loaded from the correct split file.
- .tif images, masks, and confidence files are found for each patch.
- Normalization and augmentation are applied as configured.
- Data is converted to PyTorch tensors and sent to the model correctly.

To check for missing or invalid patches, you can run a script to iterate through the dataset and print out any issues. Example:
```python
from utils.dataset import MARIDADataset
ds = MARIDADataset("train")
for i in range(len(ds)):
    try:
        sample = ds[i]
    except Exception as e:
        print(f"Error at index {i}: {e}")
```

---

## End-to-End Summary
1. Prepare and organize your data.
2. Extract features for Random Forest.
3. Train Random Forest.
4. Train U-Net model.
5. Evaluate and visualize results.
6. Predict drift using model outputs.


---

## Advanced Technical Notes & Edge Cases

### Hyperparameter Tuning
- All hyperparameters (batch size, learning rate, epochs, augmentation probabilities, drift settings) are set in `configs/config.py`.
- Early stopping, patience, optimizer choice, and loss function weights are configurable for robust training.
- Class weights are used to handle imbalance between debris and background.

### Error Handling & Debugging
- Extensive debug print statements in `train.py` for tensor shapes, mask counts, learning rate, batch timing, RAM usage, and predicted class distribution.
- NaN loss detection and batch skipping to prevent training collapse.
- Warnings for missing debris pixels, hard negatives, and patch loading issues.
- Logging to timestamped log files for reproducibility and troubleshooting.

### Integration & Extensibility
- Modular design: each script can be run independently or via `run_pipeline.py`.
- Supports CPU and GPU training; device automatically detected.
- Random Forest and deep learning models can be combined or used separately.
- Feature extraction supports spectral bands, indices, and GLCM textures for classical ML.
- Drift prediction can use synthetic or real ocean/wind fields (NetCDF).
- Outputs are GIS-ready (GeoTIFF, GeoJSON, CSV) for downstream analysis.

### Edge Cases & Data Quality
- Robust normalization handles outliers and missing values.
- Oversampling ensures rare debris patches are well represented.
- Hard negative mining improves model discrimination.
- Confidence maps weight loss for uncertain pixels.
- Morphological postprocessing removes small blobs, fills holes, and refines masks.

### Visualization & Presentation
- Visual overlays of predicted masks, RGB, and ground truth for qualitative assessment.
- Drift trajectory plots show ensemble spread and confidence ellipses.
- Confusion matrix and per-class metrics saved for quantitative evaluation.

### Reproducibility & Best Practices
- All random seeds can be set for reproducible results.
- Outputs and logs are timestamped and organized by run.
- Modular scripts allow easy integration with new datasets or models.
- All dependencies listed in `requirements.txt` with conda/pip install instructions.

---

This README now contains every granular detail, technical note, edge case, and integration tip for the project, ensuring nothing is omitted for documentation or presentation.
For any issues, check logs in `logs/` and outputs in `outputs/`.
