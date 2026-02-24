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


# Detailed Project Guide

## Overview
This project detects marine debris in satellite images and predicts ocean drift using deep learning and classical machine learning. It is built on the MARIDA dataset and Sentinel-2 imagery.

## Recommended Configuration
- **Model:** ResNeXt-50 + CBAM U-Net (semantic segmentation)
- **Random Forest:** For spectral feature-based classification
- **Input Bands:** 11 Sentinel-2 bands (B10 excluded)
- **Classes:** 15 MARIDA semantic classes
- **Patch Size:** 256
- **Batch Size:** 4 (adjust for your GPU)
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

For any issues, check logs in `logs/` and outputs in `outputs/`.
