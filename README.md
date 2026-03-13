# Marine Debris Detection and Drift Prediction Pipeline

End-to-end research pipeline for detecting floating marine debris in Sentinel-2 imagery and projecting debris motion through a physics-based ocean drift model.

This repository combines:
- a deep-learning semantic segmentation model for binary debris detection
- a Random Forest baseline built from extracted spectral and texture features
- geospatial post-processing for polygon and centroid export
- ensemble Lagrangian drift prediction for short-horizon debris movement analysis

The project is built around the MARIDA benchmark and has been adapted into a binary rare-object detection system: debris vs not-debris.

## Table of Contents

1. Project Summary
2. Pipeline Overview
3. What Is In The Repository
4. Current Technical Status
5. Research Motivation and Novelty
6. Data Definition
7. Binary Label Mapping
8. Input Feature Design
9. Deep Learning Model
10. Why The Architecture Changed
11. Loss Function Design
12. Data Loading and Normalization
13. Imbalance Handling Strategy
14. Augmentation Strategy
15. Training Pipeline
16. Checkpointing and Model Selection
17. Evaluation Pipeline
18. How To Interpret Metrics Correctly
19. Post-Processing Pipeline
20. Random Forest Baseline
21. Drift Prediction Pipeline
22. Directory Structure
23. Installation
24. Recommended Commands
25. Pipeline Orchestration Scripts
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

## Project Summary

This project solves two linked research problems.

| Problem | Method | Output |
|---|---|---|
| Marine debris detection | Binary semantic segmentation and Random Forest baseline | debris masks, GeoTIFF predictions, GeoJSON polygons, centroid CSV |
| Debris movement forecasting | RK4 particle tracking with ensemble perturbations | 6h to 72h trajectories and uncertainty geometry |

The core challenge is extreme rare-class detection. In the binary mapping used here, debris occupies only about 0.45% of valid labeled pixels. That means ordinary pixel accuracy is not a useful target by itself. The project is therefore built around recall, F1, IoU, threshold calibration, and geospatial usability.

## Pipeline Overview

The repository is designed as a staged workflow rather than a single isolated model script.

```text
Sentinel-2 patches
   -> binary label mapping
   -> normalization + spectral indices
   -> segmentation training
   -> thresholded inference + TTA
   -> morphological cleanup
   -> polygon and centroid export
   -> drift initialization
   -> trajectory forecasting
```

Operationally, the system has three layers:
- learning layer: segmentation and Random Forest baselines
- geospatial layer: raster-to-vector conversion and object summarization
- forecasting layer: drift prediction from detected debris objects

This separation is important for research reporting because each layer can be evaluated independently or as part of the full pipeline.

## What Is In The Repository

- `src/train.py`: training loop for the segmentation model
- `src/evaluate.py`: evaluation, TTA, threshold sweep, confusion matrix, prediction export
- `src/postprocess.py`: morphological cleanup and GeoJSON / CSV export
- `src/semantic_segmentation/models/resnext_cbam_unet.py`: current segmentation architecture and loss
- `src/utils/dataset.py`: dataset loading, normalization, oversampling, copy-paste augmentation, confidence weighting
- `src/utils/spectral_extraction.py`: HDF5 feature extraction for classical ML baselines
- `src/drift_prediction/`: drift prediction logic
- `src/random_forest/`: Random Forest baseline components
- `checkpoints/`: saved runs and shared best-model copies
- `outputs/`: predicted masks, confusion matrices, geospatial exports, drift products
- `archive/`: utility and analysis scripts used during experimentation and paper preparation

## Current Technical Status

The repository currently uses a corrected training path that replaces older unstable configurations.

Current deep model path:
- Architecture: `ResNeXtCBAMUNet`
- Input channels: 16
- Classes: 2
- Loss: Focal Cross-Entropy + Dice
- Optimizer: AdamW
- Scheduler: Linear warmup then cosine annealing
- Best-model criterion: debris F1 with a minimum recall guard

Important recent fixes already implemented in code:
- removed the unstable Lovasz + boundary + OHEM loss stack
- removed EMA, SWA, and encoder freezing from the active training path
- switched the model path back to the custom CBAM U-Net family
- reduced imbalance pressure from overly aggressive class weighting and augmentation
- fixed Albumentations `Affine` warning caused by an invalid `mode` argument
- migrated torchvision backbone loading to the weights API
- slimmed the decoder width to reduce overfitting pressure and CPU cost

Current model size in code:
- approximately `30.6M` parameters for the current 16-band binary configuration

## Research Motivation and Novelty

The novelty of the project is not a single isolated algorithmic contribution. It is the full system design for rare marine debris detection under severe imbalance and geospatial deployment constraints.

The main research contributions implemented in code are:

1. Binary reformulation of MARIDA for rare debris-first detection.
   The original 15-class MARIDA problem is collapsed into a binary task where class 1 in the original labels becomes debris and all other semantic classes become not-debris. This reframes the problem toward operational debris search rather than broad land/water scene parsing.

2. Multi-source input representation.
   The model uses 11 Sentinel-2 raw bands plus 5 hand-engineered spectral indices: NDVI, NDWI, FDI, PI, and RNDVI. This mixes learned deep features with physically motivated spectral cues tailored to marine debris and water-background separation.

3. Confidence-weighted supervised learning.
   Pixel confidence maps provided by MARIDA are converted into loss weights. This allows uncertain labels to influence optimization less strongly than highly confident ones.

4. Rare-object focused data pipeline.
   The project combines patch oversampling, copy-paste debris augmentation, confidence-aware loss weighting, recall-guarded model selection, and threshold sweeping at evaluation. This is a system-level imbalance strategy rather than a single trick.

5. Attention-enhanced encoder-decoder design.
   The segmentation architecture uses a ResNeXt encoder with CBAM attention inside the decoder path. The attention mechanism is applied where spatial reassembly and rare-object localization matter most.

6. Operational geospatial output path.
   Predictions are not treated as endpoint masks only. The pipeline turns them into GIS-ready polygons, centroids, CSV tables, and then drift trajectories. This makes the project stronger as an end-to-end remote-sensing workflow.

7. Tight detection-to-drift coupling.
   The project treats detection as the upstream stage of a downstream ocean-motion model. This is useful for real maritime monitoring, not just benchmark reporting.

## Data Definition

The project uses MARIDA patch data derived from Sentinel-2.

### Dataset facts

| Property | Value |
|---|---|
| Satellite | Sentinel-2 L2A |
| Raw bands used | 11 |
| Extra indices | 5 |
| Total input channels | 16 |
| Patch size | 256 x 256 |
| Binary classes | Debris, Not Debris |
| Nodata label after mapping | -1 |

Current split sizes used in the repository:

| Split | Patches |
|---|---|
| Train | 694 |
| Validation | 328 |
| Test | 359 |

The dataset is highly imbalanced:
- only a minority of patches contain debris
- debris occupies a very small fraction of valid pixels
- naive accuracy can look excellent while the detector is operationally useless

## Binary Label Mapping

The repository maps the original MARIDA class labels into a binary target:

| Original mask DN | Binary value | Meaning |
|---|---|---|
| 0 | -1 | nodata / ignored |
| 1 | 0 | marine debris |
| 2-15 | 1 | not-debris |

This mapping is implemented in the dataset loader, not only documented externally. Any interpretation of metrics must therefore be done in binary form.

## Input Feature Design

### Raw spectral bands

The raw Sentinel-2 bands used are:
- B1
- B2
- B3
- B4
- B5
- B6
- B7
- B8
- B8A
- B11
- B12

B10 is excluded.

### Spectral indices

Five indices are computed before normalization of the raw bands.

| Index | Purpose |
|---|---|
| NDVI | vegetation suppression cue |
| NDWI | water contrast cue |
| FDI | floating debris sensitivity |
| PI | plastic-related ratio heuristic |
| RNDVI | red-green contrast cue |

In `dataset.py`, the raw bands are percentile-normalized and the indices are clipped and rescaled before concatenation into the final 16-channel tensor.

## Deep Learning Model

### Current primary architecture

The active architecture is a custom `ResNeXtCBAMUNet` defined in `src/semantic_segmentation/models/resnext_cbam_unet.py`.

High-level structure:
- ResNeXt-50 backbone encoder
- first convolution adapted from 3 channels to arbitrary input channels
- pretrained ImageNet initialization when enabled
- CBAM attention module on the deepest encoder feature
- four-stage U-Net style decoder
- final upsampling and `1x1` class projection layer

### Encoder details

The encoder is built from torchvision `resnext50_32x4d` and split into:
- `enc0`: stem adapted to 16-channel input
- `enc1`
- `enc2`
- `enc3`
- `enc4`

When pretrained weights are used and input channels are not 3, the first convolution is initialized by averaging RGB weights and repeating them across all input channels. This is a pragmatic transfer-learning bridge for multispectral inputs.

### Decoder details

The current decoder is intentionally slimmer than the previous experimental version.

Current decoder widths:
- `dec4: 2048 + 1024 -> 256`
- `dec3: 256 + 512 -> 128`
- `dec2: 128 + 256 -> 64`
- `dec1: 64 + 64 -> 64`

This was changed after observing that the wider 41M-parameter decoder learned but remained too aggressive and computationally expensive on CPU. The slimmer decoder is closer to the proven configuration and reduces false-positive pressure.

### CBAM attention

Each CBAM block combines:
- channel attention using adaptive average and max pooling
- spatial attention using pooled channel descriptors

The role of CBAM here is to help the network focus on subtle debris signatures that are easily overwhelmed by water, cloud edges, wakes, and clutter.

## Why The Architecture Changed

The project went through several training configurations. Some of them were technically valid but behaviorally wrong for this dataset.

### Older unstable path

The repository previously contained or documented a more complicated path involving:
- DeepLabV3+
- Lovasz-Softmax
- boundary-aware auxiliary loss
- online hard example mining
- EMA
- SWA
- encoder freezing
- aggressive class weighting
- aggressive oversampling and copy-paste augmentation

That configuration produced collapse or precision failure under the CPU training regime used here.

Observed failure mode:
- recall could spike early
- precision could collapse to near zero
- the model started predicting debris too broadly
- later training either collapsed to majority class or remained badly calibrated

### Current stable path

The active code now uses:
- custom CBAM U-Net
- simpler Focal-CE + Dice loss
- single AdamW optimizer
- warmup + cosine schedule
- moderated imbalance handling

This is a root-cause correction, not cosmetic tuning.

## Loss Function Design

The current `HybridLoss` is intentionally simple.

### Formula

For logits $z$, targets $y$, and softmax probabilities $p$:

$$
L = (1 - \lambda_{dice}) L_{focal-ce} + \lambda_{dice} L_{dice}
$$

where currently:

$$
\lambda_{dice} = 0.7
$$

### Focal Cross-Entropy term

The cross-entropy term is computed per pixel with:
- class weights from configuration
- optional label smoothing
- nodata masking
- optional confidence weighting

Then focal modulation is applied:

$$
L_{focal-ce} = (1 - p_t)^{\gamma} L_{ce}
$$

with current:
- `FOCAL_GAMMA = 2.0`
- `LABEL_SMOOTH = 0.02`

### Dice term

The Dice term is averaged across classes using valid pixels only. It compensates for class imbalance and emphasizes overlap quality.

### Why simpler is better here

The repository previously used a much heavier loss stack. That stack made optimization more brittle than the dataset could support. The current design keeps the two components that consistently help:
- cross-entropy for calibrated classification pressure
- Dice for rare-region overlap pressure

## Data Loading and Normalization

The data pipeline is implemented in `src/utils/dataset.py`.

### File lookup strategy

The dataset builds a cached index over all patch TIFFs, then resolves:
- image TIFF
- class TIFF
- confidence TIFF

This avoids repeated recursive globbing during training.

### Percentile normalization

Per-band normalization statistics are computed from the training split using the 2nd and 98th percentiles.

For each band $b$:

$$
x_b' = \text{clip}\left(\frac{x_b - P2_b}{P98_b - P2_b + \epsilon}, 0, 1\right)
$$

This is robust to outliers and much more stable than a global min-max normalization in multispectral remote sensing.

### Nodata handling

Nodata is mapped to `-1` and excluded from loss and metric computation. This is critical. If nodata participates in optimization, the model can learn artifacts rather than debris structure.

### Confidence weighting

Confidence values from `_conf.tif` are mapped as:

| Raw confidence | Weight |
|---|---|
| 0 | 0.2 |
| 1 | 0.7 |
| 2 | 1.0 |

This means uncertain labels still contribute, but much less strongly than trusted ones.

## Imbalance Handling Strategy

This repository handles imbalance at several levels.

### Current active settings

- class weights: `[3.0, 1.0]`
- copy-paste probability: `0.25`
- oversampling heavy: `8`
- oversampling light: `4`

### Why this was changed

Earlier settings used much more aggressive numbers. That improved recall temporarily but damaged precision badly. The current values are a compromise intended to keep rare-class sensitivity without overwhelming the classifier with synthetic positives.

### Patch oversampling logic

Training patches are duplicated depending on debris count:
- `>= 20` debris pixels: heavy oversampling
- `1-19` debris pixels: light oversampling
- `0` debris pixels: kept once

This increases exposure to positive regions while still preserving negative context.

## Augmentation Strategy

The augmentation path is tuned for sparse positive pixels.

### Active spatial and spectral augmentations

- horizontal flip
- vertical flip
- random 90-degree rotation
- affine shift / scale / rotate
- Gaussian noise
- elastic transform
- grid distortion
- brightness / contrast perturbation

### Copy-paste debris augmentation

The dataset can paste debris pixels from randomly selected donor patches into the current sample.

Implemented logic:
- choose 1 to 3 donor patches from a debris pool
- optionally flip and rotate donor debris positions
- shift pasted pixels spatially within the destination patch
- copy both image values and updated binary mask
- increase confidence weight on pasted debris to high-confidence level

This is one of the strongest project-specific augmentation ideas because the native debris signal is so sparse.

## Training Pipeline

The training pipeline is implemented in `src/train.py`.

### Core loop

For each epoch:
- iterate over training batches
- run forward pass under AMP when CUDA is available
- compute confidence-weighted loss
- apply gradient accumulation
- clip gradients at `max_norm = 5.0`
- step optimizer and scheduler
- run validation
- compute IoU, precision, recall, and F1 for the debris class
- write CSV metrics and TensorBoard scalars
- save epoch and best checkpoints

### Optimizer and schedule

- optimizer: AdamW
- warmup: linear
- main schedule: cosine annealing
- minimum learning rate: `1e-6`

This schedule is designed to allow early movement out of poor local solutions and later precision refinement.

### Gradient accumulation

The training loop supports effective larger batches through accumulation:

$$
\text{effective batch} = \text{batch size} \times \text{grad accum}
$$

This matters on CPU because memory and throughput are limited.

### Device behavior

The code runs on CPU or GPU. On the current machine, training has been CPU-bound. The pipeline therefore avoids depending on GPU-only tricks.

## Checkpointing and Model Selection

Every run creates:
- per-epoch checkpoints
- `last.pth`
- `best.pth`
- `metrics.csv`
- per-metric PNG curves

### Best-model criterion

The best model is selected by debris F1, with an additional minimum-recall guard:
- if recall is below `5%`, the candidate is not accepted as best

This avoids selecting a high-accuracy, zero-recall majority-class model.

### Shared checkpoint copy

When a new best model is found, it is copied to:
- `checkpoints/resnext_cbam_best.pth`

This simplifies evaluation and downstream scripts.

## Evaluation Pipeline

Evaluation is implemented in `src/evaluate.py`.

### Evaluation flow

1. Load checkpoint
2. Reconstruct the segmentation model
3. Load the requested split
4. Run prediction with optional TTA or multi-scale TTA
5. Convert probabilities into class labels using a threshold on debris probability
6. Compute confusion matrix and class metrics
7. Save predicted masks
8. Sweep thresholds to find the best debris F1 on the evaluated split

### Test-time augmentation

Two TTA modes exist:

1. Standard TTA:
   - 4 rotations x 2 flip states = 8 geometric variants

2. Multi-scale TTA:
   - scales `0.75`, `1.0`, `1.25`
   - each scale can also use the 8-variant geometric TTA

### Threshold sweep

The evaluation script does not assume that `0.5` is optimal. It explicitly sweeps thresholds from `0.15` to `0.80` and reports the best debris F1.

This is essential for a rare-object detector because score calibration is often poor early in training.

## How To Interpret Metrics Correctly

This project must not be judged by pixel accuracy alone.

### Why accuracy is misleading

If almost all valid pixels are not-debris, then a model can achieve excellent pixel accuracy by predicting not-debris almost everywhere.

### Metrics that matter most

Use these in priority order:
- debris recall
- debris precision
- debris F1
- debris IoU
- thresholded operational behavior on test images

### Practical interpretation

- high recall and very low precision: detector is too aggressive
- high precision and very low recall: detector is missing debris
- high accuracy alone: meaningless without class-wise metrics

## Post-Processing Pipeline

Post-processing is implemented in `src/postprocess.py`.

### Operations

- optional DenseCRF refinement when available
- hole removal
- small object removal
- binary dilation
- connected-region polygonization
- per-patch GeoJSON export
- merged GeoJSON export
- centroid and bounding-box CSV export

### Why this matters

Raw segmentation masks are not ideal for GIS or drift modeling. The post-processing stage turns pixel predictions into spatial objects that can be tracked and analyzed.

## Random Forest Baseline

The repository also supports a classical baseline.

### Feature extraction

Implemented in `src/utils/spectral_extraction.py`.

Available extraction modes:
- raw bands
- spectral indices
- texture proxies

Outputs are stored in:
- `Dataset/dataset.h5`
- `Dataset/dataset_si.h5`
- `Dataset/dataset_glcm.h5`

### Why keep a classical baseline

The Random Forest path is useful for:
- ablation and sanity checks
- demonstrating the value of spatial deep learning over pure per-pixel feature models
- supporting a stronger methodological comparison section in research writing

## Drift Prediction Pipeline

The downstream drift module forecasts debris movement after detection.

### Core idea

Detected debris polygons or centroids are used as initial states for a particle-based drift simulation. Motion is integrated with RK4 and can incorporate:
- ocean currents
- wind-driven leeway
- Stokes drift
- ensemble perturbations for uncertainty quantification

### Current drift configuration

From config:
- horizon: 72 hours
- time step: 900 seconds
- ensemble size: 200
- wind coefficient: 0.035
- Stokes coefficient: 0.016

### Why this matters for novelty

This coupling of remote-sensing detection with trajectory forecasting adds operational value. It moves the project beyond segmentation benchmarking into marine monitoring and response support.

## Directory Structure

```text
Ocean_debris_detection/
|-- README.md
|-- Dataset/
|   |-- dataset.h5
|   |-- dataset_si.h5
|   |-- dataset_glcm.h5
|   |-- labels_mapping.txt
|   |-- patches/
|   `-- splits/
|-- checkpoints/
|   |-- resnext_cbam_best.pth
|   |-- resnext_cbam_last.pth
|   `-- run_<timestamp>/
|-- logs/
|   |-- external_logs/
|   `-- tsboard/
|-- outputs/
|   |-- predicted_test/
|   |-- geospatial/
|   `-- drift/
|-- archive/
`-- src/
    |-- train.py
    |-- evaluate.py
    |-- postprocess.py
    |-- run_pipeline.py
    |-- run.ps1
    |-- requirements.txt
    |-- configs/
    |   `-- config.py
    |-- utils/
    |   |-- dataset.py
    |   |-- spectral_extraction.py
    |   `-- visualize_predictions.py
    |-- semantic_segmentation/
    |   `-- models/
    |       `-- resnext_cbam_unet.py
    |-- random_forest/
    `-- drift_prediction/
```

## Installation

### Recommended environment

```powershell
conda create -n debris python=3.11 -y
conda activate debris
cd src
pip install -r requirements.txt
```

For CPU-only environments, install the CPU build of PyTorch. For GPU environments, install the matching CUDA build before the requirements file if needed.

## Recommended Commands

### Train the current model

```powershell
cd D:\Bunny\Ocean_debris_detection\src
python train.py --epochs 100 --batch 8 --workers 4 --grad_accum 4 --patience 40
```

### Evaluate an existing checkpoint

```powershell
cd D:\Bunny\Ocean_debris_detection\src
python evaluate.py --split test --ckpt ..\checkpoints\resnext_cbam_best.pth --tta
```

### Generate GIS outputs from predicted masks

```powershell
cd D:\Bunny\Ocean_debris_detection\src
python postprocess.py --pred_dir ..\outputs\predicted_test --out_dir ..\outputs\geospatial
```

### Extract Random Forest features

```powershell
cd D:\Bunny\Ocean_debris_detection\src
python utils\spectral_extraction.py
python utils\spectral_extraction.py --type indices
python utils\spectral_extraction.py --type texture
```

## Pipeline Orchestration Scripts

The repository contains two orchestration entry points in addition to direct script execution.

### `src/run_pipeline.py`

This is the preferred Python orchestrator because it matches the current script interfaces more closely.

It can:
- optionally extract features for the Random Forest path
- train the segmentation model
- evaluate on the test split
- run post-processing
- run drift prediction
- optionally train the Random Forest baseline

Useful modes:
- full pipeline
- extraction only
- skip training
- resume training
- include RF, spectral indices, and texture features

### `src/run.ps1`

This PowerShell wrapper is useful for Windows-native execution, but it contains some legacy assumptions from earlier experimental phases.

Important note:
- prefer the direct commands in this README or `run_pipeline.py` for the current model path
- if `run.ps1` is used, validate its CLI arguments against the current version of `train.py` and `evaluate.py`

In other words: `run.ps1` is still valuable as a convenience wrapper, but `run_pipeline.py` and direct script calls are the more reliable documentation targets at the moment.

## Output Inventory

The project produces several categories of outputs.

### Training outputs

Per run, the training pipeline writes:
- `best.pth`
- `last.pth`
- `epoch_XXX.pth`
- `metrics.csv`
- per-metric PNG curves
- TensorBoard event files
- external log file with epoch-by-epoch metrics

### Evaluation outputs

Evaluation writes:
- predicted GeoTIFF masks
- confusion matrix figure
- threshold sweep summary in logs
- class metrics for debris and not-debris

### Geospatial outputs

Post-processing writes:
- per-patch debris GeoJSON files
- merged debris GeoJSON
- centroid and bounding-box CSV

### Forecast outputs

Drift prediction writes:
- trajectory GeoJSON files
- merged drift outputs
- uncertainty geometry if enabled in the drift logic

## Configuration Reference

The central settings live in `src/configs/config.py`.

### Current important settings

| Key | Current value | Role |
|---|---:|---|
| `NUM_CLASSES` | 2 | binary segmentation |
| `INPUT_BANDS` | 16 | 11 raw + 5 indices |
| `CLASS_WEIGHTS` | `[3.0, 1.0]` | moderate debris upweighting |
| `FOCAL_GAMMA` | `2.0` | focal modulation |
| `LABEL_SMOOTH` | `0.02` | mild smoothing |
| `COPY_PASTE_PROB` | `0.25` | synthetic positive augmentation |
| `OVERSAMPLE_HEAVY` | `8` | heavy positive patch duplication |
| `OVERSAMPLE_LIGHT` | `4` | light positive patch duplication |
| `MIN_DEBRIS_PIXELS` | `3` | post-processing minimum object size |

Some legacy config keys remain in the file for historical reasons, but they are no longer part of the active training path.

## Key Code Logic By File

### `src/utils/dataset.py`

- builds a TIFF index once
- loads image, mask, confidence map
- maps original MARIDA labels to binary targets
- computes percentile normalization statistics from the training split
- computes and concatenates spectral indices
- oversamples debris-containing patches
- optionally performs copy-paste debris synthesis
- applies geometric and radiometric augmentation
- returns image, binary mask, confidence weights, and patch id

### `src/semantic_segmentation/models/resnext_cbam_unet.py`

- defines channel and spatial attention blocks
- defines U-Net style decoder blocks
- adapts ResNeXt-50 to multispectral input
- applies deep-feature CBAM attention
- defines the current Focal-CE + Dice loss

### `src/train.py`

- constructs train and validation datasets
- trains with AMP when available
- uses gradient accumulation and clipping
- computes class-wise validation metrics
- selects best model by debris F1 with recall guard
- saves TensorBoard metrics and CSV history

### `src/evaluate.py`

- loads model checkpoints
- runs optional TTA and multi-scale TTA
- computes confusion matrix and class metrics
- sweeps thresholds for better debris F1
- saves predicted GeoTIFF masks and confusion matrix image

### `src/postprocess.py`

- optionally refines predictions with DenseCRF
- removes holes and tiny objects
- vectorizes debris regions
- exports GeoJSON and centroid CSV tables

### `src/utils/spectral_extraction.py`

- extracts tabular per-pixel features for Random Forest experiments
- supports bands, indices, and texture modes
- stores per-split data in HDF5 tables

## Reproducibility Checklist

For a reproducible experiment or paper run, keep the following fixed and recorded:

1. exact checkpoint directory and timestamped run id
2. config values used at launch time
3. train, validation, and test split files
4. whether TTA or multi-scale TTA was used
5. threshold used for headline metrics
6. whether post-processing was applied before object export
7. hardware context: CPU or GPU, Python version, PyTorch version
8. whether the current slim decoder or an older wider decoder was used

Minimum evidence to preserve for each serious experiment:
- `metrics.csv`
- training log
- test evaluation log
- final best checkpoint
- confusion matrix image
- threshold sweep results

## Experiment Playbook

The repository is now mature enough that experiments should be structured rather than ad hoc.

### Recommended ablation groups

1. Input ablations.
   Compare 11-band only vs 16-channel input with spectral indices.

2. Imbalance ablations.
   Compare different class weights and copy-paste probabilities.

3. Decoder-capacity ablations.
   Compare the current slim decoder with the earlier wider decoder.

4. Evaluation ablations.
   Compare no TTA, standard TTA, and multi-scale TTA.

5. Post-processing ablations.
   Compare raw raster predictions against refined polygon outputs.

### Recommended experiment table for the paper

| Group | Variant | Keep fixed |
|---|---|---|
| Input | 11-band vs 16-band | architecture, loss, schedule |
| Imbalance | class weights and copy-paste | architecture, threshold policy |
| Architecture | slim vs wide decoder | data pipeline and loss |
| Inference | base vs TTA vs MS-TTA | checkpoint |
| Deployment | raw mask vs postprocessed polygons | evaluated predictions |

### Practical selection rule

Use validation debris F1 to choose checkpoints, then report test-set behavior at both:
- default threshold `0.5`
- threshold-swept optimum

This gives a more honest picture of both raw and calibrated deployment performance.

## Known Results and Training History

### Proven historical signal

An earlier February 27 run demonstrated that the dataset can support very high debris recall when the architecture and loss path behave correctly. That run is the reason the project pivoted back toward the custom CBAM U-Net family.

### Recent evaluation behavior

A more recent run showed:
- debris recall can reach around 80 percent on test data even before full convergence
- precision remains the harder part of the problem
- threshold calibration changes operational behavior substantially

This means the project is now in the right regime: training is learning debris, but precision and calibration remain the main optimization target.

### Important interpretation

For this repository, `90+ accuracy` is not the correct headline goal. Accuracy is easy to inflate because the negative class dominates. The meaningful research targets are:
- high debris recall
- usable debris precision
- improved debris F1
- good thresholded operational performance

## Limitations

The current repository is strong, but not finished in a research-perfect sense.

Known limitations include:
- strong class imbalance still makes precision recovery difficult
- current headline evaluation is still pixel-centric rather than object-centric
- CPU training makes long ablation cycles expensive
- threshold calibration is global, not yet scene-adaptive
- drift forecasting quality depends on the quality and realism of external current and wind data
- the PowerShell runner still reflects some legacy interfaces and needs synchronization over time

These are not fatal issues, but they should be acknowledged clearly in any thesis, report, or paper.

## How To Position The Novelty In A Paper

If this repository is being written up for publication or dissertation work, the strongest framing is not “we trained a segmentation model.” That is too weak.

The stronger framing is:
- binary reformulation of MARIDA for operational debris search
- multispectral plus physically motivated spectral-index fusion
- confidence-aware and imbalance-aware supervision pipeline
- attention-enhanced encoder-decoder for rare marine debris segmentation
- threshold-calibrated evaluation for operational deployment
- raster-to-vector conversion and forecast coupling for end-to-end monitoring

A concise novelty statement could be written like this:

"We propose an end-to-end marine debris monitoring workflow that integrates multispectral rare-object segmentation, confidence-aware supervision, debris-focused augmentation, threshold-calibrated inference, geospatial object extraction, and downstream drift forecasting within a single operational remote-sensing pipeline."

That is a stronger and more defensible research claim than describing the work as a standalone semantic segmentation experiment.

## Troubleshooting

### Symptom: very high accuracy but useless debris detection

Cause:
- majority-class dominance

Check:
- debris recall
- debris precision
- debris F1

### Symptom: recall spikes but precision collapses

Cause:
- imbalance handling too aggressive
- too much copy-paste or oversampling
- threshold too low

Current mitigation already implemented:
- reduced class weights
- reduced copy-paste probability
- reduced oversampling
- slimmer decoder

### Symptom: model predicts almost no debris

Cause:
- underweighted positive class
- wrong checkpoint
- threshold too high

Check:
- threshold sweep output from `evaluate.py`
- validation recall during training

### Symptom: evaluation command exits with code 1 even though results print

Cause:
- PowerShell wraps Python warnings or stderr output as non-zero in some terminal contexts

Interpretation:
- if metrics are printed and outputs are saved, the evaluation usually still succeeded

### Symptom: Albumentations warning on affine mode

Status:
- already fixed in the current codebase

## Roadmap

Recommended next research steps:

1. Continue precision recovery experiments.
   Tune threshold, oversampling, copy-paste probability, and class weights jointly rather than in isolation.

2. Add calibration analysis.
   Reliability curves, PR curves, and per-scene threshold adaptation would strengthen the paper.

3. Add architecture ablations.
   Compare the current slim decoder against the wider decoder and against the Random Forest baseline using identical splits.

4. Add object-level evaluation.
   Pixel metrics are not enough for operational debris search. Object detection success after polygonization would be stronger.

5. Tighten the detection-to-drift story.
   Use predicted debris polygons as real initialization sets and quantify forecast spread across the ensemble.

## Bottom Line

This repository is no longer just a generic segmentation project. It is a rare-object remote-sensing workflow that combines:
- multispectral debris detection
- confidence-aware supervision
- imbalance-focused augmentation
- threshold-calibrated evaluation
- GIS-ready spatial export
- downstream drift forecasting

The README has been updated to reflect the code that actually exists now, the architectural changes already made, and the pieces of logic that matter for both reproducibility and research novelty.
