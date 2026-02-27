"""
utils/dataset.py
----------------
PyTorch Dataset for MARIDA patches with:
  - robust percentile normalisation
  - geometric + spectral augmentation
  - optional confidence-weighted loss masking
"""

import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from scipy.ndimage import map_coordinates, gaussian_filter
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    PATCHES_DIR, SPLITS_DIR, INPUT_BANDS, PATCH_SIZE,
    AUG_PROB, ELASTIC_ALPHA, ELASTIC_SIGMA,
    SPECTRAL_NOISE, BRIGHTNESS_RANGE,
)


# ── Spectral normalisation stats (computed from training data; update if needed) ──
# shape: (INPUT_BANDS,)  –– percentile-based per-band clipping + scaling
BAND_P2  = np.array([0.]*INPUT_BANDS)   # will be filled lazily
BAND_P98 = np.array([1.]*INPUT_BANDS)
_STATS_COMPUTED = False


def _compute_norm_stats(split_file: str, n_samples: int = 200):
    """Compute per-band 2nd / 98th percentile from a random subset of training patches."""
    global BAND_P2, BAND_P98, _STATS_COMPUTED
    with open(split_file) as f:
        patch_ids = [l.strip() for l in f if l.strip()]
    random.shuffle(patch_ids)
    patch_ids = patch_ids[:n_samples]

    all_vals = [[] for _ in range(INPUT_BANDS)]
    for pid in patch_ids:
        # find the tif for this patch id
        tif = _find_tif(pid)
        if tif is None:
            continue
        with rasterio.open(tif) as src:
            arr = src.read().astype(np.float32)   # (B, H, W)
        arr = arr[:INPUT_BANDS]
        for b in range(min(INPUT_BANDS, arr.shape[0])):
            all_vals[b].append(arr[b].ravel())

    for b in range(INPUT_BANDS):
        if all_vals[b]:
            cat = np.concatenate(all_vals[b])
            BAND_P2[b]  = np.percentile(cat, 2)
            BAND_P98[b] = np.percentile(cat, 98)
        else:
            BAND_P2[b], BAND_P98[b] = 0.0, 1.0
    _STATS_COMPUTED = True
    print("[Norm] Band stats computed.")


def _find_tif(patch_id: str):
    """Locate the .tif file for a patch ID (searches recursively under PATCHES_DIR).
    Tries both with and without 'S2_' prefix."""
    patterns = [os.path.join(PATCHES_DIR, "**", f"{patch_id}.tif"),
                os.path.join(PATCHES_DIR, "**", f"S2_{patch_id}.tif")]
    matches = []
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        found = [m for m in found if not m.endswith("_cl.tif") and not m.endswith("_conf.tif")]
        matches.extend(found)
    return matches[0] if matches else None


def _find_mask_tif(patch_id: str):
    patterns = [os.path.join(PATCHES_DIR, "**", f"{patch_id}_cl.tif"),
                os.path.join(PATCHES_DIR, "**", f"S2_{patch_id}_cl.tif")]
    matches = []
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        matches.extend(found)
    return matches[0] if matches else None


def _find_conf_tif(patch_id: str):
    patterns = [os.path.join(PATCHES_DIR, "**", f"{patch_id}_conf.tif"),
                os.path.join(PATCHES_DIR, "**", f"S2_{patch_id}_conf.tif")]
    matches = []
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        matches.extend(found)
    return matches[0] if matches else None


def normalise(arr: np.ndarray) -> np.ndarray:
    """Percentile clip + [0,1] scale per band. arr shape: (B, H, W)."""
    out = np.zeros_like(arr, dtype=np.float32)
    for b in range(arr.shape[0]):
        lo, hi = BAND_P2[b], BAND_P98[b]
        out[b] = np.clip((arr[b] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return out


# ── Augmentation helpers ──────────────────────────────────────────────────────

def _random_hflip(img, mask):
    if random.random() < 0.5:
        img  = img[:, :, ::-1].copy()
        mask = mask[:, ::-1].copy()
    return img, mask


def _random_vflip(img, mask):
    if random.random() < 0.5:
        img  = img[:, ::-1, :].copy()
        mask = mask[::-1, :].copy()
    return img, mask


def _random_rot90(img, mask):
    k = random.randint(0, 3)
    img  = np.rot90(img,  k, axes=(1, 2)).copy()
    mask = np.rot90(mask, k, axes=(0, 1)).copy()
    return img, mask


def _elastic_transform(img, mask, alpha=ELASTIC_ALPHA, sigma=ELASTIC_SIGMA):
    """Elastic deformation – same displacement field for image and mask."""
    H, W = img.shape[1], img.shape[2]
    dx = gaussian_filter(np.random.randn(H, W), sigma) * alpha
    dy = gaussian_filter(np.random.randn(H, W), sigma) * alpha

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    coords_x = np.clip(x + dx, 0, W - 1)
    coords_y = np.clip(y + dy, 0, H - 1)
    coords = [coords_y.ravel(), coords_x.ravel()]

    out_img = np.zeros_like(img)
    for b in range(img.shape[0]):
        out_img[b] = map_coordinates(img[b], coords, order=1).reshape(H, W)
    out_mask = map_coordinates(mask.astype(np.float32), coords, order=0).reshape(H, W).astype(np.int64)
    return out_img, out_mask


def _spectral_noise(img, sigma=SPECTRAL_NOISE):
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img + noise, 0.0, 1.0)


def _spectral_brightness(img, rng=BRIGHTNESS_RANGE):
    shift = random.uniform(*rng)
    return np.clip(img + shift, 0.0, 1.0)


def augment(img: np.ndarray, mask: np.ndarray):
    """Apply random augmentation pipeline. img: (B,H,W) float32, mask: (H,W) int64."""
    if random.random() < AUG_PROB:
        img, mask = _random_hflip(img, mask)
    if random.random() < AUG_PROB:
        img, mask = _random_vflip(img, mask)
    if random.random() < AUG_PROB:
        img, mask = _random_rot90(img, mask)
    if random.random() < AUG_PROB * 0.5:
        img, mask = _elastic_transform(img, mask)
    if random.random() < AUG_PROB:
        img = _spectral_noise(img)
    if random.random() < AUG_PROB:
        img = _spectral_brightness(img)
    return img, mask


# ── Main Dataset ──────────────────────────────────────────────────────────────

class MARIDADataset(Dataset):
    """
    Parameters
    ----------
    split : str
        'train', 'val', or 'test'
    augment_data : bool
        Apply augmentation (only for train)
    use_conf_weights : bool
        Return per-pixel confidence weights for weighted loss
    """

    def __init__(self, split: str = "train", augment_data: bool = True,
                 use_conf_weights: bool = True):
        assert split in ("train", "val", "test")
        self.split         = split
        self.augment_data  = augment_data and (split == "train")
        self.use_conf_weights = use_conf_weights

        # Prefer debris-only split if available
        debris_split_file = os.path.join(SPLITS_DIR, f"{split}_X_debris.txt")
        split_file = debris_split_file if os.path.exists(debris_split_file) else os.path.join(SPLITS_DIR, f"{split}_X.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")


        with open(split_file) as f:
            patch_ids = [l.strip() for l in f if l.strip()]

        # Aggressive oversampling: duplicate debris-rich patches
        if split == "train":
            debris_patches = []
            normal_patches = []
            hard_negatives = []
            for pid in patch_ids:
                mask_tif = _find_mask_tif(pid)
                tif = _find_tif(pid)
                is_debris = False
                is_hard_negative = False
                if mask_tif:
                    with rasterio.open(mask_tif) as src:
                        mask = src.read(1).astype(np.int64) - 1
                    if (mask == 0).sum() > 10:
                        debris_patches.extend([pid]*5)
                        is_debris = True
                if not is_debris and tif:
                    with rasterio.open(tif) as src:
                        img = src.read(list(range(1, INPUT_BANDS + 1))).astype(np.float32)
                    img_norm = normalise(img)
                    # Hard negative: bright or high-variance patches (likely confused with debris)
                    if img_norm.mean() > 0.6 or img_norm.std() > 0.25:
                        hard_negatives.extend([pid]*3)
                        is_hard_negative = True
                if not is_debris and not is_hard_negative:
                    normal_patches.append(pid)
            self.patch_ids = debris_patches + hard_negatives + normal_patches
        else:
            self.patch_ids = patch_ids

        # Compute normalisation stats from training split on first call
        global _STATS_COMPUTED
        if not _STATS_COMPUTED:
            train_split = os.path.join(SPLITS_DIR, "train_X.txt")
            _compute_norm_stats(train_split)

        print(f"[Dataset] {split}: {len(self.patch_ids)} patches")

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid  = self.patch_ids[idx]

        # ── Load image ──
        tif = _find_tif(pid)
        if tif is None:
            raise FileNotFoundError(f"Patch tif not found for id: {pid}")
        with rasterio.open(tif) as src:
            img = src.read(list(range(1, INPUT_BANDS + 1))).astype(np.float32)   # (B,H,W)

        # ── Load mask ──
        mask_tif = _find_mask_tif(pid)
        if mask_tif:
            with rasterio.open(mask_tif) as src:
                mask = src.read(1).astype(np.int64)  # (H,W) DN 1-15 or 0=nodata
        else:
            mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.int64)

        # Convert DN [1-15] → binary: debris=0, not-debris=1; nodata (0) → -1
        mask = mask - 1   # [0-14], nodata= -1
        debris_mask = (mask == 0)
        not_debris_mask = (mask >= 1) & (mask <= 14)
        mask_bin = np.full_like(mask, fill_value=1, dtype=np.int64)  # default not-debris
        mask_bin[debris_mask] = 0
        mask_bin[mask == -1] = -1
        mask = mask_bin
        # Filter out patches with too few pixels of either class
        debris_count = np.sum(mask == 0)
        not_debris_count = np.sum(mask == 1)
        if debris_count < 10 or not_debris_count < 10:
            # Return None or skip this patch
            return self.__getitem__((idx + 1) % len(self.patch_ids))

        # ── Load confidence ──
        conf = np.ones_like(mask, dtype=np.float32)
        if self.use_conf_weights:
            conf_tif = _find_conf_tif(pid)
            if conf_tif:
                with rasterio.open(conf_tif) as src:
                    raw_conf = src.read(1).astype(np.float32)   # 0, 1, 2
                # map to weights: uncertain=0.2, confident=0.7, highly=1.0
                conf = np.where(raw_conf == 0, 0.2,
                       np.where(raw_conf == 1, 0.7, 1.0))

        # ── Normalise ──
        img = normalise(img)


        # ── Augment ──
        if self.augment_data:
            # Strongest augmentation for debris patches
            if (mask == 0).sum() > 0:
                for _ in range(5):
                    img, mask = augment(img, mask)
            else:
                for _ in range(2):
                    img, mask = augment(img, mask)

        img  = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        conf = torch.from_numpy(conf).float()

        return {"image": img, "mask": mask, "conf": conf, "id": pid}
