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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    PATCHES_DIR, SPLITS_DIR, INPUT_BANDS, PATCH_SIZE,
    AUG_PROB, ELASTIC_ALPHA, ELASTIC_SIGMA,
    COPY_PASTE_PROB, OVERSAMPLE_HEAVY, OVERSAMPLE_LIGHT,
    RAW_BANDS, USE_SPECTRAL_INDICES,
    MIXUP_PROB, MIXUP_ALPHA,
)


# ── Cached file-path index (built once, avoids repeated recursive glob) ──
_FILE_INDEX = {}   # basename (without ext) → full path


def _build_file_index():
    """Walk PATCHES_DIR once and index every .tif by its stem."""
    global _FILE_INDEX
    if _FILE_INDEX:
        return
    for root, _dirs, files in os.walk(PATCHES_DIR):
        for fname in files:
            if fname.lower().endswith(".tif"):
                stem = fname[:-4]  # e.g. "S2_1-12-19_48MYU_2" or "..._cl" / "..._conf"
                _FILE_INDEX[stem] = os.path.join(root, fname)
    print(f"[Index] {len(_FILE_INDEX)} TIF files indexed from {PATCHES_DIR}")


def _find_tif(patch_id: str):
    """Locate the image .tif for *patch_id* using the cached index."""
    _build_file_index()
    for key in (patch_id, f"S2_{patch_id}"):
        path = _FILE_INDEX.get(key)
        if path:
            return path
    return None


def _find_mask_tif(patch_id: str):
    _build_file_index()
    for key in (f"{patch_id}_cl", f"S2_{patch_id}_cl"):
        path = _FILE_INDEX.get(key)
        if path:
            return path
    return None


def _find_conf_tif(patch_id: str):
    _build_file_index()
    for key in (f"{patch_id}_conf", f"S2_{patch_id}_conf"):
        path = _FILE_INDEX.get(key)
        if path:
            return path
    return None


# ── Spectral normalisation stats (computed from training data; update if needed) ──
# shape: (RAW_BANDS,)  –– percentile-based per-band clipping + scaling
BAND_P2  = np.array([0.]*RAW_BANDS)   # will be filled lazily
BAND_P98 = np.array([1.]*RAW_BANDS)
_STATS_COMPUTED = False


def _compute_spectral_indices(arr: np.ndarray) -> np.ndarray:
    """Compute 5 spectral indices from raw Sentinel-2 bands.
    arr: (11, H, W) band order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
    Returns: (5, H, W) -> NDVI, NDWI, FDI, PI, RNDVI"""
    eps = 1e-8
    B3  = arr[2].astype(np.float32)   # Green
    B4  = arr[3].astype(np.float32)   # Red
    B8  = arr[7].astype(np.float32)   # NIR
    B12 = arr[10].astype(np.float32)  # SWIR2

    NDVI  = (B8 - B4) / (B8 + B4 + eps)
    NDWI  = (B3 - B8) / (B3 + B8 + eps)
    FDI   = B8 - (B4 + (B12 - B4) * (832.8 - 664.6) / (2201.5 - 664.6))
    PI    = B4 / (B3 + B4 + B8 + eps)
    RNDVI = (B4 - B3) / (B4 + B3 + eps)

    return np.stack([NDVI, NDWI, FDI, PI, RNDVI], axis=0).astype(np.float32)


def _compute_norm_stats(split_file: str, n_samples: int = 0):
    """Compute per-band 2nd / 98th percentile from training patches.
    Uses ALL patches (n_samples=0) for deterministic, reproducible stats."""
    global BAND_P2, BAND_P98, _STATS_COMPUTED
    with open(split_file) as f:
        patch_ids = [l.strip() for l in f if l.strip()]
    if n_samples > 0 and n_samples < len(patch_ids):
        random.seed(42)                       # fixed seed for reproducibility
        random.shuffle(patch_ids)
        patch_ids = patch_ids[:n_samples]
        random.seed()                         # re-seed

    all_vals = [[] for _ in range(RAW_BANDS)]
    for pid in patch_ids:
        tif = _find_tif(pid)
        if tif is None:
            continue
        with rasterio.open(tif) as src:
            arr = src.read().astype(np.float32)   # (B, H, W)
        arr = arr[:RAW_BANDS]
        for b in range(min(RAW_BANDS, arr.shape[0])):
            all_vals[b].append(arr[b].ravel())

    for b in range(RAW_BANDS):
        if all_vals[b]:
            cat = np.concatenate(all_vals[b])
            BAND_P2[b]  = np.nanpercentile(cat, 2)
            BAND_P98[b] = np.nanpercentile(cat, 98)
        else:
            BAND_P2[b], BAND_P98[b] = 0.0, 1.0
    _STATS_COMPUTED = True
    print("[Norm] Band stats computed.")


def normalise(arr: np.ndarray) -> np.ndarray:
    """Percentile clip + [0,1] scale per band. arr shape: (B, H, W)."""
    out = np.zeros_like(arr, dtype=np.float32)
    for b in range(arr.shape[0]):
        lo, hi = BAND_P2[b], BAND_P98[b]
        out[b] = np.clip((arr[b] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return out


# ── Albumentations advanced augmentation pipeline ──
def get_advanced_augment():
    # NOTE: Labels are extremely sparse (~30 pixels / 65536).
    #   Avoid RandomResizedCrop and CoarseDropout – they can discard the
    #   few labeled pixels entirely.
    # NOTE: var_limit must be tiny for float32 [0,1] images (default 10-50
    #   is for uint8 and would destroy the image).
    # albumentations v2.x uses std_range (fraction of value range).
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
        A.ElasticTransform(alpha=ELASTIC_ALPHA, sigma=ELASTIC_SIGMA, p=0.15),
        A.GridDistortion(p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
    ], additional_targets={'conf_raw': 'mask'})


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

        # Always use full split (model needs both debris AND non-debris examples)
        split_file = os.path.join(SPLITS_DIR, f"{split}_X.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")


        with open(split_file) as f:
            patch_ids = [l.strip() for l in f if l.strip()]

        # ── Filter out patches with zero labeled pixels (pure nodata) ──
        valid_pids = []
        for pid in patch_ids:
            mask_tif = _find_mask_tif(pid)
            if mask_tif is None:
                continue
            with rasterio.open(mask_tif) as src:
                raw = src.read(1)
            # DN: 0=nodata, 1-15=classes.  Any non-zero pixel means has labels.
            if (raw > 0).sum() > 0:
                valid_pids.append(pid)
        dropped = len(patch_ids) - len(valid_pids)
        if dropped:
            print(f"[Dataset] {split}: dropped {dropped} patches with zero labels")
        patch_ids = valid_pids

        # Oversampling for training: duplicate ALL debris patches
        if split == "train":
            debris_patches = []
            normal_patches = []
            self._debris_pool = []  # IDs of patches with debris (for copy-paste aug)
            for pid in patch_ids:
                mask_tif = _find_mask_tif(pid)
                if mask_tif:
                    with rasterio.open(mask_tif) as src:
                        mask = src.read(1).astype(np.int64) - 1
                    n_debris = (mask == 0).sum()
                    if n_debris >= 20:
                        debris_patches.extend([pid] * OVERSAMPLE_HEAVY)
                        self._debris_pool.append(pid)
                    elif n_debris >= 1:
                        debris_patches.extend([pid] * OVERSAMPLE_LIGHT)
                        self._debris_pool.append(pid)
                    else:
                        normal_patches.append(pid)
                else:
                    normal_patches.append(pid)
            self.patch_ids = debris_patches + normal_patches
            print(f"[Dataset] Copy-paste pool: {len(self._debris_pool)} debris patches")
        else:
            self.patch_ids = patch_ids
            self._debris_pool = []

        # Compute normalisation stats from training split on first call
        global _STATS_COMPUTED
        if not _STATS_COMPUTED:
            train_split = os.path.join(SPLITS_DIR, "train_X.txt")
            _compute_norm_stats(train_split)

        print(f"[Dataset] {split}: {len(self.patch_ids)} patches")

    def __len__(self):
        return len(self.patch_ids)

    def _copy_paste_debris(self, img, mask, conf_raw):
        """Copy-paste augmentation: paste debris pixels from a random donor patch.
        This dramatically increases effective debris pixel count per sample.
        img: (B,H,W) normalised float32, mask: (H,W) int64, conf_raw: (H,W) int64."""
        if not self._debris_pool or random.random() > COPY_PASTE_PROB:
            return img, mask, conf_raw

        # Paste from 1 to 3 donors for more debris diversity
        n_donors = random.randint(1, 5)   # increased from 3 for richer debris diversity
        for _ in range(n_donors):
            donor_pid = random.choice(self._debris_pool)
            donor_tif = _find_tif(donor_pid)
            donor_mask_tif = _find_mask_tif(donor_pid)
            if not donor_tif or not donor_mask_tif:
                continue

            # Load donor image + mask
            with rasterio.open(donor_tif) as src:
                donor_img = src.read(list(range(1, RAW_BANDS + 1))).astype(np.float32)
            np.nan_to_num(donor_img, copy=False)
            # Compute donor spectral indices before normalisation
            if USE_SPECTRAL_INDICES:
                donor_si = _compute_spectral_indices(donor_img)
            donor_img = normalise(donor_img)
            if USE_SPECTRAL_INDICES:
                donor_si = np.clip(donor_si, -3.0, 3.0)
                donor_si = (donor_si + 3.0) / 6.0
                donor_img = np.concatenate([donor_img, donor_si], axis=0)

            with rasterio.open(donor_mask_tif) as src:
                donor_mask_raw = src.read(1).astype(np.int64) - 1
            donor_bin = np.full_like(donor_mask_raw, fill_value=1, dtype=np.int64)
            donor_bin[donor_mask_raw == 0] = 0   # debris
            donor_bin[donor_mask_raw == -1] = -1  # nodata

            # Find debris pixel coordinates in donor
            debris_ys, debris_xs = np.where(donor_bin == 0)
            if len(debris_ys) == 0:
                continue

            # Random spatial offset for pasting (shift debris cluster)
            dy = random.randint(-128, 128)
            dx = random.randint(-128, 128)

            # Random geometric transform: flip/rotate
            do_hflip = random.random() > 0.5
            do_vflip = random.random() > 0.5
            k_rot = random.randint(0, 3)  # 0, 90, 180, 270 degrees

            H, W = mask.shape
            for y, x in zip(debris_ys, debris_xs):
                # Apply flip
                sy, sx = y, x
                if do_hflip:
                    sx = W - 1 - sx
                if do_vflip:
                    sy = H - 1 - sy
                # Apply rotation (90-degree increments)
                for _ in range(k_rot):
                    sy, sx = sx, H - 1 - sy

                ny, nx = sy + dy, sx + dx
                if 0 <= ny < H and 0 <= nx < W:
                    img[:, ny, nx] = donor_img[:, y, x]
                    mask[ny, nx] = 0        # debris
                    conf_raw[ny, nx] = 2    # high confidence

        return img, mask, conf_raw

    def __getitem__(self, idx):
        pid  = self.patch_ids[idx]

        # ── Load image ──
        tif = _find_tif(pid)
        if tif is None:
            raise FileNotFoundError(f"Patch tif not found for id: {pid}")
        with rasterio.open(tif) as src:
            img = src.read(list(range(1, RAW_BANDS + 1))).astype(np.float32)   # (11,H,W)
        np.nan_to_num(img, copy=False)  # replace NaN with 0 (some patches have NaN in bands 9-10)

        # ── Compute spectral indices before normalisation (need raw reflectance) ──
        if USE_SPECTRAL_INDICES:
            si = _compute_spectral_indices(img)  # (5, H, W)

        # ── Load mask ──
        mask_tif = _find_mask_tif(pid)
        if mask_tif:
            with rasterio.open(mask_tif) as src:
                mask = src.read(1).astype(np.int64)  # (H,W) DN 1-15 or 0=nodata
        else:
            mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.int64)

        # Convert DN [1-15] → binary: debris=0, not-debris=1; nodata (0) → -1
        mask = mask - 1   # [0-14], nodata= -1
        mask_bin = np.full_like(mask, fill_value=1, dtype=np.int64)  # default not-debris
        mask_bin[mask == 0]  = 0   # debris
        mask_bin[mask == -1] = -1  # nodata
        mask = mask_bin

        # ── Load confidence as integer (0=uncertain, 1=confident, 2=highly) ──
        conf_raw = np.full(mask.shape, fill_value=2, dtype=np.int64)
        if self.use_conf_weights:
            conf_tif = _find_conf_tif(pid)
            if conf_tif:
                with rasterio.open(conf_tif) as src:
                    conf_raw = src.read(1).astype(np.int64)  # 0, 1, 2

        # ── Normalise raw bands ──
        img = normalise(img)

        # ── Append spectral indices (already in meaningful range, clip to [-1,1] then shift to [0,1]) ──
        if USE_SPECTRAL_INDICES:
            si = np.clip(si, -3.0, 3.0)  # clip extreme values
            si = (si + 3.0) / 6.0        # map to [0, 1]
            img = np.concatenate([img, si], axis=0)  # (16, H, W)

        # ── Copy-paste debris augmentation (before spatial aug, after normalisation) ──
        if self.augment_data:
            img, mask, conf_raw = self._copy_paste_debris(img, mask, conf_raw)

        # ── MixUp augmentation (blend two training patches) ──
        # Applied after copy-paste so synthetic debris is not diluted
        # Uses a soft label blending approach: masks are mixed proportionally
        if self.augment_data and random.random() < MIXUP_PROB and len(self.patch_ids) > 1:
            mix_lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            mix_lam = max(mix_lam, 1.0 - mix_lam)   # keep dominant patch ≥ 0.5 weight
            # Load another random patch from this dataset
            mix_idx = random.randrange(len(self.patch_ids))
            mix_pid = self.patch_ids[mix_idx]
            mix_tif = _find_tif(mix_pid)
            mix_mask_tif = _find_mask_tif(mix_pid)
            if mix_tif and mix_mask_tif:
                with rasterio.open(mix_tif) as src:
                    mix_img = src.read(list(range(1, RAW_BANDS + 1))).astype(np.float32)
                np.nan_to_num(mix_img, copy=False)
                if USE_SPECTRAL_INDICES:
                    mix_si = _compute_spectral_indices(mix_img)
                mix_img = normalise(mix_img)
                if USE_SPECTRAL_INDICES:
                    mix_si = np.clip(mix_si, -3.0, 3.0)
                    mix_si = (mix_si + 3.0) / 6.0
                    mix_img = np.concatenate([mix_img, mix_si], axis=0)
                with rasterio.open(mix_mask_tif) as src:
                    mix_mask_raw = src.read(1).astype(np.int64) - 1
                mix_mask_bin = np.full_like(mix_mask_raw, 1, dtype=np.int64)
                mix_mask_bin[mix_mask_raw == 0]  = 0
                mix_mask_bin[mix_mask_raw == -1] = -1
                # Blend images; for masks use the current patch's mask (dominant)
                # This avoids nodata confusion from the second patch
                img = mix_lam * img + (1.0 - mix_lam) * mix_img

        # ── Albumentations expects (H,W,C) for images, (H,W) for mask ──
        img = np.transpose(img, (1, 2, 0))  # (H,W,B)

        # ── Augment image, mask AND conf together (same spatial transform) ──
        if self.augment_data:
            aug = get_advanced_augment()
            augmented = aug(image=img, mask=mask, conf_raw=conf_raw)
            img      = augmented["image"]
            mask     = augmented["mask"]
            conf_raw = augmented["conf_raw"]
        else:
            img  = img.copy()
            mask = mask.copy()

        # Map conf_raw integers → float weights (after augmentation so alignment is correct)
        conf = np.where(conf_raw == 0, 0.2,
               np.where(conf_raw == 1, 0.7, 1.0)).astype(np.float32)

        # Convert back to (B,H,W)
        img = np.transpose(img, (2, 0, 1))

        img  = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        conf = torch.from_numpy(conf).float()

        return {"image": img, "mask": mask, "conf": conf, "id": pid}
