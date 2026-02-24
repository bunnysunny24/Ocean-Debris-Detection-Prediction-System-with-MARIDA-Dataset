"""
utils/spectral_extraction.py
-----------------------------
Extract per-pixel spectral signatures from MARIDA patches and save to HDF5.
Required for Random Forest training.

Usage:
    python utils/spectral_extraction.py               # raw bands  → Dataset/dataset.h5
    python utils/spectral_extraction.py --type indices # SI         → Dataset/dataset_si.h5
    python utils/spectral_extraction.py --type texture # GLCM       → Dataset/dataset_glcm.h5
"""

import os, sys, glob, argparse
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    PATCHES_DIR, SPLITS_DIR, DATA_DIR, INPUT_BANDS, CLASS_NAMES
)

BAND_NAMES = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]


# ── Spectral indices ──────────────────────────────────────────────────────────

def compute_spectral_indices(arr: np.ndarray) -> np.ndarray:
    """
    arr : (11, H, W)  band order: B1..B12 excluding B10
    Returns stacked SI array (n_indices, H, W)
    """
    eps = 1e-8
    B3  = arr[2].astype(float)   # Green
    B4  = arr[3].astype(float)   # Red
    B8  = arr[7].astype(float)   # NIR
    B11 = arr[9].astype(float)   # SWIR1
    B12 = arr[10].astype(float)  # SWIR2

    NDVI  = (B8 - B4) / (B8 + B4 + eps)
    NDWI  = (B3 - B8) / (B3 + B8 + eps)
    FDI   = B8 - (B4 + (B12 - B4) * (832.8 - 664.6) / (2201.5 - 664.6))   # Floating Debris Index
    PI    = B4 / (B3 + B4 + B8 + eps)                                        # Plastic Index
    RNDVI = (B4 - B3) / (B4 + B3 + eps)

    return np.stack([NDVI, NDWI, FDI, PI, RNDVI], axis=0).astype(np.float32)


# ── GLCM texture ──────────────────────────────────────────────────────────────

def compute_glcm_features(arr: np.ndarray, band_idx: int = 7) -> np.ndarray:
    """
    Simple local texture using variance, entropy approximation on one band.
    Full GLCM is extremely slow; this is a fast proxy.
    Returns (2, H, W): [local_variance, local_mean]
    """
    from scipy.ndimage import uniform_filter, generic_filter
    band = arr[band_idx].astype(np.float32)
    win  = 5
    mean = uniform_filter(band, win)
    sq   = uniform_filter(band**2, win)
    var  = np.maximum(sq - mean**2, 0.0)
    return np.stack([var, mean], axis=0)


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_split(split: str, mode: str = "bands") -> pd.DataFrame:
    """
    mode: 'bands' | 'indices' | 'texture'
    Returns DataFrame with columns = band/feature names + 'label' + 'conf'
    """
    split_file = os.path.join(SPLITS_DIR, f"{split}_X.txt")
    with open(split_file) as f:
        patch_ids = [l.strip() for l in f if l.strip()]

    rows = []
    for pid in tqdm(patch_ids, desc=f"[{split}]"):
        # Try both with and without S2_ prefix
        tif_patterns = [os.path.join(PATCHES_DIR, "**", f"{pid}.tif"),
                       os.path.join(PATCHES_DIR, "**", f"S2_{pid}.tif")]
        matches = []
        for pattern in tif_patterns:
            found = glob.glob(pattern, recursive=True)
            found = [m for m in found if not m.endswith(("_cl.tif", "_conf.tif"))]
            matches.extend(found)
        if not matches:
            continue
        tif = matches[0]

        cl_patterns = [os.path.join(PATCHES_DIR, "**", f"{pid}_cl.tif"),
                       os.path.join(PATCHES_DIR, "**", f"S2_{pid}_cl.tif")]
        cl_matches = []
        for pattern in cl_patterns:
            cl_matches.extend(glob.glob(pattern, recursive=True))

        conf_patterns = [os.path.join(PATCHES_DIR, "**", f"{pid}_conf.tif"),
                         os.path.join(PATCHES_DIR, "**", f"S2_{pid}_conf.tif")]
        conf_matches = []
        for pattern in conf_patterns:
            conf_matches.extend(glob.glob(pattern, recursive=True))

        if not cl_matches:
            continue

        with rasterio.open(tif) as src:
            arr = src.read(list(range(1, INPUT_BANDS + 1))).astype(np.float32)  # (11,H,W)
        with rasterio.open(cl_matches[0]) as src:
            cl  = src.read(1)  # (H,W)
        conf = np.zeros_like(cl, dtype=np.uint8)
        if conf_matches:
            with rasterio.open(conf_matches[0]) as src:
                conf = src.read(1).astype(np.uint8)

        # Build feature array
        if mode == "bands":
            feats = arr.reshape(INPUT_BANDS, -1).T   # (H*W, 11)
            col_names = BAND_NAMES
        elif mode == "indices":
            si    = compute_spectral_indices(arr)
            feats = si.reshape(si.shape[0], -1).T
            col_names = ["NDVI","NDWI","FDI","PI","RNDVI"]
        elif mode == "texture":
            tex   = compute_glcm_features(arr)
            feats = tex.reshape(tex.shape[0], -1).T
            col_names = ["NIR_var","NIR_mean"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        labels = cl.ravel()
        confs  = conf.ravel()

        # Only keep annotated pixels
        valid = labels > 0
        feats  = feats[valid]
        labels = labels[valid]
        confs  = confs[valid]

        df_patch = pd.DataFrame(feats, columns=col_names)
        df_patch["label"] = labels.astype(np.uint8)
        df_patch["conf"]  = confs.astype(np.uint8)
        df_patch["patch"] = pid
        rows.append(df_patch)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(args):
    mode     = args.type
    out_name = {"bands": "dataset.h5", "indices": "dataset_si.h5", "texture": "dataset_glcm.h5"}[mode]
    out_path = os.path.join(DATA_DIR, out_name)

    print(f"Extracting spectral signatures | mode={mode} → {out_path}")

    with pd.HDFStore(out_path, mode="w", complevel=5, complib="blosc") as store:
        for split in ("train", "val", "test"):
            df = extract_split(split, mode)
            store.put(split, df, format="table", data_columns=True)
            print(f"  {split}: {len(df):,} pixels")

    print(f"Done → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--type", default="bands", choices=["bands", "indices", "texture"])
    main(p.parse_args())
