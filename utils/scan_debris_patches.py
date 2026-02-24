import os
import numpy as np
import rasterio
from configs.config import SPLITS_DIR, PATCHES_DIR
from utils.dataset import _find_mask_tif

def scan_split(split_name):
    split_file = os.path.join(SPLITS_DIR, f"{split_name}_X.txt")
    with open(split_file) as f:
        patch_ids = [l.strip() for l in f if l.strip()]
    debris_patches = []
    for pid in patch_ids:
        mask_tif = _find_mask_tif(pid)
        if not mask_tif:
            print(f"[WARN] No mask for patch {pid}")
            continue
        with rasterio.open(mask_tif) as src:
            mask = src.read(1).astype(np.int64) - 1  # [0-14], -1=nodata
        debris_pixels = np.sum(mask == 0)
        if debris_pixels > 0:
            debris_patches.append(pid)
        print(f"Patch {pid}: debris_pixels={debris_pixels}")
    print(f"Total patches with debris in {split_name}: {len(debris_patches)} / {len(patch_ids)}")
    # Optionally, write new split file with only debris patches
    out_file = os.path.join(SPLITS_DIR, f"{split_name}_X_debris.txt")
    with open(out_file, "w") as f:
        for pid in debris_patches:
            f.write(pid + "\n")
    print(f"Wrote: {out_file}")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        print(f"\n=== Scanning {split} split ===")
        scan_split(split)