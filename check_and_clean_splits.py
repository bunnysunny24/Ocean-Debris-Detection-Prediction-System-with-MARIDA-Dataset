"""
check_and_clean_splits.py
------------------------
Checks all split files for missing patch .tif images and cleans them.
Removes any patch IDs from the split files that do not have a corresponding .tif in Dataset/patches.
"""

import os, glob

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'Dataset')
PATCHES_DIR = os.path.join(DATASET_DIR, 'patches')
SPLITS_DIR = os.path.join(DATASET_DIR, 'splits')

for split_name in ['train_X.txt', 'val_X.txt', 'test_X.txt']:
    split_path = os.path.join(SPLITS_DIR, split_name)
    if not os.path.exists(split_path):
        print(f"[WARN] Split file not found: {split_path}")
        continue
    with open(split_path) as f:
        patch_ids = [l.strip() for l in f if l.strip()]
    print(f"Checking {split_name} ({len(patch_ids)} IDs)...")
    valid_ids = []
    missing_ids = []
    for pid in patch_ids:
        tif_pattern = os.path.join(PATCHES_DIR, '**', f'{pid}.tif')
        matches = glob.glob(tif_pattern, recursive=True)
        matches = [m for m in matches if not m.endswith('_cl.tif') and not m.endswith('_conf.tif')]
        if matches:
            valid_ids.append(pid)
        else:
            missing_ids.append(pid)
    if missing_ids:
        print(f"  Missing ({len(missing_ids)}): {missing_ids}")
    else:
        print("  All patch IDs found.")
    # Overwrite split file with only valid IDs
    with open(split_path, 'w') as f:
        for pid in valid_ids:
            f.write(pid + '\n')
    print(f"  Cleaned {split_name}: {len(valid_ids)} valid IDs remain.\n")
print("Done. All split files cleaned.")
