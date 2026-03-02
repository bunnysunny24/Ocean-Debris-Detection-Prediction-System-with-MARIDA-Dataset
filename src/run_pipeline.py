"""
run_pipeline.py
---------------
Master script: runs the full end-to-end marine debris detection + drift pipeline.
Steps:
    1. Spectral extraction     (for RF; optional)
    2. Train segmentation model
    3. Evaluate on test set
    4. Post-process → GeoJSON
    5. Drift prediction

Usage:
    python run_pipeline.py                          # full pipeline, deep learning only
    python run_pipeline.py --with_rf               # also train Random Forest
    python run_pipeline.py --skip_train --eval_only # skip training, just evaluate + postprocess
    python run_pipeline.py --ocean_nc cmems.nc --wind_nc era5.nc  # real current fields
"""


import os, sys, argparse, subprocess
from configs.config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE, NUM_WORKERS,
    CHECKPOINTS_DIR, LOGS_DIR, OUTPUTS_DIR, PATCHES_DIR, SPLITS_DIR,
    H5_PATH, H5_SI_PATH, H5_GLCM_PATH, MIN_DEBRIS_PIXELS
)

BASE = os.path.dirname(__file__)


def run(cmd: list):
    print(f"\n{'='*60}")
    print(f">>> {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main(args):
    # ── Step 0: Spectral extraction (always needed for RF) ──
    import os
    from configs.config import H5_PATH, H5_SI_PATH, H5_GLCM_PATH
    if args.with_rf or args.extract_only:
        # Bands
        if not os.path.exists(H5_PATH):
            run([sys.executable, "utils/spectral_extraction.py", "--type", "bands"])
        else:
            print(f"[INFO] Skipping bands extraction: {H5_PATH} already exists.")
        # Indices
        if args.with_si:
            if not os.path.exists(H5_SI_PATH):
                run([sys.executable, "utils/spectral_extraction.py", "--type", "indices"])
            else:
                print(f"[INFO] Skipping indices extraction: {H5_SI_PATH} already exists.")
        # Texture
        if args.with_glcm:
            if not os.path.exists(H5_GLCM_PATH):
                run([sys.executable, "utils/spectral_extraction.py", "--type", "texture"])
            else:
                print(f"[INFO] Skipping texture extraction: {H5_GLCM_PATH} already exists.")

    if args.extract_only:
        print("Extraction done. Exiting.")
        return

    # ── Step 1: Train segmentation model ──
    if not args.skip_train:
        train_cmd = [
            sys.executable, "train.py",
            "--epochs", str(args.epochs),
            "--batch",  str(args.batch),
            "--lr",     str(args.lr),
            "--workers", str(args.workers),
            "--weight_decay", str(args.weight_decay),
            "--patience", str(args.patience),
        ]
        if args.resume:
            train_cmd.append("--resume")
        run(train_cmd)

    # ── Step 2: Evaluate ──
    run([sys.executable, "evaluate.py", "--split", "test", "--save_masks"])

    # ── Step 3: Post-process ──
    from configs.config import OUTPUTS_DIR as _OUT
    run([
        sys.executable, "postprocess.py",
        "--pred_dir", os.path.join(_OUT, "predicted_test"),
        "--out_dir",  os.path.join(_OUT, "geospatial"),
    ])

    # ── Step 4: Drift prediction ──
    drift_cmd = [
        sys.executable, "drift_prediction/drift.py",
        "--geojson", os.path.join(_OUT, "geospatial", "all_debris.geojson"),
        "--out_dir", os.path.join(_OUT, "drift"),
    ]
    if args.ocean_nc:
        drift_cmd += ["--ocean_nc", args.ocean_nc]
    if args.wind_nc:
        drift_cmd += ["--wind_nc", args.wind_nc]
    run(drift_cmd)

    # ── Step 5 (optional): Random Forest ──
    if args.with_rf:
        rf_cmd = [sys.executable, "random_forest/train_eval.py"]
        if args.with_si:   rf_cmd.append("--use_si")
        if args.with_glcm: rf_cmd.append("--use_glcm")
        run(rf_cmd)

    print("\n✅ Full pipeline complete!")
    print("   Outputs → ./outputs/")
    print("   Logs    → ./logs/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Pipeline control
    p.add_argument("--skip_train",   action="store_true", help="Skip model training")
    p.add_argument("--eval_only",    action="store_true", help="Only evaluate + postprocess")
    p.add_argument("--extract_only", action="store_true", help="Only run spectral extraction")
    p.add_argument("--resume",       action="store_true", help="Resume training from checkpoint")
    # Training (imported from config)
    p.add_argument("--epochs", type=int,   default=EPOCHS)
    p.add_argument("--batch",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",     type=float, default=LR)
    p.add_argument("--workers", type=int,  default=NUM_WORKERS)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--patience", type=int, default=PATIENCE)
    # Random Forest
    p.add_argument("--with_rf",   action="store_true", help="Also train Random Forest")
    p.add_argument("--with_si",   action="store_true", help="Include spectral indices for RF")
    p.add_argument("--with_glcm", action="store_true", help="Include GLCM features for RF")
    # Drift
    p.add_argument("--ocean_nc", default=None, help="CMEMS NetCDF path")
    p.add_argument("--wind_nc",  default=None, help="ERA5 NetCDF path")
    main(p.parse_args())
