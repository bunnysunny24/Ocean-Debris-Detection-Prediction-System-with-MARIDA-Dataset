"""
random_forest/train_eval.py
----------------------------
Train and evaluate a Random Forest classifier on MARIDA spectral features.
Requires: Dataset/dataset.h5 (run utils/spectral_extraction.py first)

Usage:
    python random_forest/train_eval.py
    python random_forest/train_eval.py --use_si --use_glcm
"""

import os, sys, argparse, logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import DATA_DIR, OUTPUTS_DIR, LOGS_DIR, CLASS_NAMES

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def load_split(h5_path: str, split: str, max_pixels: int = 500_000):
    store = pd.HDFStore(h5_path, mode="r")
    df    = store.select(split)
    store.close()
    if len(df) > max_pixels:
        df = df.sample(n=max_pixels, random_state=42)
    return df


def merge_features(*dfs):
    """Horizontally merge multiple feature DataFrames aligned on index."""
    base = dfs[0]
    label_conf = base[["label", "conf", "patch"]]
    feature_cols = [df.drop(columns=["label", "conf", "patch"], errors="ignore") for df in dfs]
    merged = pd.concat([label_conf] + feature_cols, axis=1)
    return merged


def main(args):
    log_path = os.path.join(LOGS_DIR, "rf_train_eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger()

    # ── Load features ──
    h5_main = os.path.join(DATA_DIR, "dataset.h5")
    if not os.path.exists(h5_main):
        log.error(f"dataset.h5 not found. Run:  python utils/spectral_extraction.py")
        sys.exit(1)

    log.info("Loading spectral signatures …")
    train_df = load_split(h5_main, "train")
    val_df   = load_split(h5_main, "val")
    test_df  = load_split(h5_main, "test")

    if args.use_si:
        h5_si = os.path.join(DATA_DIR, "dataset_si.h5")
        if os.path.exists(h5_si):
            log.info("Merging spectral indices …")
            train_df = merge_features(train_df, load_split(h5_si, "train"))
            val_df   = merge_features(val_df,   load_split(h5_si, "val"))
            test_df  = merge_features(test_df,  load_split(h5_si, "test"))

    if args.use_glcm:
        h5_glcm = os.path.join(DATA_DIR, "dataset_glcm.h5")
        if os.path.exists(h5_glcm):
            log.info("Merging GLCM features …")
            train_df = merge_features(train_df, load_split(h5_glcm, "train"))
            val_df   = merge_features(val_df,   load_split(h5_glcm, "val"))
            test_df  = merge_features(test_df,  load_split(h5_glcm, "test"))

    # ── Prepare arrays ──
    drop_cols = ["label", "conf", "patch"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore").values.astype(np.float32)
    y_train = (train_df["label"].values - 1).astype(int)   # 0-indexed
    X_test  = test_df.drop(columns=drop_cols, errors="ignore").values.astype(np.float32)
    y_test  = (test_df["label"].values - 1).astype(int)

    log.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Weight by confidence (0→0.2, 1→0.7, 2→1.0)
    conf_w  = np.where(train_df["conf"] == 0, 0.2,
              np.where(train_df["conf"] == 1, 0.7, 1.0))

    # ── Train RF ──
    log.info("Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        oob_score=True,
    )
    rf.fit(X_train, y_train, sample_weight=conf_w)
    log.info(f"OOB score: {rf.oob_score_:.4f}")

    # ── Evaluate ──
    y_pred = rf.predict(X_test)
    class_names = [CLASS_NAMES[i+1] for i in range(len(CLASS_NAMES))]

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)

    log.info(f"\nBalanced accuracy: {bal_acc:.4f}")
    log.info(f"\n{report}")

    # ── Save model ──
    model_path = os.path.join(OUTPUTS_DIR, "random_forest_model.joblib")
    joblib.dump(rf, model_path)
    log.info(f"Model saved → {model_path}")

    # ── Confusion matrix plot ──
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title("Random Forest – Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "rf_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    log.info(f"Confusion matrix saved → {cm_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--use_si",   action="store_true", help="Include spectral indices")
    p.add_argument("--use_glcm", action="store_true", help="Include GLCM texture features")
    main(p.parse_args())
