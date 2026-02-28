"""
evaluate.py
-----------
Evaluate the trained model on the test set.
Outputs:
  - Per-class IoU, F1, Precision, Recall, Dice
  - Confusion matrix (saved as PNG)
  - Predicted masks saved as GeoTIFF + PNG overlays

Usage:
    python evaluate.py
    python evaluate.py --split val --ckpt checkpoints/resnext_cbam_best.pth
"""

import os, sys, argparse, logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import rasterio
from rasterio.transform import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))
from configs.config import (
    INPUT_BANDS, NUM_CLASSES, CLASS_NAMES,
    CHECKPOINTS_DIR, OUTPUTS_DIR, LOGS_DIR, PATCHES_DIR
)
from utils.dataset import MARIDADataset, _find_tif
from semantic_segmentation.models.resnext_cbam_unet import ResNeXtCBAMUNet
try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None

# Color map for 15 MARIDA classes (index 0-14)
CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
]


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=6)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {save_path}")


def save_predicted_mask(pred_mask: np.ndarray, patch_id: str, out_dir: str):
    """Save predicted mask as GeoTIFF, copying CRS/transform from source patch."""
    os.makedirs(out_dir, exist_ok=True)
    tif = _find_tif(patch_id)
    out_path = os.path.join(out_dir, f"{patch_id}_pred.tif")

    if tif:
        with rasterio.open(tif) as src:
            profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write((pred_mask + 1).astype(np.uint8)[np.newaxis])  # DN 1-15
    else:
        np.save(out_path.replace(".tif", ".npy"), pred_mask)


@torch.no_grad()
def evaluate(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOGS_DIR, f"eval_{args.split}.log")),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model selection based on checkpoint keys ──
    # Load checkpoint first, then instantiate matching architecture
    try:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(args.ckpt, map_location=device)

    state_keys = list(ck.get("model", {}).keys()) if isinstance(ck, dict) else []
    use_smp = any(k.startswith("encoder.") or k.startswith("decoder.") or k.startswith("segmentation_head.") for k in state_keys)

    if use_smp and smp is not None:
        log.info("Checkpoint looks like an smp model. Instantiating DeepLabV3Plus to match checkpoint.")
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=INPUT_BANDS,
            classes=NUM_CLASSES,
            activation=None,
        ).to(device)
    else:
        model = ResNeXtCBAMUNet(in_channels=INPUT_BANDS, num_classes=NUM_CLASSES,
                                pretrained=False).to(device)

    # Try strict load first, fall back to non-strict with a warning
    try:
        model.load_state_dict(ck["model"])
    except RuntimeError as e:
        log.warning(f"Strict state_dict load failed: {e}. Trying non-strict load and continuing.")
        model.load_state_dict(ck["model"], strict=False)
    model.eval()
    log.info(f"Loaded checkpoint: {args.ckpt}")

    # ── Data ──
    ds = MARIDADataset(args.split, augment_data=False, use_conf_weights=False)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

    # ── Accumulate ──
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    pred_out_dir = os.path.join(OUTPUTS_DIR, f"predicted_{args.split}")

    from scipy.ndimage import label
    from configs.config import MIN_DEBRIS_PIXELS
    for batch in dl:
        imgs  = batch["image"].to(device)
        masks = batch["mask"]   # (B,H,W) cpu


        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B,2,H,W)
        debris_prob = probs[:,0,:,:]
        threshold = args.threshold if hasattr(args, 'threshold') else 0.7
        preds = (debris_prob > threshold).astype(np.uint8)
        masks_np = masks.numpy()
        preds = np.where(preds == 0, 0, 1)
        masks_np = np.where(masks_np == 0, 0, 1)

        # Post-processing: remove small debris blobs (configurable)
        if not args.no_postprocessing:
            min_pix = args.min_debris_pixels if hasattr(args, 'min_debris_pixels') else MIN_DEBRIS_PIXELS
            for i in range(preds.shape[0]):
                # preds: 1 == debris, 0 == not-debris
                debris_mask = (preds[i] == 1)
                labeled, n = label(debris_mask)
                for region in range(1, n+1):
                    if (labeled == region).sum() < min_pix:
                        preds[i][labeled == region] = 0  # set small regions to not-debris

        # Confusion matrix
        for p, t in zip(preds, masks_np):
            valid = t != -1
            t_v   = t[valid]
            p_v   = p[valid]
            np.add.at(cm, (t_v, p_v), 1)

        # Save masks
        if args.save_masks:
            for pid, p in zip(batch["id"], preds):
                save_predicted_mask(p, pid, pred_out_dir)

    # ── Per-class metrics ──
    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    tp   = np.diag(cm).astype(float)
    fp   = cm.sum(0) - tp
    fn   = cm.sum(1) - tp
    total = cm.sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-8)
    acc       = tp.sum() / total

    log.info("\n" + "-"*80)
    log.info(f"{'Class':<30} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Dice':>8}")
    log.info("-"*80)
    for i, name in enumerate(class_names):
        log.info(f"{name:<30} {iou[i]:8.4f} {f1[i]:8.4f} {precision[i]:8.4f} {recall[i]:8.4f} {dice[i]:8.4f}")
    log.info("-"*80)
    log.info(f"{'Mean (valid classes)':<30} {np.nanmean(iou):8.4f} {np.nanmean(f1):8.4f}")
    log.info(f"Overall pixel accuracy: {acc:.4f}")

    # ── Plots ──
    plot_confusion_matrix(cm, class_names,
                          os.path.join(OUTPUTS_DIR, f"confusion_matrix_{args.split}.png"))
    log.info("Evaluation complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--ckpt",       default=os.path.join(CHECKPOINTS_DIR, "resnext_cbam_best.pth"))
    p.add_argument("--save_masks", action="store_true", default=True)
    evaluate(p.parse_args())
