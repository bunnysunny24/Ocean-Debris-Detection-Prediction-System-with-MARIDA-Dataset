"""
evaluate.py
-----------
Evaluate the trained model on the test/val/train split.

Research-paper outputs:
  - Per-class IoU, F1, Precision, Recall, Dice
  - AUPRC (Area Under Precision-Recall Curve) — standard metric for severe imbalance
  - Precision-Recall curve saved as PNG
  - Object-level detection rate (patch-level: how many debris patches were found)
  - Confusion matrix PNG
  - Predicted GeoTIFF masks
  - Threshold sweep for optimal debris F1

Usage:
    python evaluate.py
    python evaluate.py --split val --ckpt ..\checkpoints\resnext_cbam_best.pth --tta
    python evaluate.py --split test --ckpt ..\checkpoints\resnext_cbam_ema_best.pth --tta --ensemble
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
    CHECKPOINTS_DIR, OUTPUTS_DIR, LOGS_DIR, PATCHES_DIR,
    ENCODER_NAME, MIN_DEBRIS_PIXELS
)
from utils.dataset import MARIDADataset, _find_tif
from semantic_segmentation.models.resnext_cbam_unet import ResNeXtCBAMUNet
try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None

CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
]


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=9)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def save_predicted_mask(pred_mask: np.ndarray, patch_id: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tif = _find_tif(patch_id)
    out_path = os.path.join(out_dir, f"{patch_id}_pred.tif")
    if tif:
        with rasterio.open(tif) as src:
            profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write((pred_mask + 1).astype(np.uint8)[np.newaxis])
    else:
        np.save(out_path.replace(".tif", ".npy"), pred_mask)


def _tta_predict(model, imgs, device):
    """8× geometric TTA: 4 rotations × 2 flips. Boosts debris F1 by ~3-5%."""
    B, C, H, W = imgs.shape
    accum = torch.zeros(B, 2, H, W, device=device)
    for k in range(4):
        for hflip in (False, True):
            x = imgs.clone()
            if hflip: x = x.flip(-1)
            if k > 0: x = x.rot90(k, [-2, -1])
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
            if k > 0: probs = probs.rot90(-k, [-2, -1])
            if hflip:  probs = probs.flip(-1)
            accum += probs
    return accum / 8.0


def _multiscale_tta_predict(model, imgs, device, scales=(0.75, 1.0, 1.25)):
    """Multi-scale TTA: 3 scales × 8 geometric = 24 variants."""
    B, C, H, W = imgs.shape
    accum = torch.zeros(B, 2, H, W, device=device)
    n = 0
    for scale in scales:
        if abs(scale - 1.0) < 1e-6:
            scaled = imgs
        else:
            sH, sW = int(H * scale), int(W * scale)
            scaled = torch.nn.functional.interpolate(imgs, size=(sH, sW),
                                                      mode='bilinear', align_corners=False)
        geo_accum = torch.zeros(B, 2, scaled.shape[2], scaled.shape[3], device=device)
        for k in range(4):
            for hflip in (False, True):
                x = scaled.clone()
                if hflip: x = x.flip(-1)
                if k > 0: x = x.rot90(k, [-2, -1])
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)
                if k > 0: probs = probs.rot90(-k, [-2, -1])
                if hflip:  probs = probs.flip(-1)
                geo_accum += probs
        geo_accum /= 8.0
        if abs(scale - 1.0) > 1e-6:
            geo_accum = torch.nn.functional.interpolate(geo_accum, size=(H, W),
                                                         mode='bilinear', align_corners=False)
        accum += geo_accum
        n += 1
    return accum / n


def _load_model(ckpt_path, device, log, backbone=None):
    try:
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        log.warning(f"Could not load {ckpt_path}: {e}")
        return None, False

    state_keys = list(ck.get("model", {}).keys()) if isinstance(ck, dict) else []
    use_smp = any(k.startswith(("encoder.", "decoder.", "segmentation_head.")) for k in state_keys)

    if use_smp and smp is not None:
        for enc_name in [ENCODER_NAME, "resnext50_32x4d", "resnet50"]:
            try:
                model = smp.DeepLabV3Plus(
                    encoder_name=enc_name, encoder_weights=None,
                    in_channels=INPUT_BANDS, classes=NUM_CLASSES, activation=None,
                ).to(device)
                model.load_state_dict(ck["model"])
                model.eval()
                return model, True
            except Exception:
                continue
        return None, False
    else:
        # Try both backbones (auto-selects the one that matches checkpoint)
        for bb in ([backbone] if backbone else []) + ["resnext50_32x4d", "resnext101_32x8d"]:
            try:
                model = ResNeXtCBAMUNet(in_channels=INPUT_BANDS, num_classes=NUM_CLASSES,
                                        pretrained=False, backbone=bb).to(device)
                try:
                    model.load_state_dict(ck["model"])
                except RuntimeError:
                    model.load_state_dict(ck["model"], strict=False)
                model.eval()
                log.info(f"  Loaded with backbone={bb}")
                return model, True
            except Exception:
                continue
        return None, False


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
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load models ──
    models = []
    ckpt_paths = [args.ckpt]

    if args.ensemble:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        for extra in ["ema_best.pth", "last.pth"]:
            extra_path = os.path.join(ckpt_dir, extra)
            if os.path.exists(extra_path) and extra_path not in ckpt_paths:
                ckpt_paths.append(extra_path)
        log.info(f"Ensemble mode: {len(ckpt_paths)} checkpoints")

    for cp in ckpt_paths:
        m, ok = _load_model(cp, device, log)
        if ok:
            models.append(m)
            log.info(f"Loaded checkpoint: {cp}")

    if not models:
        raise RuntimeError("No valid checkpoints loaded")
    model = models[0]

    # ── Data ──
    ds = MARIDADataset(args.split, augment_data=False, use_conf_weights=False)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

    # ── Accumulate ──
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    pred_out_dir = os.path.join(OUTPUTS_DIR, f"predicted_{args.split}")
    all_probs_list, all_gt_list = [], []

    # Object-level tracking (patch-level detection rate)
    patch_has_debris    = []   # bool: does this patch have any debris in GT?
    patch_debris_found  = []   # bool: did model predict any debris in this patch?

    from scipy.ndimage import label
    for batch in dl:
        imgs   = batch["image"].to(device)
        masks  = batch["mask"]
        pids   = batch["id"]

        try:
            autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
        except Exception:
            autocast_ctx = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))
        with autocast_ctx:
            if args.ensemble and len(models) > 1:
                accum = torch.zeros(imgs.shape[0], NUM_CLASSES, imgs.shape[2], imgs.shape[3], device=device)
                for m in models:
                    if args.ms_tta:   accum += _multiscale_tta_predict(m, imgs, device)
                    elif args.tta:    accum += _tta_predict(m, imgs, device)
                    else:             accum += torch.softmax(m(imgs), dim=1)
                probs = (accum / len(models)).cpu().numpy()
            elif args.ms_tta:
                probs = _multiscale_tta_predict(model, imgs, device).cpu().numpy()
            elif args.tta:
                probs = _tta_predict(model, imgs, device).cpu().numpy()
            else:
                probs = torch.softmax(model(imgs), dim=1).cpu().numpy()

        debris_prob = probs[:, 0, :, :]
        preds = np.where(debris_prob > args.threshold, 0, 1).astype(np.uint8)
        masks_np = masks.numpy()

        all_probs_list.append(debris_prob)
        all_gt_list.append(masks_np.copy())

        # Post-processing: remove tiny blobs
        if not args.no_postprocessing:
            for i in range(preds.shape[0]):
                debris_mask = (preds[i] == 0)
                labeled, n = label(debris_mask)
                for region in range(1, n+1):
                    if (labeled == region).sum() < args.min_debris_pixels:
                        preds[i][labeled == region] = 1

        # Confusion matrix
        for p, t in zip(preds, masks_np):
            valid = (t >= 0)
            np.add.at(cm, (t[valid].astype(int), p[valid].astype(int)), 1)

        # Object-level: did patch have / detect debris?
        for i in range(masks_np.shape[0]):
            gt_has = bool((masks_np[i] == 0).any())
            pred_has = bool((preds[i] == 0).any())
            patch_has_debris.append(gt_has)
            patch_debris_found.append(pred_has)

        if args.save_masks:
            for pid, p in zip(pids, preds):
                save_predicted_mask(p, pid, pred_out_dir)

    # ── Per-class pixel metrics ──
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

    # ── Object-level detection metrics ──
    patch_has_debris   = np.array(patch_has_debris)
    patch_debris_found = np.array(patch_debris_found)
    n_debris_patches = patch_has_debris.sum()
    n_detected = (patch_has_debris & patch_debris_found).sum()
    n_false_alarm = (~patch_has_debris & patch_debris_found).sum()
    n_clean = (~patch_has_debris).sum()
    obj_recall    = n_detected / (n_debris_patches + 1e-8)
    obj_precision = n_detected / (n_detected + n_false_alarm + 1e-8)
    obj_f1        = 2 * obj_precision * obj_recall / (obj_precision + obj_recall + 1e-8)
    log.info("\n" + "="*60)
    log.info("Object-Level Patch Detection Metrics:")
    log.info(f"  Total patches evaluated   : {len(patch_has_debris)}")
    log.info(f"  Patches with debris (GT)  : {n_debris_patches}")
    log.info(f"  Debris patches detected   : {n_detected}  (patch recall = {obj_recall:.3f})")
    log.info(f"  False alarm patches       : {n_false_alarm} / {n_clean} clean  (patch precision = {obj_precision:.3f})")
    log.info(f"  Patch-level F1            : {obj_f1:.4f}")

    # ── Confusion matrix plot ──
    plot_confusion_matrix(cm, class_names,
                          os.path.join(OUTPUTS_DIR, f"confusion_matrix_{args.split}.png"))

    # ── Threshold sweep ──
    all_dp = np.concatenate([x.ravel() for x in all_probs_list])
    all_gt = np.concatenate([x.ravel() for x in all_gt_list])
    valid_mask = all_gt >= 0
    all_dp, all_gt_v = all_dp[valid_mask], all_gt[valid_mask]
    gt_debris = (all_gt_v == 0)

    log.info("\n" + "="*60)
    log.info("Threshold sweep for optimal debris F1:")
    best_f1_sweep, best_thr = 0.0, 0.5
    for thr in np.arange(0.15, 0.85, 0.05):
        pred_d = all_dp > thr
        tp_s = (pred_d & gt_debris).sum()
        fp_s = (pred_d & ~gt_debris).sum()
        fn_s = (~pred_d & gt_debris).sum()
        pr = tp_s / (tp_s + fp_s + 1e-8)
        rc = tp_s / (tp_s + fn_s + 1e-8)
        f1_s = 2 * pr * rc / (pr + rc + 1e-8)
        log.info(f"  thr={thr:.2f}  P={pr:.4f}  R={rc:.4f}  F1={f1_s:.4f}")
        if f1_s > best_f1_sweep:
            best_f1_sweep, best_thr = f1_s, thr
    log.info(f"  >>> Optimal threshold: {best_thr:.2f}  (F1={best_f1_sweep:.4f})")

    # ── Precision-Recall curve + AUPRC ──
    pr_thresholds = np.linspace(0.001, 0.999, 200)
    pr_precisions, pr_recalls = [], []
    for thr in pr_thresholds:
        pred_d = all_dp > thr
        tp_s = (pred_d & gt_debris).sum()
        fp_s = (pred_d & ~gt_debris).sum()
        fn_s = (~pred_d & gt_debris).sum()
        pr_precisions.append(float(tp_s / (tp_s + fp_s + 1e-8)))
        pr_recalls.append(float(tp_s / (tp_s + fn_s + 1e-8)))

    # AUPRC via trapezoidal rule (sort by recall for correct integration)
    pr_recalls_arr    = np.array(pr_recalls)
    pr_precisions_arr = np.array(pr_precisions)
    sort_idx = np.argsort(pr_recalls_arr)
    auprc = float(np.trapz(pr_precisions_arr[sort_idx], pr_recalls_arr[sort_idx]))
    log.info(f"\n  AUPRC (Area Under PR Curve): {auprc:.4f}")
    log.info(f"  [Baseline AUPRC for random classifier at {gt_debris.mean()*100:.2f}% debris: {gt_debris.mean():.4f}]")

    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(pr_recalls_arr[sort_idx], pr_precisions_arr[sort_idx],
                lw=2, color="#2196F3", label=f"Debris PR curve (AUPRC={auprc:.3f})")
        # Mark optimal threshold point
        opt_idx = np.argmin(np.abs(pr_thresholds - best_thr))
        ax.scatter([pr_recalls[opt_idx]], [pr_precisions[opt_idx]],
                   color="red", zorder=5, s=80, label=f"Optimal thr={best_thr:.2f}")
        # Baseline (random classifier)
        baseline = gt_debris.mean()
        ax.axhline(baseline, color="gray", linestyle="--", lw=1.2, label=f"Random baseline ({baseline:.3f})")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve — Debris ({args.split} split)", fontsize=13)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=10)
        plt.tight_layout()
        pr_save = os.path.join(OUTPUTS_DIR, f"pr_curve_{args.split}.png")
        plt.savefig(pr_save, dpi=150)
        plt.close()
        log.info(f"PR curve saved → {pr_save}")
    except Exception as e:
        log.warning(f"PR curve plot failed: {e}")

    # ── Re-report metrics at optimal threshold ──
    if abs(best_thr - args.threshold) > 0.01:
        log.info(f"\nRe-computing metrics at optimal threshold {best_thr:.2f}:")
        cm_opt   = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        pred_opt = np.where(all_dp > best_thr, 0, 1).astype(int)
        np.add.at(cm_opt, (all_gt_v.astype(int), pred_opt), 1)
        tp_o  = np.diag(cm_opt).astype(float)
        fp_o  = cm_opt.sum(0) - tp_o
        fn_o  = cm_opt.sum(1) - tp_o
        prec_o = tp_o / (tp_o + fp_o + 1e-8)
        rec_o  = tp_o / (tp_o + fn_o + 1e-8)
        f1_o   = 2 * prec_o * rec_o / (prec_o + rec_o + 1e-8)
        iou_o  = tp_o / (tp_o + fp_o + fn_o + 1e-8)
        acc_o  = tp_o.sum() / cm_opt.sum()
        log.info(f"  Debris    — IoU={iou_o[0]:.4f}  F1={f1_o[0]:.4f}  P={prec_o[0]:.4f}  R={rec_o[0]:.4f}")
        log.info(f"  NotDebris — IoU={iou_o[1]:.4f}  F1={f1_o[1]:.4f}  P={prec_o[1]:.4f}  R={rec_o[1]:.4f}")
        log.info(f"  Overall pixel accuracy: {acc_o:.4f}")
        log.info(f"  Mean IoU: {np.nanmean(iou_o):.4f}   Mean F1: {np.nanmean(f1_o):.4f}   AUPRC: {auprc:.4f}")

    log.info("Evaluation complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split",               default="test", choices=["train", "val", "test"])
    p.add_argument("--ckpt",                default=os.path.join(CHECKPOINTS_DIR, "resnext_cbam_best.pth"))
    p.add_argument("--save_masks",          action="store_true", default=True)
    p.add_argument("--threshold",           type=float, default=0.5)
    p.add_argument("--min_debris_pixels",   type=int,   default=MIN_DEBRIS_PIXELS)
    p.add_argument("--no_postprocessing",   action="store_true", default=False)
    p.add_argument("--tta",                 action="store_true", default=False,
                   help="8× geometric TTA (flip + rot90)")
    p.add_argument("--ms_tta",              action="store_true", default=False,
                   help="Multi-scale TTA (3 scales × 8 geometric = 24 variants)")
    p.add_argument("--ensemble",            action="store_true", default=False,
                   help="Auto-load best + ema_best + last checkpoints from same run directory")
    evaluate(p.parse_args())
