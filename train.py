"""
train.py
--------
End-to-end training for the ResNeXt-50 + CBAM U-Net on MARIDA.

Usage:
    python train.py
    python train.py --epochs 80 --batch 4 --lr 5e-5
"""

import os, sys, argparse, time, logging
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    INPUT_BANDS, NUM_CLASSES, CLASS_WEIGHTS,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE, NUM_WORKERS,
    CHECKPOINTS_DIR, LOGS_DIR,
)
from utils.dataset import MARIDADataset
import segmentation_models_pytorch as smp
from semantic_segmentation.models.resnext_cbam_unet import HybridLoss


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou(pred, target, num_classes, ignore_index=-1):
    ious = []
    pred   = pred.view(-1)
    target = target.view(-1)
    valid  = target != ignore_index
    pred, target = pred[valid], target[valid]
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().float()
        fp = ((pred == c) & (target != c)).sum().float()
        fn = ((pred != c) & (target == c)).sum().float()
        denom = tp + fp + fn
        ious.append((tp / (denom + 1e-8)).item() if denom > 0 else float("nan"))
    return ious


# ── Train / Val loops ─────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    import time
    try:
        import psutil, os
        psutil_available = True
    except ImportError:
        psutil_available = False
    for batch in tqdm(loader, desc="[train]"):
        batch_start = time.time()
        imgs   = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        confs  = batch["conf"].to(device)

        # Debug: tensor shapes
        print(f"[DEBUG] imgs shape: {imgs.shape}, masks shape: {masks.shape}, confs shape: {confs.shape}")

        # Debug: count debris pixels (class 0)
        debris_pixels = (masks == 0).sum().item()
        not_debris_pixels = (masks == 1).sum().item()
        print(f"[TRAIN DEBUG] Batch mask counts: debris={debris_pixels}, not_debris={not_debris_pixels}")
        if debris_pixels == 0:
            print("[WARN] No debris pixels in this batch!")

        # Debug: print current learning rate
        for param_group in optimizer.param_groups:
            print(f"[DEBUG] Current learning rate: {param_group['lr']}")

        optimizer.zero_grad()
        try:
            autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
        except Exception:
            autocast_ctx = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))
        with autocast_ctx:
            logits = model(imgs)
            loss   = criterion(logits, masks, confs)

        preds = logits.argmax(dim=1)
        unique_preds, counts = torch.unique(preds.cpu(), return_counts=True)
        print(f"[TRAIN DEBUG] Unique predicted classes: {unique_preds.tolist()} counts: {counts.tolist()}")

        if torch.isnan(loss):
            print("[ERROR] NaN loss encountered! Skipping batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        # Debug: batch timing
        print(f"[DEBUG] Batch time: {time.time() - batch_start:.2f}s")

        # Debug: memory usage
        if psutil_available:
            print(f"[DEBUG] RAM usage: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds, all_masks = [], []
    for batch in tqdm(loader, desc="[train]"):
        imgs   = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        confs  = batch["conf"].to(device)

        # Count debris pixels (class 0)
        debris_pixels = (masks == 0).sum().item()
        if debris_pixels == 0:
            print("[WARN] No debris pixels in this batch! Skipping batch.")
            continue

        try:
            autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
        except Exception:
            autocast_ctx = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))
        with autocast_ctx:
            logits = model(imgs)
            loss   = criterion(logits, masks, confs)

        if torch.isnan(loss):
            print("[ERROR] NaN loss encountered! Skipping batch.")
            continue

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        # Debug: print unique predicted classes and counts for this batch
        unique_preds, counts = torch.unique(preds.cpu(), return_counts=True)
        print(f"[VAL DEBUG] Unique predicted classes: {unique_preds.tolist()} counts: {counts.tolist()}")
        all_preds.append(preds.cpu())
        all_masks.append(masks.cpu())

    if len(all_preds) == 0:
        return float('nan'), float('nan'), [float('nan')]*num_classes
    preds_cat = torch.cat([p.view(-1) for p in all_preds])
    masks_cat = torch.cat([m.view(-1) for m in all_masks])
    ious = compute_iou(preds_cat, masks_cat, num_classes)
    mean_iou = np.nanmean(ious)
    return total_loss / len(loader), mean_iou, ious


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # Logging
    import datetime
    log_dir = os.path.join(LOGS_DIR, "external_logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{timestamp}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    try:
        stream_handler.stream.reconfigure(encoding='utf-8')
    except Exception:
        pass  # For Python <3.7 or if reconfigure is unavailable
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[file_handler, stream_handler],
    )
    log = logging.getLogger()
    log.info(f"Logging to {log_path}")

    # Create a timestamped checkpoint directory for this run
    ckpt_dir = os.path.join(CHECKPOINTS_DIR, f"run_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Prepare CSV for metrics logging
    import csv
    metrics_csv_path = os.path.join(ckpt_dir, "metrics.csv")
    metrics_csv_file = open(metrics_csv_path, mode="w", newline="", encoding="utf-8")
    metrics_writer = csv.writer(metrics_csv_file)
    metrics_writer.writerow([
        "epoch", "train_loss", "val_loss", "mIoU",
        "iou_debris", "iou_not_debris",
        "precision", "recall", "f1"
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Datasets
    train_ds = MARIDADataset("train", augment_data=True)
    val_ds   = MARIDADataset("val",   augment_data=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # Model: DeepLabV3+ with configurable encoder
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",  # or try "timm-resnest50d", "resnext50_32x4d"
        encoder_weights="imagenet",
        in_channels=INPUT_BANDS,
        classes=NUM_CLASSES,
        activation=None,
    ).to(device)
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss: Strong rare-class weighting, high dice, focal enabled
    focal = True
    focal_gamma = 2.0
    # Stronger rare-class weighting (debris is rare)
    class_weights = np.array([3.0, 1.0])
    criterion = HybridLoss(class_weights=class_weights, dice_weight=0.7, use_focal=focal, focal_gamma=focal_gamma)

    # Optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                    factor=0.5, patience=5)
    try:
        scaler = torch.amp.GradScaler(device if device.type == "cuda" else None, enabled=(device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Resume?
    start_epoch  = 0
    best_iou     = 0.0
    no_improve   = 0
    ckpt_path    = os.path.join(ckpt_dir, "best.pth")

    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_iou    = ck["best_iou"]
        log.info(f"Resumed from epoch {start_epoch}, best mIoU={best_iou:.4f}")

    writer = SummaryWriter(log_dir=os.path.join(LOGS_DIR, "tsboard"))

    log.info("─" * 60)
    patience = args.patience if hasattr(args, 'patience') else PATIENCE
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_dl, optimizer, criterion, device, scaler)
        val_loss, mean_iou, per_class_iou = val_epoch(model, val_dl, criterion, device, NUM_CLASSES)

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"mIoU={mean_iou:.4f}  ({elapsed:.1f}s)"
        )
        print(f"[CHECKPOINT] Saving model for epoch {epoch+1} to {ckpt_dir}")
        # Save checkpoint for every epoch
        epoch_ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "best_iou": best_iou,
        }
        torch.save(epoch_ckpt, os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pth"))
        # Log per-class IoU and detection counts (binary)
        class_names = ["Debris", "Not Debris"]
        for i, (iou, cname) in enumerate(zip(per_class_iou, class_names)):
            log.info(f"Class {i}: {cname:15s} IoU={iou:.4f}")
        # Count detections in val set
        val_ds = val_dl.dataset
        class_counts = {i: 0 for i in range(len(class_names))}
        for pid in getattr(val_ds, 'patch_ids', []):
            mask_tif = None
            try:
                from utils.dataset import _find_mask_tif
                mask_tif = _find_mask_tif(pid)
            except Exception:
                pass
            if mask_tif:
                import rasterio
                with rasterio.open(mask_tif) as src:
                    mask = src.read(1).astype(int) - 1
                # Binary: debris=0, not-debris=1, nodata=-1
                debris_mask = (mask == 0)
                not_debris_mask = (mask >= 1) & (mask <= 14)
                mask_bin = np.full_like(mask, fill_value=1, dtype=np.int64)
                mask_bin[debris_mask] = 0
                mask_bin[mask == -1] = -1
                for i in range(len(class_names)):
                    class_counts[i] += np.sum(mask_bin == i)
        log.info("Validation set class pixel counts:")
        for i, cname in enumerate(class_names):
            log.info(f"  {cname:15s}: {class_counts[i]}")

        # Calculate debris precision/recall/F1 on validation predictions
        debris_precision = debris_recall = debris_f1 = float('nan')
        if len(per_class_iou) > 0 and hasattr(val_dl, 'dataset'):
            all_preds, all_masks = [], []
            for batch in val_dl:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                debris_pixels = (masks == 0).sum().item()
                if debris_pixels == 0:
                    continue
                with torch.no_grad():
                    logits = model(imgs)
                    preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())
            if len(all_preds) > 0:
                preds_cat = torch.cat([p.view(-1) for p in all_preds])
                masks_cat = torch.cat([m.view(-1) for m in all_masks])
                debris_tp = ((preds_cat == 0) & (masks_cat == 0)).sum().item()
                debris_fp = ((preds_cat == 0) & (masks_cat == 1)).sum().item()
                debris_fn = ((preds_cat == 1) & (masks_cat == 0)).sum().item()
                debris_precision = debris_tp / (debris_tp + debris_fp + 1e-8)
                debris_recall = debris_tp / (debris_tp + debris_fn + 1e-8)
                debris_f1 = 2 * debris_precision * debris_recall / (debris_precision + debris_recall + 1e-8)
                log.info(f"Debris precision: {debris_precision:.4f}  recall: {debris_recall:.4f}  F1: {debris_f1:.4f}")

        # Save metrics to CSV
        metrics_writer.writerow([
            epoch+1, train_loss, val_loss, mean_iou,
            per_class_iou[0] if len(per_class_iou) > 0 else float('nan'),
            per_class_iou[1] if len(per_class_iou) > 1 else float('nan'),
            debris_precision, debris_recall, debris_f1
        ])
        metrics_csv_file.flush()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("mIoU/val",   mean_iou,   epoch)

        scheduler.step(mean_iou)

        # Checkpoint
        # Save last checkpoint (for resuming)
        ck = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "best_iou": best_iou,
        }
        torch.save(ck, os.path.join(ckpt_dir, "last.pth"))

        if mean_iou > best_iou:
            best_iou   = mean_iou
            no_improve = 0
            # Save best model inside this run folder
            torch.save(ck, ckpt_path)
            # Also copy the best model to a canonical path for easy access
            shared_best = os.path.join(CHECKPOINTS_DIR, "resnext_cbam_best.pth")
            try:
                import shutil
                shutil.copy2(ckpt_path, shared_best)
                log.info(f"  ✓ New best mIoU: {best_iou:.4f} (epoch {epoch+1}) — saved to {ckpt_path} and copied to {shared_best}")
            except Exception as e:
                # Don't fail training if the copy fails; log a warning instead
                log.warning(f"  ✓ New best mIoU: {best_iou:.4f} (epoch {epoch+1}) — saved to {ckpt_path}, but failed to copy to {shared_best}: {e}")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping after {patience} epochs without improvement.")
                break

    log.info(f"Training complete. Best val mIoU: {best_iou:.4f}")
    metrics_csv_file.close()
    writer.close()

    # Plot and save graphs for all metrics
    import pandas as pd
    import matplotlib.pyplot as plt
    metrics_df = pd.read_csv(metrics_csv_path)
    metric_names = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Validation Loss"),
        ("mIoU", "Mean IoU"),
        ("iou_debris", "IoU: Debris"),
        ("iou_not_debris", "IoU: Not Debris"),
        ("precision", "Debris Precision"),
        ("recall", "Debris Recall"),
        ("f1", "Debris F1")
    ]
    for col, title in metric_names:
        plt.figure()
        plt.plot(metrics_df["epoch"], metrics_df[col], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title + " vs. Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, f"{col}.png"))
        plt.close()
    log.info(f"Saved metric graphs to {ckpt_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int,   default=EPOCHS)
    p.add_argument("--batch",   type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",      type=float, default=LR)
    p.add_argument("--workers", type=int,   default=NUM_WORKERS)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--resume",  action="store_true")
    main(p.parse_args())
