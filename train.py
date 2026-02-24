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
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    CHECKPOINTS_DIR, LOGS_DIR,
)
from utils.dataset import MARIDADataset
from semantic_segmentation.models.resnext_cbam_unet import ResNeXtCBAMUNet, HybridLoss


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
    for batch in tqdm(loader, desc="[train]"):
        imgs   = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        confs  = batch["conf"].to(device)

        # Debug: count debris pixels (class 0)
        debris_pixels = (masks == 0).sum().item()
        if debris_pixels == 0:
            print("[WARN] No debris pixels in this batch!")

        optimizer.zero_grad()
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

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds, all_masks = [], []
    for batch in tqdm(loader, desc="[val]"):
        imgs   = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        confs  = batch["conf"].to(device)

        # Debug: count debris pixels (class 0)
        debris_pixels = (masks == 0).sum().item()
        if debris_pixels == 0:
            print("[WARN] No debris pixels in this batch!")

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
        all_preds.append(preds.cpu())
        all_masks.append(masks.cpu())

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Datasets
    train_ds = MARIDADataset("train", augment_data=True)
    val_ds   = MARIDADataset("val",   augment_data=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # Model
    model = ResNeXtCBAMUNet(in_channels=INPUT_BANDS, num_classes=NUM_CLASSES,
                             pretrained=True).to(device)
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    criterion = HybridLoss(class_weights=CLASS_WEIGHTS, dice_weight=0.4)

    # Optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
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
    ckpt_path    = os.path.join(CHECKPOINTS_DIR, "resnext_cbam_best.pth")

    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_iou    = ck["best_iou"]
        log.info(f"Resumed from epoch {start_epoch}, best mIoU={best_iou:.4f}")

    writer = SummaryWriter(log_dir=os.path.join(LOGS_DIR, "tsboard"))

    log.info("─" * 60)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_dl, optimizer, criterion, device, scaler)
        val_loss, mean_iou, per_class_iou = val_epoch(model, val_dl, criterion, device, NUM_CLASSES)

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch+1:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"mIoU={mean_iou:.4f}  ({elapsed:.1f}s)"
        )
        # Log per-class IoU and detection counts
        class_names = ["Debris", "Dense Sargassum", "Sparse Sargassum", "Natural Organic Material", "Ship", "Clouds", "Marine Water", "Sediment-Laden Water", "Foam", "Turbid Water", "Shallow Water", "Waves", "Cloud Shadows", "Wakes", "Mixed Water"]
        for i, (iou, cname) in enumerate(zip(per_class_iou, class_names)):
            log.info(f"Class {i}: {cname:25s} IoU={iou:.4f}")
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
                for i in range(len(class_names)):
                    class_counts[i] += np.sum(mask == i)
        log.info("Validation set class pixel counts:")
        for i, cname in enumerate(class_names):
            log.info(f"  {cname:25s}: {class_counts[i]}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("mIoU/val",   mean_iou,   epoch)

        scheduler.step(mean_iou)

        # Checkpoint
        ck = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "best_iou": best_iou,
        }
        torch.save(ck, os.path.join(CHECKPOINTS_DIR, "resnext_cbam_last.pth"))

        if mean_iou > best_iou:
            best_iou   = mean_iou
            no_improve = 0
            torch.save(ck, ckpt_path)
            log.info(f"  ✓ New best mIoU: {best_iou:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log.info(f"Early stopping after {PATIENCE} epochs without improvement.")
                break

    log.info(f"Training complete. Best val mIoU: {best_iou:.4f}")
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int,   default=EPOCHS)
    p.add_argument("--batch",   type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",      type=float, default=LR)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--resume",  action="store_true")
    main(p.parse_args())
