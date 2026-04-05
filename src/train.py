"""
train.py
--------
End-to-end training for the ResNeXt-50 + CBAM U-Net on MARIDA.

Research-paper improvements over original:
  - ModelEMA (Exponential Moving Average) for smoother generalisation
  - CosineAnnealingWarmRestarts (SGDR) to escape instability valleys
  - Encoder freeze for first ENCODER_FREEZE_EPOCHS (transfer learning warmup)
  - Deep supervision support (aux logits from model during training)
  - EMA checkpoint saved as ema_best.pth alongside best.pth

Usage:
    python train.py
    python train.py --epochs 100 --batch 8 --workers 4 --grad_accum 4 --patience 999
"""

import os, sys, argparse, time, logging, copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    INPUT_BANDS, NUM_CLASSES, CLASS_WEIGHTS,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE, NUM_WORKERS,
    CHECKPOINTS_DIR, LOGS_DIR,
    GRAD_ACCUM, LABEL_SMOOTH, FOCAL_GAMMA,
    EMA_DECAY, ENCODER_FREEZE_EPOCHS, ENCODER_NAME,
    TVERSKY_ALPHA, TVERSKY_BETA, TVERSKY_WEIGHT, AUX_LOSS_WEIGHT,
)
from utils.dataset import MARIDADataset
from semantic_segmentation.models.resnext_cbam_unet import ResNeXtCBAMUNet, HybridLoss


# ── EMA Helper ────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy of the model with smoothed weights — consistently
    outperforms raw checkpoint at test time, especially under noisy CPU training.

    Usage:
        ema = ModelEMA(model, decay=0.9998)
        # after each optimiser step:
        ema.update(model)
        # evaluate with:
        ema.apply_shadow()  → model now has EMA weights
        ema.restore()       → model restored to training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        # shadow is a deepcopy kept on CPU to avoid OOM on small GPU
        self.decay  = decay
        self.shadow = copy.deepcopy(model).cpu()
        self.shadow.eval()
        # disable gradients for shadow
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for (name, s_param), (_, m_param) in zip(
            self.shadow.named_parameters(), model.named_parameters()
        ):
            s_param.data.mul_(self.decay).add_(m_param.detach().cpu().data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def apply_shadow(self, model: nn.Module):
        """Copy EMA weights into model for evaluation."""
        self._backup = copy.deepcopy(model.state_dict())
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in self.shadow.state_dict().items()})

    def restore(self, model: nn.Module):
        """Restore original (training) weights into model."""
        model.load_state_dict(self._backup)




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

def train_epoch(model, loader, optimizer, criterion, device, scaler, grad_accum=1):
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="[train]")):
        imgs   = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        confs  = batch["conf"].to(device)

        try:
            autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
        except Exception:
            autocast_ctx = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))
        with autocast_ctx:
            # model returns (main_logits, aux_logits) when training + deep_supervision=True
            out  = model(imgs)
            loss = criterion(out, masks, confs) / grad_accum

        if torch.isnan(loss):
            continue

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def val_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_masks = [], []
    for batch in tqdm(loader, desc="[val]"):
        imgs   = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        confs  = batch["conf"].to(device)

        try:
            autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
        except Exception:
            autocast_ctx = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))
        with autocast_ctx:
            logits = model(imgs)   # eval mode → always returns single tensor
            loss   = criterion(logits, masks, confs)

        if torch.isnan(loss):
            continue

        total_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_masks.append(masks.cpu())

    if len(all_preds) == 0:
        return float('nan'), float('nan'), [float('nan')]*num_classes, float('nan'), float('nan'), float('nan')
    preds_cat = torch.cat([p.view(-1) for p in all_preds])
    masks_cat = torch.cat([m.view(-1) for m in all_masks])
    ious = compute_iou(preds_cat, masks_cat, num_classes)
    mean_iou = np.nanmean(ious)

    # Debris precision / recall / F1 (class 0 = debris)
    valid = masks_cat != -1
    p_v = preds_cat[valid]
    m_v = masks_cat[valid]
    debris_tp = ((p_v == 0) & (m_v == 0)).sum().item()
    debris_fp = ((p_v == 0) & (m_v != 0)).sum().item()
    debris_fn = ((p_v != 0) & (m_v == 0)).sum().item()
    d_prec = debris_tp / (debris_tp + debris_fp + 1e-8)
    d_rec  = debris_tp / (debris_tp + debris_fn + 1e-8)
    d_f1   = 2 * d_prec * d_rec / (d_prec + d_rec + 1e-8)

    return total_loss / max(n_batches, 1), mean_iou, ious, d_prec, d_rec, d_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # Maximize CPU usage
    torch.set_num_threads(8)

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
        pass
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
        "precision", "recall", "f1",
        "ema_f1", "lr"
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
        log.info(f"Backbone (auto-selected for GPU): {args.backbone}")
    else:
        log.info(f"Backbone (auto-selected for CPU): {args.backbone}")
        log.info("TIP: Run on GPU tomorrow for 10-20x speedup and resnext101 backbone.")

    # Datasets
    train_ds = MARIDADataset("train", augment_data=True)
    val_ds   = MARIDADataset("val",   augment_data=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # Model: ResNeXt-50 + CBAM U-Net + deep supervision
    model = ResNeXtCBAMUNet(
        in_channels=INPUT_BANDS,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=0.1,
        deep_supervision=True,
        backbone=args.backbone,
    ).to(device)
    log.info(f"Architecture: ResNeXtCBAMUNet | backbone={args.backbone} | in_ch={INPUT_BANDS} | classes={NUM_CLASSES} | deep_sup=True | dropout=0.1")
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── EMA shadow model ──
    ema = ModelEMA(model, decay=EMA_DECAY)
    log.info(f"EMA decay: {EMA_DECAY}")

    # ── Encoder freeze warmup ──
    freeze_epochs = args.encoder_freeze if hasattr(args, 'encoder_freeze') else ENCODER_FREEZE_EPOCHS
    if freeze_epochs > 0:
        for name, param in model.named_parameters():
            if any(name.startswith(enc) for enc in ["enc0", "enc1", "enc2", "enc3", "enc4", "pool"]):
                param.requires_grad_(False)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        log.info(f"Encoder frozen for first {freeze_epochs} epochs ({frozen_params:,} params frozen)")

    # Loss: Focal-CE + Tversky (recall-first rare-class learning)
    class_weights = np.array(CLASS_WEIGHTS, dtype=np.float32)
    criterion = HybridLoss(
        class_weights=class_weights,
        tversky_weight=TVERSKY_WEIGHT,
        use_focal=True,
        focal_gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTH,
        tversky_alpha=TVERSKY_ALPHA,
        tversky_beta=TVERSKY_BETA,
        aux_weight=AUX_LOSS_WEIGHT,
    )
    log.info(f"Loss: Focal-CE(γ={FOCAL_GAMMA}) + Tversky(α={TVERSKY_ALPHA},β={TVERSKY_BETA},w={TVERSKY_WEIGHT}) + AuxHead(w={AUX_LOSS_WEIGHT})")
    log.info(f"Class weights: debris={CLASS_WEIGHTS[0]}, not-debris={CLASS_WEIGHTS[1]}")

    # Gradient accumulation
    grad_accum = args.grad_accum if hasattr(args, 'grad_accum') and args.grad_accum else GRAD_ACCUM
    log.info(f"Gradient accumulation: {grad_accum} steps (effective batch = {args.batch * grad_accum})")

    # Optimiser: only optimise unfrozen parameters initially
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler: Linear warmup → SGDR (CosineAnnealingWarmRestarts)
    # Warm restarts escape instability valleys by periodically resetting LR
    warmup_sched  = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    sgdr_sched    = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, sgdr_sched], milestones=[5])
    log.info("Scheduler: LinearWarmup(5ep) → SGDR(T0=25, Tmult=2, eta_min=1e-6)")

    try:
        scaler = torch.amp.GradScaler(device if device.type == "cuda" else None, enabled=(device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Resume?
    start_epoch = 0
    best_f1     = 0.0
    best_ema_f1 = 0.0
    no_improve  = 0
    ckpt_path   = os.path.join(ckpt_dir, "best.pth")
    ema_ckpt_path = os.path.join(ckpt_dir, "ema_best.pth")

    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_f1     = ck.get("best_f1", 0.0)
        log.info(f"Resumed from epoch {start_epoch}, best F1={best_f1:.4f}")

    writer = SummaryWriter(log_dir=os.path.join(LOGS_DIR, "tsboard"))

    log.info("─" * 60)
    patience = args.patience if hasattr(args, 'patience') else PATIENCE

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # ── Unfreeze encoder after warmup ──
        if epoch == freeze_epochs and freeze_epochs > 0:
            for param in model.parameters():
                param.requires_grad_(True)
            # Re-create optimizer with all parameters
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1,
                                    weight_decay=args.weight_decay)
            # ── CRITICAL: reconnect scheduler to new optimizer ──
            # Without this, scheduler.step() updates the OLD optimizer and the
            # new one is stuck at a fixed LR forever — SGDR restarts never fire.
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=25, T_mult=2, eta_min=1e-6
            )
            log.info(f"  ► Encoder unfrozen at epoch {epoch+1}, lr reset to {args.lr * 0.1:.2e}")
            log.info(f"  ► SGDR scheduler reconnected to new optimizer (T0=25, Tmult=2)")

        train_loss = train_epoch(model, train_dl, optimizer, criterion, device, scaler, grad_accum)

        # ── Update EMA after each training epoch ──
        ema.update(model)

        val_loss, mean_iou, per_class_iou, debris_precision, debris_recall, debris_f1 = \
            val_epoch(model, val_dl, criterion, device, NUM_CLASSES)

        # ── Validate EMA model ──
        ema.apply_shadow(model)
        _, _, _, ema_prec, ema_rec, ema_f1 = val_epoch(model, val_dl, criterion, device, NUM_CLASSES)
        ema.restore(model)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        log.info(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"mIoU={mean_iou:.4f}  lr={current_lr:.2e}  ({elapsed:.1f}s)"
        )

        # Save checkpoint for every epoch
        epoch_ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "best_f1": best_f1,
        }
        torch.save(epoch_ckpt, os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pth"))

        # Log per-class IoU
        class_names = ["Debris", "Not Debris"]
        for i, (iou, cname) in enumerate(zip(per_class_iou, class_names)):
            log.info(f"  Class {i}: {cname:15s} IoU={iou:.4f}")
        log.info(f"  Debris  P={debris_precision:.4f}  R={debris_recall:.4f}  F1={debris_f1:.4f}")
        log.info(f"  [EMA]   P={ema_prec:.4f}        R={ema_rec:.4f}        F1={ema_f1:.4f}")

        # Warn if collapsing to majority class
        if epoch > 5 and debris_recall < 0.01:
            log.warning(f"⚠ DEBRIS RECALL = {debris_recall:.4f} — model may be collapsing to majority class!")

        # Save metrics to CSV
        metrics_writer.writerow([
            epoch+1, train_loss, val_loss, mean_iou,
            per_class_iou[0] if len(per_class_iou) > 0 else float('nan'),
            per_class_iou[1] if len(per_class_iou) > 1 else float('nan'),
            debris_precision, debris_recall, debris_f1,
            ema_f1, current_lr
        ])
        metrics_csv_file.flush()

        # TensorBoard
        writer.add_scalar("Loss/train",       train_loss,       epoch)
        writer.add_scalar("Loss/val",         val_loss,         epoch)
        writer.add_scalar("mIoU/val",         mean_iou,         epoch)
        writer.add_scalar("Debris/F1",        debris_f1,        epoch)
        writer.add_scalar("Debris/F1_EMA",    ema_f1,           epoch)
        writer.add_scalar("Debris/Precision", debris_precision,  epoch)
        writer.add_scalar("Debris/Recall",    debris_recall,     epoch)
        writer.add_scalar("Debris/IoU",       per_class_iou[0] if len(per_class_iou) > 0 else 0.0, epoch)
        writer.add_scalar("LR",               current_lr,        epoch)

        scheduler.step()

        # ── Save last checkpoint ──
        ck = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "best_f1": best_f1,
        }
        torch.save(ck, os.path.join(ckpt_dir, "last.pth"))

        # ── Best model selection by debris F1 with minimum recall guard ──
        current_score = debris_f1 if debris_recall >= 0.05 else 0.0

        if current_score > best_f1:
            best_f1    = debris_f1
            no_improve = 0
            best_ck = {
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "best_f1": best_f1,
            }
            torch.save(best_ck, ckpt_path)
            shared_best = os.path.join(CHECKPOINTS_DIR, "resnext_cbam_best.pth")
            try:
                import shutil
                shutil.copy2(ckpt_path, shared_best)
                log.info(f"  ✓ New best F1: {best_f1:.4f} (epoch {epoch+1}) → saved + copied to {shared_best}")
            except Exception as e:
                log.warning(f"  ✓ New best F1: {best_f1:.4f} (epoch {epoch+1}) → saved (copy failed: {e})")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping after {patience} epochs without improvement.")
                break

        # ── Best EMA checkpoint ──
        ema_score = ema_f1 if ema_rec >= 0.05 else 0.0
        if ema_score > best_ema_f1:
            best_ema_f1 = ema_f1
            ema_ck = {
                "epoch": epoch, "model": ema.state_dict(), "best_f1": best_ema_f1,
            }
            torch.save(ema_ck, ema_ckpt_path)
            shared_ema = os.path.join(CHECKPOINTS_DIR, "resnext_cbam_ema_best.pth")
            try:
                import shutil
                shutil.copy2(ema_ckpt_path, shared_ema)
                log.info(f"  ✓ New best EMA F1: {best_ema_f1:.4f} (epoch {epoch+1}) → saved to {shared_ema}")
            except Exception:
                pass

    log.info(f"Training complete. Best debris F1: {best_f1:.4f}  |  Best EMA F1: {best_ema_f1:.4f}")
    metrics_csv_file.close()
    writer.close()

    # Plot and save graphs for all metrics
    import pandas as pd
    import matplotlib.pyplot as plt
    metrics_df = pd.read_csv(metrics_csv_path)
    metric_names = [
        ("train_loss",    "Train Loss"),
        ("val_loss",      "Validation Loss"),
        ("mIoU",          "Mean IoU"),
        ("iou_debris",    "IoU: Debris"),
        ("iou_not_debris","IoU: Not Debris"),
        ("precision",     "Debris Precision"),
        ("recall",        "Debris Recall"),
        ("f1",            "Debris F1"),
        ("ema_f1",        "Debris F1 (EMA)"),
        ("lr",            "Learning Rate"),
    ]
    for col, title in metric_names:
        if col not in metrics_df.columns:
            continue
        plt.figure()
        plt.plot(metrics_df["epoch"], metrics_df[col], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title + " vs. Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, f"{col}.png"))
        plt.close()

    # F1 comparison: raw vs EMA on same plot
    if "ema_f1" in metrics_df.columns and "f1" in metrics_df.columns:
        plt.figure()
        plt.plot(metrics_df["epoch"], metrics_df["f1"],     marker="o", label="Raw F1")
        plt.plot(metrics_df["epoch"], metrics_df["ema_f1"], marker="s", label="EMA F1", linestyle="--")
        plt.xlabel("Epoch"); plt.ylabel("Debris F1")
        plt.title("Debris F1: Raw vs EMA")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "f1_raw_vs_ema.png"))
        plt.close()

    log.info(f"Saved metric graphs to {ckpt_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",         type=int,   default=EPOCHS)
    p.add_argument("--batch",          type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--workers",        type=int,   default=NUM_WORKERS)
    p.add_argument("--weight_decay",   type=float, default=WEIGHT_DECAY)
    p.add_argument("--patience",       type=int,   default=PATIENCE)
    p.add_argument("--grad_accum",     type=int,   default=GRAD_ACCUM,
                   help="Gradient accumulation steps")
    p.add_argument("--encoder_freeze", type=int,   default=ENCODER_FREEZE_EPOCHS,
                   help="Freeze encoder for first N epochs")
    p.add_argument("--backbone",      type=str,   default=ENCODER_NAME,
                   choices=["resnext50_32x4d", "resnext101_32x8d"],
                   help="Encoder backbone. Auto-selected based on GPU/CPU if not set.")
    p.add_argument("--resume",         action="store_true")
    main(p.parse_args())
