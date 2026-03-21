"""
semantic_segmentation/models/resnext_cbam_unet.py
-------------------------------------------------
ResNeXt encoder (50 or 101) + CBAM attention + U-Net decoder
for binary marine debris segmentation (debris vs not-debris).

Architecture:
  - Dynamic backbone: resnext50_32x4d or resnext101_32x8d
  - Dropout2d(0.1) in every decoder block  → reduces overfitting
  - Deep supervision auxiliary head at dec3 → stronger mid-level gradient signal
  - HybridLoss: Focal-CE + Tversky (α=0.3, β=0.7) → recall-first rare-class learning
  - Log-Tversky for numerical stability

Requires: torch, torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNeXt50_32X4D_Weights, resnext50_32x4d,
    ResNeXt101_32X8D_Weights, resnext101_32x8d,
)


# ── CBAM ──────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        r = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * a


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        a = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * a


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


# ── Decoder block ─────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """U-Net decoder block with CBAM + Dropout2d for regularisation."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.cbam(x)


# ── Main Model ────────────────────────────────────────────────────────────────

# Backbone encoder channel sizes are identical for resnext50 and resnext101:
#   enc0: 64 ch, enc1: 256 ch, enc2: 512 ch, enc3: 1024 ch, enc4: 2048 ch
# So the decoder is compatible with both — only the backbone loading differs.

SUPPORTED_BACKBONES = {
    "resnext50_32x4d":  (resnext50_32x4d,  ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
    "resnext101_32x8d": (resnext101_32x8d, ResNeXt101_32X8D_Weights.IMAGENET1K_V1),
}


class ResNeXtCBAMUNet(nn.Module):
    """
    ResNeXt + CBAM U-Net with deep supervision.

    Parameters
    ----------
    in_channels : int
        Number of input spectral channels (16 for MARIDA with spectral indices).
    num_classes : int
        Number of output classes (2 for binary debris/not-debris).
    pretrained : bool
        Load ImageNet weights (first conv averaged across input channels).
    dropout : float
        Dropout2d probability in decoder blocks.
    deep_supervision : bool
        Return auxiliary logits from dec3 during training.
    backbone : str
        'resnext50_32x4d' (default, ~41M params, works well on CPU/GPU)
        'resnext101_32x8d' (recommended for GPU with ≥8GB VRAM, ~88M params)
    """

    def __init__(self, in_channels: int = 16, num_classes: int = 2,
                 pretrained: bool = True, dropout: float = 0.1,
                 deep_supervision: bool = True,
                 backbone: str = "resnext50_32x4d"):
        super().__init__()
        self.deep_supervision = deep_supervision

        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"backbone must be one of {list(SUPPORTED_BACKBONES.keys())}, got '{backbone}'")

        build_fn, weights_enum = SUPPORTED_BACKBONES[backbone]
        weights = weights_enum if pretrained else None
        bb = build_fn(weights=weights)

        # ── Adapt first conv to arbitrary input channels ──
        orig_conv = bb.conv1
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            bb.bn1,
            bb.relu,
        )
        if pretrained and in_channels != 3:
            with torch.no_grad():
                mean_w = orig_conv.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
                new_w  = mean_w.repeat(1, in_channels, 1, 1)
                self.enc0[0].weight.copy_(new_w)

        self.pool = bb.maxpool   # stride-2
        self.enc1 = bb.layer1   # 256 ch
        self.enc2 = bb.layer2   # 512 ch
        self.enc3 = bb.layer3   # 1024 ch
        self.enc4 = bb.layer4   # 2048 ch

        # CBAM on deepest encoder feature
        self.bottleneck_cbam = CBAM(2048)

        # Decoder — same widths work for both resnext50 and resnext101
        self.dec4 = DecoderBlock(2048, 1024, 256, dropout=dropout)
        self.dec3 = DecoderBlock(256,   512, 128, dropout=dropout)
        self.dec2 = DecoderBlock(128,   256,  64, dropout=dropout)
        self.dec1 = DecoderBlock(64,     64,  64, dropout=dropout)

        self.final_up   = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # Deep supervision auxiliary head (dec3 output, 1/8 resolution)
        if deep_supervision:
            self.aux_conv = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(64, num_classes, kernel_size=1),
            )

        self._init_decoder()

    def _init_decoder(self):
        modules = [self.dec4, self.dec3, self.dec2, self.dec1,
                   self.final_up, self.final_conv]
        if self.deep_supervision:
            modules.append(self.aux_conv)
        for m in modules:
            for p in m.modules():
                if isinstance(p, nn.Conv2d):
                    nn.init.kaiming_normal_(p.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(p, nn.BatchNorm2d):
                    nn.init.constant_(p.weight, 1)
                    nn.init.constant_(p.bias, 0)

    def forward(self, x):
        e0 = self.enc0(x)
        p  = self.pool(e0)
        e1 = self.enc1(p)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b  = self.bottleneck_cbam(e4)

        d4 = self.dec4(b,  e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)

        out = self.final_up(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        main_logits = self.final_conv(out)

        if self.deep_supervision and self.training:
            aux_logits = self.aux_conv(d3)
            aux_logits = F.interpolate(aux_logits, size=x.shape[2:], mode="bilinear", align_corners=False)
            return main_logits, aux_logits

        return main_logits


# ── Loss ──────────────────────────────────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Focal-CE + Tversky Loss for rare-class debris detection.

    Tversky with β > α penalises missed debris (FN) more than false alarms (FP).
    Default: α=0.3, β=0.7  →  FN penalised 2.3× more than FP.
    Log-Tversky ensures gradient stability at extreme class imbalance.

    Also handles deep supervision: accepts (main_logits, aux_logits) tuple.
    Nodata-safe: ignores index -1 everywhere.
    """

    def __init__(self, class_weights=None, tversky_weight=0.7,
                 ignore_index=-1, use_focal=True,
                 focal_gamma=2.0, label_smoothing=0.0,
                 tversky_alpha=0.3, tversky_beta=0.7,
                 aux_weight=0.4):
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.tversky_w  = tversky_weight
        self.ce_w       = 1.0 - tversky_weight
        self.ignore_idx = ignore_index
        self.use_focal  = use_focal
        self.focal_gamma = focal_gamma
        self.label_smooth = label_smoothing
        self.alpha      = tversky_alpha
        self.beta       = tversky_beta
        self.aux_weight = aux_weight

    def _tversky(self, pred_soft, target, num_classes):
        eps  = 1e-6
        valid = (target != self.ignore_idx)
        losses = []
        for c in range(num_classes):
            t   = ((target == c) & valid).float()
            p   = pred_soft[:, c] * valid.float()
            tp  = (p * t).sum()
            fp  = (p * (1 - t)).sum()
            fn  = ((1 - p) * t).sum()
            tversky_score = (tp + eps) / (tp + self.alpha * fp + self.beta * fn + eps)
            losses.append(-torch.log(tversky_score + eps))
        return sum(losses) / num_classes

    def _single_loss(self, logits, target, conf_weights=None):
        device  = logits.device
        weights = self.class_weights.to(device) if self.class_weights is not None else None

        valid_mask  = (target != self.ignore_idx).float()
        safe_target = target.clone()
        safe_target[target == self.ignore_idx] = 0

        ce_fn  = nn.CrossEntropyLoss(weight=weights, reduction="none",
                                     label_smoothing=self.label_smooth)
        ce_map = ce_fn(logits, safe_target)
        if self.use_focal:
            pt     = torch.exp(-ce_map)
            ce_map = ((1 - pt) ** self.focal_gamma) * ce_map
        ce_map = ce_map * valid_mask
        if conf_weights is not None:
            ce_map = ce_map * conf_weights
        ce_loss = ce_map.sum() / (valid_mask.sum() + 1e-8)

        pred_soft    = torch.softmax(logits, dim=1)
        tversky_loss = self._tversky(pred_soft, target, logits.shape[1])

        return self.ce_w * ce_loss + self.tversky_w * tversky_loss

    def forward(self, logits_or_tuple, target, conf_weights=None):
        if isinstance(logits_or_tuple, tuple):
            main_logits, aux_logits = logits_or_tuple
            return (self._single_loss(main_logits, target, conf_weights)
                    + self.aux_weight * self._single_loss(aux_logits, target, conf_weights))
        return self._single_loss(logits_or_tuple, target, conf_weights)
