"""
semantic_segmentation/models/resnext_cbam_unet.py
-------------------------------------------------
ResNeXt-50 encoder + CBAM attention + U-Net decoder
for 11-band → 15-class semantic segmentation.

Requires: torch, torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d


# ── CBAM ─────────────────────────────────────────────────────────────────────

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
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd spatial sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.cbam(x)


# ── Main Model ────────────────────────────────────────────────────────────────

class ResNeXtCBAMUNet(nn.Module):
    """
    Parameters
    ----------
    in_channels : int
        Number of input spectral bands (default 11 for MARIDA).
    num_classes : int
        Number of output classes (default 15).
    pretrained : bool
        Load ImageNet weights for ResNeXt-50 (first conv adapted to in_channels).
    """

    def __init__(self, in_channels: int = 11, num_classes: int = 15, pretrained: bool = True):
        super().__init__()

        backbone = resnext50_32x4d(pretrained=pretrained)

        # ── Adapt first conv to arbitrary input channels ──
        orig_conv = backbone.conv1
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            backbone.bn1,
            backbone.relu,
        )
        if pretrained and in_channels != 3:
            # Average the pretrained RGB weights across channel dim, then tile
            with torch.no_grad():
                mean_w = orig_conv.weight.mean(dim=1, keepdim=True)   # (64,1,7,7)
                new_w  = mean_w.repeat(1, in_channels, 1, 1)
                self.enc0[0].weight.copy_(new_w)

        self.pool    = backbone.maxpool   # stride-2

        self.enc1    = backbone.layer1    # 256 ch,  stride 1
        self.enc2    = backbone.layer2    # 512 ch,  stride 2
        self.enc3    = backbone.layer3    # 1024 ch, stride 2
        self.enc4    = backbone.layer4    # 2048 ch, stride 2

        # CBAM on deepest encoder feature
        self.bottleneck_cbam = CBAM(2048)

        # ── Decoder ──
        self.dec4 = DecoderBlock(2048, 1024, 512)
        self.dec3 = DecoderBlock(512,   512,  256)
        self.dec2 = DecoderBlock(256,   256,  128)
        self.dec1 = DecoderBlock(128,    64,   64)

        # Final upsampling to input resolution
        self.final_up   = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_decoder()

    def _init_decoder(self):
        for m in [self.dec4, self.dec3, self.dec2, self.dec1,
                  self.final_up, self.final_conv]:
            for p in m.modules():
                if isinstance(p, nn.Conv2d):
                    nn.init.kaiming_normal_(p.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(p, nn.BatchNorm2d):
                    nn.init.constant_(p.weight, 1)
                    nn.init.constant_(p.bias, 0)

    def forward(self, x):
        # ── Encoder ──
        e0 = self.enc0(x)           # /2
        p  = self.pool(e0)          # /4
        e1 = self.enc1(p)           # /4   256ch
        e2 = self.enc2(e1)          # /8   512ch
        e3 = self.enc3(e2)          # /16  1024ch
        e4 = self.enc4(e3)          # /32  2048ch

        b  = self.bottleneck_cbam(e4)

        # ── Decoder ──
        d4 = self.dec4(b,  e3)      # /16  512ch
        d3 = self.dec3(d4, e2)      # /8   256ch
        d2 = self.dec2(d3, e1)      # /4   128ch
        d1 = self.dec1(d2, e0)      # /2   64ch

        out = self.final_up(d1)     # /1
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.final_conv(out)  # (B, num_classes, H, W)


# ── Loss ─────────────────────────────────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Weighted Cross-Entropy + Dice loss.
    Supports per-pixel confidence weights and ignores index -1 (nodata).
    """

    def __init__(self, class_weights=None, dice_weight: float = 0.4,
                 ignore_index: int = -1, use_focal: bool = False, focal_gamma: float = 2.0):
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.dice_w      = dice_weight
        self.ignore_idx  = ignore_index
        self.use_focal   = use_focal
        self.focal_gamma = focal_gamma
        self.ce = None  # Will be initialized per device

    def _dice(self, pred_soft, target, num_classes):
        """pred_soft: (B,C,H,W) softmax; target: (B,H,W) long."""
        eps   = 1e-6
        valid = (target != self.ignore_idx)
        dice  = 0.0
        for c in range(num_classes):
            t = ((target == c) & valid).float()
            p = pred_soft[:, c] * valid.float()
            dice += (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
        return 1.0 - dice / num_classes

    def forward(self, logits, target, conf_weights=None):
        """
        logits       : (B, C, H, W)
        target       : (B, H, W)  long, -1 = nodata
        conf_weights : (B, H, W)  float in [0,1] or None
        """
        device = logits.device
        if self.class_weights is not None:
            weights = self.class_weights.to(device)
        else:
            weights = None
        # Re-create CE loss on the correct device
        self.ce = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=self.ignore_idx, reduction="none")
        ce_map = self.ce(logits, target.clamp(min=0))   # ignore nodata handled by CE
        if self.use_focal:
            # Focal loss scaling
            pt = torch.exp(-ce_map)
            focal_factor = (1 - pt) ** self.focal_gamma
            ce_map = focal_factor * ce_map
        if conf_weights is not None:
            ce_map = ce_map * conf_weights
        ce_loss = ce_map.mean()

        pred_soft = torch.softmax(logits, dim=1)
        dice_loss = self._dice(pred_soft, target, logits.shape[1])

        return (1.0 - self.dice_w) * ce_loss + self.dice_w * dice_loss
