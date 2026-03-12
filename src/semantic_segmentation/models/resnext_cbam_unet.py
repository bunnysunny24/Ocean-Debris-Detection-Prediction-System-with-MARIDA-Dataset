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

# ── Lovász-Softmax (better IoU surrogate than Dice) ─────────────────────────

def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors.
    Ref: Berman et al., "The Lovász-Softmax loss" (CVPR 2018)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_softmax_flat(probas, labels, classes='present'):
    """Multi-class Lovász-Softmax loss on flattened tensors.
    probas: [P, C] softmax probabilities, labels: [P] ground truth."""
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ('all', 'present') else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        fg_class = probas[:, c]
        errors = (fg - fg_class).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm.data]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=probas.device, requires_grad=True)


def lovasz_softmax(probas, labels, classes='present', ignore_index=-1):
    """Multi-class Lovász-Softmax loss for 4D tensors (B,C,H,W)."""
    B, C, H, W = probas.shape
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore_index is not None:
        valid = labels != ignore_index
        probas = probas[valid]
        labels = labels[valid]
    return _lovasz_softmax_flat(probas, labels, classes=classes)


class HybridLoss(nn.Module):
    """
    Weighted Focal-CE + Lovász-Softmax + Dice loss + Boundary-aware auxiliary.
    Supports per-pixel confidence weights and ignores index -1 (nodata).
    Lovász directly optimises IoU; Dice provides overlap gradient; Focal-CE handles hard examples.
    Boundary term sharpens edges of small debris patches.
    """

    def __init__(self, class_weights=None, dice_weight: float = 0.3,
                 lovasz_weight: float = 0.4, boundary_weight: float = 0.1,
                 ignore_index: int = -1, use_focal: bool = True,
                 focal_gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.dice_w       = dice_weight
        self.lovasz_w     = lovasz_weight
        self.boundary_w   = boundary_weight
        self.ce_w         = 1.0 - dice_weight - lovasz_weight - boundary_weight
        self.ignore_idx   = ignore_index
        self.use_focal    = use_focal
        self.focal_gamma  = focal_gamma
        self.label_smooth = label_smoothing

    @staticmethod
    def _extract_boundary(mask, ignore_idx=-1):
        """Extract boundary pixels via morphological dilation - erosion on the debris mask.
        Returns a float tensor of same shape with 1.0 at boundary pixels."""
        debris = (mask == 0).float()  # debris class = 0
        valid = (mask != ignore_idx).float()
        debris = debris * valid
        # Use max-pool as dilation and -max_pool(-x) as erosion (3x3 kernel)
        debris_4d = debris.unsqueeze(1)  # (B,1,H,W)
        dilated = torch.nn.functional.max_pool2d(debris_4d, kernel_size=3, stride=1, padding=1)
        eroded  = -torch.nn.functional.max_pool2d(-debris_4d, kernel_size=3, stride=1, padding=1)
        boundary = ((dilated - eroded) > 0).float().squeeze(1)  # (B,H,W)
        return boundary * valid

    def _boundary_loss(self, logits, target):
        """BCE loss on debris probability at boundary pixels."""
        boundary = self._extract_boundary(target, self.ignore_idx)
        n_boundary = boundary.sum()
        if n_boundary < 1:
            return torch.tensor(0.0, device=logits.device)
        pred_debris_prob = torch.softmax(logits, dim=1)[:, 0]  # P(debris)
        gt_debris = ((target == 0) & (target != self.ignore_idx)).float()
        bce = torch.nn.functional.binary_cross_entropy(
            pred_debris_prob.clamp(1e-6, 1 - 1e-6), gt_debris, reduction='none'
        )
        return (bce * boundary).sum() / (n_boundary + 1e-8)

    def _dice(self, pred_soft, target, num_classes):
        """Debris-weighted Dice loss. Debris class gets 2x weight in Dice."""
        eps   = 1e-6
        valid = (target != self.ignore_idx)
        dice_per_class = []
        for c in range(num_classes):
            t = ((target == c) & valid).float()
            p = pred_soft[:, c] * valid.float()
            dice_score = (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
            dice_per_class.append(dice_score)
        # Weight debris class (c=0) more heavily in dice
        if num_classes >= 2:
            weighted_dice = 0.7 * dice_per_class[0] + 0.3 * dice_per_class[1]
        else:
            weighted_dice = sum(dice_per_class) / num_classes
        return 1.0 - weighted_dice

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

        # ── Valid-pixel mask (nodata = -1 must not contribute to loss) ──
        valid_mask = (target != self.ignore_idx).float()
        safe_target = target.clone()
        safe_target[target == self.ignore_idx] = 0

        # ── Focal Cross-Entropy ──
        ce_fn  = torch.nn.CrossEntropyLoss(
            weight=weights, reduction="none",
            label_smoothing=self.label_smooth,
        )
        ce_map = ce_fn(logits, safe_target)

        if self.use_focal:
            pt = torch.exp(-ce_map)
            ce_map = ((1 - pt) ** self.focal_gamma) * ce_map

        ce_map = ce_map * valid_mask
        if conf_weights is not None:
            ce_map = ce_map * conf_weights
        ce_loss = ce_map.sum() / (valid_mask.sum() + 1e-8)

        # ── Online Hard Example Mining: boost loss on misclassified debris ──
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            debris_mask = (safe_target == 0) & (target != self.ignore_idx)
            misclassified_debris = debris_mask & (preds != 0)
            ohem_weight = torch.ones_like(valid_mask)
            ohem_weight[misclassified_debris] = 3.0  # Extra penalty for missed debris
        ce_loss_ohem = (ce_map * ohem_weight).sum() / (valid_mask.sum() + 1e-8)
        ce_loss = 0.5 * ce_loss + 0.5 * ce_loss_ohem

        # ── Softmax probabilities ──
        pred_soft = torch.softmax(logits, dim=1)

        # ── Dice loss (debris-weighted) ──
        dice_loss = self._dice(pred_soft, target, logits.shape[1])

        # ── Lovász-Softmax loss (classes='all' to always optimise debris even when not predicted) ──
        lovasz_loss = lovasz_softmax(pred_soft, target, classes='all',
                                     ignore_index=self.ignore_idx)

        # ── Boundary-aware auxiliary loss ──
        boundary_loss = self._boundary_loss(logits, target) if self.boundary_w > 0 else 0.0

        return self.ce_w * ce_loss + self.dice_w * dice_loss + self.lovasz_w * lovasz_loss + self.boundary_w * boundary_loss
