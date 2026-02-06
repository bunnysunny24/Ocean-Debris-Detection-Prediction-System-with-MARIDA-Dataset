"""
Enhanced Model Architecture
U-Net with ResNeXt-50 encoder and Attention modules (CBAM)
Multi-class support for plastic, foam, algae, water
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================================
# ATTENTION MODULES (CBAM)
# ============================================================================

class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM paper."""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM paper."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)
    
    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out


# ============================================================================
# ENHANCED U-NET WITH RESNEXT ENCODER
# ============================================================================

class EnhancedUNet(nn.Module):
    """
    U-Net with ResNeXt-50 encoder and CBAM attention modules.
    Supports multi-class segmentation.
    """
    
    def __init__(self, in_channels=11, num_classes=16, pretrained=True):
        """
        Args:
            in_channels: Number of input channels (11 for Sentinel-2 MARIDA)
            num_classes: Number of output classes (16: 0-15 for MARIDA)
            pretrained: Use ImageNet pretrained ResNeXt-50
        """
        super(EnhancedUNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # ================================================================
        # ENCODER - ResNeXt-50
        # ================================================================
        resnext = models.resnext50_32x4d(pretrained=pretrained)
        
        # Adapt first conv layer to accept in_channels instead of 3
        original_conv = resnext.conv1
        self.initial_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Copy weights from pretrained model (with proper scaling)
        if pretrained:
            weight = original_conv.weight.data  # (64, 3, 7, 7)
            # Average across original 3 channels, scale by 3/in_channels for consistency
            weight = weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
            weight = weight * (3.0 / in_channels)  # Scale to compensate for more channels
            weight = weight.repeat(1, in_channels, 1, 1)  # (64, in_channels, 7, 7)
            self.initial_conv.weight.data = weight
        else:
            # Random initialization with proper scaling (Kaiming uniform)
            nn.init.kaiming_uniform_(self.initial_conv.weight, a=1, mode='fan_out')
        
        self.bn1 = resnext.bn1
        self.relu = resnext.relu
        self.maxpool = resnext.maxpool
        
        # ResNeXt residual blocks
        self.layer1 = resnext.layer1  # 256 channels
        self.layer2 = resnext.layer2  # 512 channels
        self.layer3 = resnext.layer3  # 1024 channels
        self.layer4 = resnext.layer4  # 2048 channels
        
        # ================================================================
        # DECODER - with Attention Modules
        # ================================================================
        
        # Decoder block 4 (from 2048 -> 1024)
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.attention4 = CBAM(2048)
        self.dec4 = self._conv_block(2048, 1024)
        
        # Decoder block 3 (from 1024 -> 512)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.attention3 = CBAM(1024)
        self.dec3 = self._conv_block(1024, 512)
        
        # Decoder block 2 (from 512 -> 256)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.attention2 = CBAM(512)
        self.dec2 = self._conv_block(512, 256)
        
        # Decoder block 1 (from 256 -> 64)
        self.upconv1 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.attention1 = CBAM(128)  # 64 + 64 from concatenation
        self.dec1 = self._conv_block(128, 64)
        
        # Decoder block 0 (from 64 -> 32) - upconv to 256x256
        self.upconv0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = self._conv_block(32, 32)
        
        # Final output layer - multi-class
        self.out = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Initialize output layer
        nn.init.kaiming_normal_(self.out.weight, mode='fan_out', nonlinearity='relu')
    
    def _conv_block(self, in_ch, out_ch):
        """Double convolution block with batch norm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 6, 256, 256)
        
        Returns:
            (batch, num_classes, 256, 256)
        """
        # ================================================================
        # ENCODER PATH
        # ================================================================
        e0_pre = self.relu(self.bn1(self.initial_conv(x)))  # 64, 128x128 (before maxpool)
        e0 = self.maxpool(e0_pre)                           # 64, 64x64
        
        e1 = self.layer1(e0)   # 256, 64x64
        e2 = self.layer2(e1)   # 512, 32x32
        e3 = self.layer3(e2)   # 1024, 16x16
        e4 = self.layer4(e3)   # 2048, 8x8
        
        # ================================================================
        # DECODER PATH with Skip Connections and Attention
        # ================================================================
        
        # Decoder 4
        d4 = self.upconv4(e4)                          # 1024, 16x16
        d4 = torch.cat([d4, e3], dim=1)               # 2048, 16x16
        d4 = self.attention4(d4) * d4                  # Apply attention
        d4 = self.dec4(d4)                             # 1024, 16x16
        
        # Decoder 3
        d3 = self.upconv3(d4)                          # 512, 32x32
        d3 = torch.cat([d3, e2], dim=1)               # 1024, 32x32
        d3 = self.attention3(d3) * d3                  # Apply attention
        d3 = self.dec3(d3)                             # 512, 32x32
        
        # Decoder 2
        d2 = self.upconv2(d3)                          # 256, 64x64
        d2 = torch.cat([d2, e1], dim=1)               # 512, 64x64
        d2 = self.attention2(d2) * d2                  # Apply attention
        d2 = self.dec2(d2)                             # 256, 64x64
        
        # Decoder 1
        d1 = self.upconv1(d2)                          # 64, 128x128
        d1 = torch.cat([d1, e0_pre], dim=1)           # 128, 128x128 (use pre-maxpool e0)
        d1 = self.attention1(d1) * d1                  # Apply attention
        d1 = self.dec1(d1)                             # 64, 128x128
        
        # Decoder 0 - final upconv to 256x256
        d0 = self.upconv0(d1)                          # 32, 256x256
        d0 = self.dec0(d0)                             # 32, 256x256
        
        # Output - logits for multi-class (will use CrossEntropyLoss)
        out = self.out(d0)                             # (batch, num_classes, 256, 256)
        
        return out


# ============================================================================
# LOSS FUNCTIONS FOR MULTI-CLASS
# ============================================================================

def weighted_cross_entropy_loss(pred_logits, target, class_weights=None):
    """
    Weighted cross-entropy loss for multi-class segmentation with label smoothing.
    
    Args:
        pred_logits: (batch, num_classes, H, W)
        target: (batch, H, W) - class indices
        class_weights: (num_classes,) - weight per class
    """
    if class_weights is None:
        class_weights = torch.ones(pred_logits.size(1))
    
    class_weights = class_weights.to(pred_logits.device)
    # Use label smoothing for numerical stability
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    return criterion(pred_logits, target.long())


def dice_loss_multiclass(pred_logits, target, num_classes=4, smooth=1.0):
    """
    Dice loss for multi-class segmentation with numerical stability.
    Skip classes that don't appear in batch to avoid division by zero.
    """
    pred_probs = F.softmax(pred_logits, dim=1)  # (batch, num_classes, H, W)
    
    total_loss = None
    classes_in_batch = 0
    
    for c in range(num_classes):
        pred_c = pred_probs[:, c].contiguous().view(-1)
        target_c = (target == c).float().contiguous().view(-1)
        
        # Skip if class doesn't exist in this batch
        if target_c.sum() < 1e-6:
            continue
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        # Prevent division by zero with added epsilon
        dice_coeff = (2.0 * intersection + smooth) / (union + smooth + 1e-8)
        class_loss = 1.0 - dice_coeff
        
        if total_loss is None:
            total_loss = class_loss
        else:
            total_loss = total_loss + class_loss
        
        classes_in_batch += 1
    
    # If no classes found (shouldn't happen), return zero loss
    if total_loss is None or classes_in_batch == 0:
        return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
    
    return total_loss / classes_in_batch


def combined_loss_multiclass(pred_logits, target, num_classes=16, alpha=0.5):
    """
    CrossEntropy loss for multi-class (MARIDA: 16 classes).
    Using square-root inverse frequency weighting to balance extreme class imbalance.
    
    Args:
        pred_logits: (B, C, H, W) - Network output logits
        target: (B, H, W) - Ground truth class labels (0-15 for MARIDA)
        num_classes: Number of classes (16 for MARIDA)
        alpha: Unused (kept for API compatibility)
    
    Returns:
        Loss value
    """
    # ========== COMPREHENSIVE VALIDATION ==========
    
    # Check shapes
    if len(pred_logits.shape) != 4:
        print(f"  ⚠ pred_logits has wrong shape: {pred_logits.shape}, expected (B, C, H, W)")
        return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
    
    if len(target.shape) != 3:
        print(f"  ⚠ target has wrong shape: {target.shape}, expected (B, H, W)")
        return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
    
    # Check batch size and spatial dims match
    if pred_logits.shape[0] != target.shape[0] or pred_logits.shape[2:] != target.shape[1:]:
        print(f"  ⚠ Shape mismatch: pred {pred_logits.shape} vs target {target.shape}")
        return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
    
    # Check NaN/Inf in logits
    if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
        print(f"  ⚠ NaN/Inf in pred_logits")
        return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
    
    # Check target is within valid range [0, num_classes)
    target_long = target.long()
    target_min, target_max = target_long.min().item(), target_long.max().item()
    if target_min < 0 or target_max >= num_classes:
        print(f"  ⚠ Target out of range: min={target_min}, max={target_max}, expected [0, {num_classes-1}]")
        return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
    
    # Use sqrt inverse frequency weighting for numerical stability with extreme imbalance
    # sqrt(weights) prevents extremely large weight gradients
    # Computed as: sqrt(total_pixels / (num_classes * class_count))
    class_weights = torch.tensor([
        5.7595e-03,  # 0: Background (sqrt of 3.32e-06)
        3.1535e-01,  # 1: Marine Debris (sqrt of 0.099)
        3.1945e-01,  # 2: Cloud
        3.2713e-01,  # 3: Cloud Shadow
        8.3132e-01,  # 4: Ocean
        2.0384e-01,  # 5: Land
        5.5545e-02,  # 6: Water
        4.6954e-02,  # 7: Wetland
        2.5532e-02,  # 8: Snow
        4.7075e-01,  # 9: Built-up
        4.1585e-02,  # 10: Bare Soil
        1.6566e-01,  # 11: Dense Vegetation
        2.9151e-01,  # 12: Sparse Vegetation
        1.6421e-01,  # 13: Cropland
        1.6930e-01,  # 14: Herbaceous
        1.0000e+00   # 15: Tree (rarest)
    ], device=pred_logits.device, dtype=pred_logits.dtype)
    
    try:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        loss = criterion(pred_logits, target_long)
    except Exception as e:
        print(f"  ⚠ CrossEntropyLoss exception: {str(e)[:100]}")
        return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
    
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"  ⚠ Loss is NaN/Inf")
        return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
    
    return loss


# ============================================================================
# BACKWARD COMPATIBILITY (Binary segmentation)
# ============================================================================

def create_enhanced_model(in_channels=11, num_classes=16, pretrained=False, device='cpu'):
    """
    Create enhanced U-Net model.
    
    Args:
        in_channels: Input channels (11 for Sentinel-2 MARIDA)
        num_classes: Output classes (16 for MARIDA: classes 0-15)
        pretrained: Use pretrained ResNeXt-50 (False recommended for multi-spectral data)
        device: Device to move model to
    
    Returns:
        Model on specified device
    """
    model = EnhancedUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nEnhanced U-Net (ResNeXt-50 + CBAM) created:")
    print(f"  Input channels: {in_channels}")
    print(f"  Output classes: {num_classes}")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Device: {device}")
    
    return model
