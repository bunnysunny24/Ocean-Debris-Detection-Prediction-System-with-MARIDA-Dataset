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
        
        # Copy weights from pretrained model (average across channels)
        if pretrained:
            weight = original_conv.weight.data  # (64, 3, 7, 7)
            # Average across original 3 channels, then repeat for in_channels
            weight = weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
            weight = weight.repeat(1, in_channels, 1, 1)  # (64, in_channels, 7, 7)
            self.initial_conv.weight.data = weight
        
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
    Weighted cross-entropy loss for multi-class segmentation.
    
    Args:
        pred_logits: (batch, num_classes, H, W)
        target: (batch, H, W) - class indices
        class_weights: (num_classes,) - weight per class
    """
    if class_weights is None:
        class_weights = torch.ones(pred_logits.size(1))
    
    class_weights = class_weights.to(pred_logits.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    return criterion(pred_logits, target.long())


def dice_loss_multiclass(pred_logits, target, num_classes=4, smooth=1.0):
    """
    Dice loss for multi-class segmentation.
    """
    pred_probs = F.softmax(pred_logits, dim=1)  # (batch, num_classes, H, W)
    
    total_loss = 0.0
    for c in range(num_classes):
        pred_c = pred_probs[:, c].contiguous().view(-1)
        target_c = (target == c).float().contiguous().view(-1)
        
        intersection = (pred_c * target_c).sum()
        dice_coeff = (2.0 * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        total_loss += (1.0 - dice_coeff)
    
    return total_loss / num_classes


def combined_loss_multiclass(pred_logits, target, num_classes=16, alpha=0.5):
    """
    Combined CrossEntropy + Dice loss for multi-class (MARIDA: 16 classes).
    
    Args:
        pred_logits: (B, C, H, W) - Network output logits
        target: (B, H, W) - Ground truth class labels (0-15 for MARIDA)
        num_classes: Number of classes (16 for MARIDA)
        alpha: Balance between CE (alpha) and Dice (1-alpha)
    
    Returns:
        Combined loss
    """
    # Create class weights (emphasize minority classes like Marine Debris)
    # Class 1 is Marine Debris - give it higher weight
    class_weights = torch.ones(num_classes, device=pred_logits.device)
    class_weights[1] = 2.0  # Marine Debris gets 2x weight
    class_weights[0] = 0.5  # Background/Unknown gets lower weight
    
    ce = weighted_cross_entropy_loss(pred_logits, target, class_weights)
    dice = dice_loss_multiclass(pred_logits, target, num_classes)
    
    # Handle NaN
    if torch.isnan(ce):
        ce = torch.tensor(0.0, device=pred_logits.device)
    if torch.isnan(dice):
        dice = torch.tensor(0.0, device=pred_logits.device)
    
    combined = alpha * ce + (1.0 - alpha) * dice
    
    return combined


# ============================================================================
# BACKWARD COMPATIBILITY (Binary segmentation)
# ============================================================================

def create_enhanced_model(in_channels=11, num_classes=16, pretrained=True, device='cpu'):
    """
    Create enhanced U-Net model.
    
    Args:
        in_channels: Input channels (11 for Sentinel-2 MARIDA)
        num_classes: Output classes (16 for MARIDA: classes 0-15)
        pretrained: Use pretrained ResNeXt-50
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
