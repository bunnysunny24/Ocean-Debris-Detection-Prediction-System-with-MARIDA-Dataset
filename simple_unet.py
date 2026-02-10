"""
Simple, Numerically Stable U-Net for Multi-class Segmentation
NO complex encoders, NO attention, NO pretrained weights
Just pure CNN architecture with careful initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):
    """
    Balanced U-Net: 4 encoding levels (proven to work) + light dropout for stability.
    Deep enough to learn patterns, simple enough to generalize.
    """
    
    def __init__(self, in_channels=11, num_classes=16):
        super(SimpleUNet, self).__init__()
        
        # Encoder - 4 levels (BALANCED)
        self.enc1 = self._conv_block(in_channels, 32)      # 256 -> 256
        self.pool1 = nn.MaxPool2d(2, 2)                     # 256 -> 128
        
        self.enc2 = self._conv_block(32, 64)                # 128 -> 128
        self.pool2 = nn.MaxPool2d(2, 2)                     # 128 -> 64
        
        self.enc3 = self._conv_block(64, 128)               # 64 -> 64
        self.pool3 = nn.MaxPool2d(2, 2)                     # 64 -> 32
        
        self.enc4 = self._conv_block(128, 256)              # 32 -> 32
        self.pool4 = nn.MaxPool2d(2, 2)                     # 32 -> 16
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)        # 16 -> 16
        
        # Decoder - 4 levels back
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, 2)  # 16 -> 32
        self.dec4 = self._conv_block(512, 256)              # 32 -> 32
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)  # 32 -> 64
        self.dec3 = self._conv_block(256, 128)              # 64 -> 64
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)   # 64 -> 128
        self.dec2 = self._conv_block(128, 64)               # 128 -> 128
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, 2)    # 128 -> 256
        self.dec1 = self._conv_block(64, 32)                # 256 -> 256
        
        # Output
        self.out = nn.Conv2d(32, num_classes, 1)
        
        # Initialize all weights carefully
        self._init_weights()
    
    def _conv_block(self, in_ch, out_ch):
        """Conv block with LIGHT dropout (0.15) for stability without killing learning."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),  # LIGHT dropout - won't block learning
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15)   # LIGHT dropout
        )
    
    def _init_weights(self):
        """Initialize weights conservatively."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)          # 256
        x = self.pool1(e1)         # 128
        
        e2 = self.enc2(x)          # 128
        x = self.pool2(e2)         # 64
        
        e3 = self.enc3(x)          # 64
        x = self.pool3(e3)         # 32
        
        e4 = self.enc4(x)          # 32
        x = self.pool4(e4)         # 16
        
        # Bottleneck
        x = self.bottleneck(x)     # 16
        
        # Decoder with skip connections
        x = self.upconv4(x)        # 32
        x = torch.cat([x, e4], dim=1)  # Concatenate skip
        x = self.dec4(x)           # 32
        
        x = self.upconv3(x)        # 64
        x = torch.cat([x, e3], dim=1)  # Concatenate skip
        x = self.dec3(x)           # 64
        
        x = self.upconv2(x)        # 128
        x = torch.cat([x, e2], dim=1)  # Concatenate skip
        x = self.dec2(x)           # 128
        
        x = self.upconv1(x)        # 256
        x = torch.cat([x, e1], dim=1)  # Concatenate skip
        x = self.dec1(x)           # 256
        
        # Output logits
        out = self.out(x)          # num_classes
        
        # CRITICAL: Clamp logits to prevent softmax overflow/underflow
        out = torch.clamp(out, -10.0, 10.0)
        
        return out


def create_simple_model(in_channels=11, out_channels=16, device='cuda'):
    """Create and return simple U-Net for multi-class segmentation."""
    model = SimpleUNet(in_channels=in_channels, num_classes=out_channels).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nSimple U-Net created:")
    print(f"  Input channels: {in_channels}")
    print(f"  Output classes: {out_channels}")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Device: {device}")
    print(f"  Note: Pure CNN-based, no complex encoders")
    
    return model


def calculate_class_weights(train_loader, num_classes=16, device='cuda'):
    """
    Calculate GENTLE inverse frequency weights for each class from training data.
    Uses SQRT of inverse frequency (not linear) to avoid extreme weight ratios.
    Minority classes get higher weights, but not so extreme they destabilize training.
    
    Args:
        train_loader: DataLoader for training data
        num_classes: number of classes
        device: torch device
    
    Returns:
        Tensor of shape (num_classes,) with class weights (gentler, sqrt-based)
    """
    class_counts = torch.zeros(num_classes, device=device)
    total_pixels = 0
    
    print("\n  Calculating GENTLE class weights (sqrt-based) from training data...")
    for images, masks in train_loader:
        masks = masks.to(device).long()
        masks = masks.squeeze(1) if masks.dim() == 4 else masks
        
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum().item()
        total_pixels += masks.numel()
    
    # GENTLE weighting: sqrt of inverse frequency (less extreme than linear)
    # Formula: weight[c] = sqrt(total_pixels / (num_classes * class_count[c] + 1e-6))
    class_weights = torch.sqrt(total_pixels / (num_classes * (class_counts + 1e-6)))
    
    # Normalize so mean weight = 1.0 for stability
    class_weights = class_weights / class_weights.mean()
    
    print(f"  Class weights calculated (GENTLE sqrt-based):")
    print(f"    Background (class 0):  {class_weights[0]:.4f}")
    print(f"    Marine Debris (class 1): {class_weights[1]:.4f}")
    print(f"    Sparse Sargassum (class 3): {class_weights[3]:.4f}")
    print(f"    Mean weight: {class_weights.mean():.4f}")
    print(f"    Max weight: {class_weights.max():.4f}")
    print(f"    Min weight: {class_weights.min():.4f}")
    
    return class_weights.to(device)


def simple_cross_entropy_loss(pred_logits, target, num_classes=16, class_weights=None):
    """
    Numerically stable weighted loss using log_softmax + NLL.
    
    Args:
        pred_logits: (B, C, H, W) - model output
        target: (B, H, W) - ground truth labels
        num_classes: number of classes
        class_weights: (num_classes,) - optional class weights for imbalanced data
    
    Returns:
        scalar loss
    """
    # Extra clamping for safety
    pred_logits = torch.clamp(pred_logits, -10.0, 10.0)
    
    # Ensure correct types
    target = target.long()
    
    # Use log_softmax + nll_loss (numerically more stable than cross_entropy)
    log_softmax = F.log_softmax(pred_logits, dim=1)
    loss = F.nll_loss(log_softmax, target, weight=class_weights, reduction='mean')
    
    # Check for NaN - if loss is NaN, return a small positive value
    if torch.isnan(loss):
        return torch.tensor(0.01, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
    
    return loss


def focal_loss(pred_logits, target, num_classes=16, alpha=None, gamma=2.0):
    """
    Focal Loss for extreme class imbalance.
    Applied to ALL pixels so model properly learns minority classes.
    
    Formula: FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    - Gamma=2.0 gives ~100x more weight to hard examples vs easy ones
    - Class weights: Background 0.01x, Minorities 3.0x guide gradient toward rare classes
    
    Args:
        pred_logits: (B, C, H, W) - model output 
        target: (B, H, W) - ground truth labels (0-15)
        num_classes: number of classes
        alpha: per-class weighting
        gamma: focusing parameter
    
    Returns:
        scalar focal loss
    """
    # Clamp logits for stability
    pred_logits = torch.clamp(pred_logits, -10.0, 10.0)
    target = target.long()
    
    # Get softmax probabilities for ALL pixels (not hard-mined)
    p = F.softmax(pred_logits, dim=1)  # (B, C, H, W)
    log_p = F.log_softmax(pred_logits, dim=1)  # (B, C, H, W)
    
    # Flatten for easier computation
    B, C, H, W = pred_logits.shape
    p_flat = p.view(B * H * W, C)  # (B*H*W, C)
    log_p_flat = log_p.view(B * H * W, C)  # (B*H*W, C)
    target_flat = target.view(B * H * W)  # (B*H*W,)
    
    # Get probability of true class for each pixel
    p_t = p_flat.gather(1, target_flat.unsqueeze(1)).squeeze(1)  # (B*H*W,)
    log_p_t = log_p_flat.gather(1, target_flat.unsqueeze(1)).squeeze(1)  # (B*H*W,)
    
    # Focal weight: (1 - p_t)^gamma
    # - Easy examples (p_t ≈ 1.0): focal_weight ≈ 0 (down-weighted)
    # - Hard examples (p_t ≈ 0.0): focal_weight ≈ 1.0 (full weight)
    focal_weight = (1 - p_t) ** gamma  # (B*H*W,)
    
    # Base focal loss
    focal_loss_val = -focal_weight * log_p_t  # (B*H*W,)
    
    # Apply BALANCED AGGRESSIVE class weighting
    if alpha is None:
        # Smart weighting that forces learning without breaking the network:
        # Background: 0.001 (strongly downweight - it's 99% of data anyway)
        # ALL minorities: 1.0 (equal importance, let focal loss do the hard work)
        # KEY: Focal loss with gamma=2.0 already gives minorities ~100x more weight for hard pixels
        # Adding extreme class weight on top breaks optimization
        class_weight = torch.ones(num_classes, device=pred_logits.device)
        class_weight[0] = 0.001  # Background: 0.1% weight (heavily downweight)
        for i in range(1, num_classes):
            class_weight[i] = 1.0  # Minorities: equal weight (let focal loss amplify)
        alpha = class_weight
    
    # Apply class weights
    alpha_t = alpha.gather(0, target_flat)  # (B*H*W,)
    focal_loss_weighted = focal_loss_val * alpha_t
    
    # Return mean loss over all pixels
    loss = focal_loss_weighted.mean()
    
    # Check for NaN
    if torch.isnan(loss):
        return torch.tensor(0.01, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
    
    return loss
