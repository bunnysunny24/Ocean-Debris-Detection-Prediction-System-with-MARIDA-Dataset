"""
U-Net Model Architecture and Parameters
Defines the neural network model and loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MODEL PARAMETERS
# ============================================================================

class ModelConfig:
    """Configuration for ResNet-50 + CBAM 16-class model"""
    
    # Input/Output
    IN_CHANNELS = 11          # All Sentinel-2 bands (B1-B12 except B10)
    OUT_CHANNELS = 16         # 16-class MARIDA segmentation
    
    # Architecture
    INITIAL_FILTERS = 64      # Starting number of filters for ResNet
    MAX_FILTERS = 2048        # ResNet-50 max filters at bottleneck
    
    # Training - STABLE SETTINGS FOR SIMPLEUNET
    BATCH_SIZE = 20           # Proven stable batch size
    LEARNING_RATE = 0.0001    # Standard learning rate
    NUM_EPOCHS = 50           # 50 epochs with early stopping
    WEIGHT_DECAY = 1e-5       # L2 regularization
    LR_SCHEDULER_PATIENCE = 10 # ReduceLROnPlateau patience
    
    # Inference
    CONFIDENCE_THRESHOLD = 0.5
    
    # Device
    USE_GPU = True


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def dice_loss(pred_logits, target, smooth=1.0):
    """
    Dice Loss for semantic segmentation.
    Handles class imbalance better than BCE alone.
    
    Args:
        pred_logits: Raw model output (before sigmoid)
        target: Ground truth binary mask
        smooth: Smoothing constant to avoid division by zero
    
    Returns:
        Dice loss value
    """
    # Convert logits to probabilities
    pred_sigmoid = torch.sigmoid(pred_logits)
    pred_sigmoid = torch.clamp(pred_sigmoid, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    
    # Flatten
    pred_flat = pred_sigmoid.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Calculate Dice coefficient
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1.0 - dice_coeff


def combined_loss(pred_logits, target, alpha=0.5):
    """
    Combined BCE + Dice Loss
    More stable than Dice alone, handles imbalance better than BCE alone.
    
    Args:
        pred_logits: Raw model output (logits)
        target: Ground truth binary mask
        alpha: Weight for BCE vs Dice (alpha=0.5 means equal weight)
    
    Returns:
        Combined loss value
    """
    # Ensure valid ranges
    target = torch.clamp(target, 0.0, 1.0)
    
    # BCE with logits (numerically stable)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    
    # Dice loss
    dice = dice_loss(pred_logits, target)
    
    # Combine
    return alpha * bce + (1.0 - alpha) * dice


# ============================================================================
# U-NET ARCHITECTURE
# ============================================================================

class ConvBlock(nn.Module):
    """
    Double convolution block with batch norm and ReLU activation.
    Structure: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class UNet(nn.Module):
    """
    U-Net Architecture for semantic segmentation.
    
    Structure:
    - Encoder: Downsampling path with skip connections
    - Bottleneck: Deepest feature representation
    - Decoder: Upsampling path with skip concatenations
    
    Typical architecture:
    Input (6 channels) -> 64 -> 128 -> 256 -> 512 -> 1024 (bottleneck)
                                          <- 512 <- 256 <- 128 <- 64 -> Output (1 channel)
    """
    
    def __init__(self, in_channels=6, out_channels=1, initial_filters=64):
        super(UNet, self).__init__()
        
        # Encoder blocks (downsampling)
        self.enc1 = ConvBlock(in_channels, initial_filters)                    # 64
        self.enc2 = ConvBlock(initial_filters, initial_filters * 2)            # 128
        self.enc3 = ConvBlock(initial_filters * 2, initial_filters * 4)        # 256
        self.enc4 = ConvBlock(initial_filters * 4, initial_filters * 8)        # 512
        
        # Bottleneck
        self.bottleneck = ConvBlock(initial_filters * 8, initial_filters * 16) # 1024
        
        # Decoder blocks (upsampling)
        self.upconv4 = nn.ConvTranspose2d(initial_filters * 16, initial_filters * 8, 2, stride=2)
        self.dec4 = ConvBlock(initial_filters * 16, initial_filters * 8)       # 512 + 512 -> 512
        
        self.upconv3 = nn.ConvTranspose2d(initial_filters * 8, initial_filters * 4, 2, stride=2)
        self.dec3 = ConvBlock(initial_filters * 8, initial_filters * 4)        # 256 + 256 -> 256
        
        self.upconv2 = nn.ConvTranspose2d(initial_filters * 4, initial_filters * 2, 2, stride=2)
        self.dec2 = ConvBlock(initial_filters * 4, initial_filters * 2)        # 128 + 128 -> 128
        
        self.upconv1 = nn.ConvTranspose2d(initial_filters * 2, initial_filters, 2, stride=2)
        self.dec1 = ConvBlock(initial_filters * 2, initial_filters)            # 64 + 64 -> 64
        
        # Output layer
        self.out = nn.Conv2d(initial_filters, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (batch, 6, 256, 256)
        
        Returns:
            Output tensor of shape (batch, 1, 256, 256)
        """
        # Encoder - downsampling path with skip connections
        e1 = self.enc1(x)                                    # (B, 64, 256, 256)
        e2 = self.enc2(F.max_pool2d(e1, 2))                  # (B, 128, 128, 128)
        e3 = self.enc3(F.max_pool2d(e2, 2))                  # (B, 256, 64, 64)
        e4 = self.enc4(F.max_pool2d(e3, 2))                  # (B, 512, 32, 32)
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))             # (B, 1024, 16, 16)
        
        # Decoder - upsampling path with skip concatenations
        d4 = self.upconv4(b)                                 # (B, 512, 32, 32)
        d4 = torch.cat([d4, e4], dim=1)                      # (B, 1024, 32, 32)
        d4 = self.dec4(d4)                                   # (B, 512, 32, 32)
        
        d3 = self.upconv3(d4)                                # (B, 256, 64, 64)
        d3 = torch.cat([d3, e3], dim=1)                      # (B, 512, 64, 64)
        d3 = self.dec3(d3)                                   # (B, 256, 64, 64)
        
        d2 = self.upconv2(d3)                                # (B, 128, 128, 128)
        d2 = torch.cat([d2, e2], dim=1)                      # (B, 256, 128, 128)
        d2 = self.dec2(d2)                                   # (B, 128, 128, 128)
        
        d1 = self.upconv1(d2)                                # (B, 64, 256, 256)
        d1 = torch.cat([d1, e1], dim=1)                      # (B, 128, 256, 256)
        d1 = self.dec1(d1)                                   # (B, 64, 256, 256)
        
        # Output - logits (no activation, will use BCEWithLogitsLoss)
        out = self.out(d1)                                   # (B, 1, 256, 256)
        
        return out


def create_model(device='cpu'):
    """
    Create and initialize U-Net model.
    
    Args:
        device: Device to move model to ('cpu' or 'cuda')
    
    Returns:
        Model on specified device
    """
    model = UNet(
        in_channels=ModelConfig.IN_CHANNELS,
        out_channels=ModelConfig.OUT_CHANNELS,
        initial_filters=ModelConfig.INITIAL_FILTERS
    ).to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nU-Net Model created:")
    print(f"  Input channels: {ModelConfig.IN_CHANNELS}")
    print(f"  Output channels: {ModelConfig.OUT_CHANNELS}")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Device: {device}")
    
    return model


def get_optimizer(model, lr=None):
    """
    Create Adam optimizer for model.
    
    Args:
        model: PyTorch model
        lr: Learning rate (uses ModelConfig.LEARNING_RATE if None)
    
    Returns:
        Optimizer instance
    """
    if lr is None:
        lr = ModelConfig.LEARNING_RATE
    
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=ModelConfig.WEIGHT_DECAY
    )


def get_scheduler(optimizer, patience=None):
    """
    Create learning rate scheduler.
    Reduces LR when validation loss plateaus.
    
    Args:
        optimizer: PyTorch optimizer
        patience: Patience for scheduler (uses ModelConfig.LR_SCHEDULER_PATIENCE if None)
    
    Returns:
        Learning rate scheduler instance
    """
    if patience is None:
        patience = ModelConfig.LR_SCHEDULER_PATIENCE
    
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience
    )
