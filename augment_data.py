"""
Data Augmentation Module
Geometric, spectral, and GAN-based augmentation for training data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================================
# GEOMETRIC AUGMENTATION
# ============================================================================

class GeometricAugmentation:
    """Random rotations, flips, scaling for spatial augmentation."""
    
    def __init__(self, rotate_limit=360, scale_limit=0.1, p=0.5):
        self.transform = A.Compose([
            A.Rotate(limit=rotate_limit, p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.Affine(scale=(1.0 - scale_limit, 1.0 + scale_limit), p=p),
        ], additional_targets={'mask': 'mask'})
    
    def __call__(self, image, mask=None):
        """
        Args:
            image: (C, H, W) numpy array
            mask: (H, W) numpy array or None
        
        Returns:
            Augmented (image, mask)
        """
        # Convert to (H, W, C) for albumentations
        image_hwc = np.transpose(image, (1, 2, 0))
        
        if mask is not None:
            result = self.transform(image=image_hwc, mask=mask)
            image_hwc = result['image']
            mask = result['mask']
        else:
            result = self.transform(image=image_hwc)
            image_hwc = result['image']
        
        # Convert back to (C, H, W)
        image = np.transpose(image_hwc, (2, 0, 1))
        
        return image, mask


# ============================================================================
# SPECTRAL AUGMENTATION
# ============================================================================

class SpectralAugmentation:
    """Add Gaussian noise, brightness/contrast jitter to bands."""
    
    def __init__(self, noise_std=0.01, brightness_limit=0.1, contrast_limit=0.1, p=0.5):
        self.noise_std = noise_std
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p
    
    def __call__(self, image):
        """
        Args:
            image: (C, H, W) numpy array
        
        Returns:
            Augmented image
        """
        image = image.copy()
        
        # Add Gaussian noise
        if np.random.rand() < self.p:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        # Brightness jitter
        if np.random.rand() < self.p:
            brightness = np.random.uniform(-self.brightness_limit, self.brightness_limit)
            image = np.clip(image + brightness, 0, 1)
        
        # Contrast jitter
        if np.random.rand() < self.p:
            contrast = np.random.uniform(1 - self.contrast_limit, 1 + self.contrast_limit)
            image = np.clip(image * contrast, 0, 1)
        
        return image


# ============================================================================
# GAN-BASED SYNTHESIS (Simple DCGAN)
# ============================================================================

class Generator(nn.Module):
    """Generate synthetic plastic debris patches."""
    
    def __init__(self, latent_dim=100, channels=6):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),  # 128 -> 256
            nn.Sigmoid()
        )
    
    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim) noise vector
        
        Returns:
            (batch, channels, 256, 256) synthetic image
        """
        x = self.fc(z)
        x = x.view(-1, 256, 16, 16)
        x = self.conv_layers(x)
        return x


class Discriminator(nn.Module):
    """Distinguish real vs synthetic patches."""
    
    def __init__(self, channels=6):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 16, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, 256, 256) image
        
        Returns:
            (batch, 1) probability of being real
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# AUGMENTED DATASET
# ============================================================================

class AugmentedDebrisDataset:
    """Dataset with on-the-fly augmentation."""
    
    def __init__(self, base_dataset, use_geometric=True, use_spectral=True, p_aug=0.5):
        """
        Args:
            base_dataset: Original DebrisDataset
            use_geometric: Apply geometric augmentation
            use_spectral: Apply spectral augmentation
            p_aug: Probability of applying augmentation
        """
        self.base_dataset = base_dataset
        self.p_aug = p_aug
        
        self.geometric_aug = GeometricAugmentation(p=p_aug) if use_geometric else None
        self.spectral_aug = SpectralAugmentation(p=p_aug) if use_spectral else None
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        
        # Convert torch tensors to numpy
        image_np = image.numpy()  # (C, H, W)
        mask_np = mask.squeeze(0).numpy()  # (H, W)
        
        # Apply augmentations
        if self.geometric_aug and np.random.rand() < self.p_aug:
            image_np, mask_np = self.geometric_aug(image_np, mask_np)
        
        if self.spectral_aug and np.random.rand() < self.p_aug:
            image_np = self.spectral_aug(image_np)
        
        # Convert back to torch
        image = torch.from_numpy(image_np).float()
        mask = torch.from_numpy(mask_np).float().unsqueeze(0)
        
        return image, mask


# ============================================================================
# SEMI-SUPERVISED LEARNING (Pseudo-labeling)
# ============================================================================

def generate_pseudo_labels(model, unlabeled_loader, device, confidence_threshold=0.95):
    """
    Generate high-confidence pseudo-labels from unlabeled data.
    
    Args:
        model: Trained U-Net model
        unlabeled_loader: DataLoader with unlabeled images
        device: Device to run on
        confidence_threshold: Only keep predictions > this threshold
    
    Returns:
        List of (image, pseudo_label) tuples
    """
    model.eval()
    pseudo_labeled_data = []
    
    with torch.no_grad():
        for images, _ in unlabeled_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            # Get high-confidence predictions
            for i in range(images.size(0)):
                confidence = torch.max(probs[i])
                if confidence > confidence_threshold:
                    pseudo_labeled_data.append(
                        (images[i].cpu(), probs[i].cpu())
                    )
    
    return pseudo_labeled_data


def train_with_pseudo_labels(model, labeled_loader, pseudo_labeled_loader, 
                            optimizer, device, num_epochs=10):
    """
    Train model with both labeled and pseudo-labeled data.
    
    Args:
        model: U-Net model
        labeled_loader: Original labeled data
        pseudo_labeled_loader: Pseudo-labeled data
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs
    """
    from unet_baseline import combined_loss
    from tqdm import tqdm
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        # Train on labeled data
        for images, masks in labeled_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Train on pseudo-labeled data (with lower weight)
        for images, masks in pseudo_labeled_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = 0.5 * combined_loss(outputs, masks)  # Lower weight
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Semi-supervised Epoch {epoch+1}: Loss = {total_loss:.4f}")
