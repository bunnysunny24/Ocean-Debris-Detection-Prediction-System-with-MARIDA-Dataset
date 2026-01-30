"""
MARIDA Dataset Loader - 15-Class Semantic Segmentation
Loads Sentinel-2 imagery and semantic segmentation masks from MARIDA dataset
"""

import os
import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader


# MARIDA Class Definitions (15 classes + background)
MARIDA_CLASSES = {
    0: 'Background/Unknown',
    1: 'Marine Debris',
    2: 'Dense Sargassum',
    3: 'Sparse Sargassum',
    4: 'Natural Organic Material',
    5: 'Ship',
    6: 'Clouds',
    7: 'Marine Water',
    8: 'Sediment-Laden Water',
    9: 'Foam',
    10: 'Turbid Water',
    11: 'Shallow Water',
    12: 'Waves',
    13: 'Cloud Shadows',
    14: 'Wakes',
    15: 'Mixed Water'
}

NUM_CLASSES = 16  # Classes 0-15




def load_marida_patch(patch_path):
    """
    Load a MARIDA patch with image and mask.
    
    Args:
        patch_path: Path to image file (e.g., S2_14-11-18_48PZC_0.tif)
    
    Returns:
        image: (H, W, 11) - All 11 Sentinel-2 bands
        mask: (H, W) - Class labels 0-15
        confidence: (H, W) - Confidence levels
    """
    if not str(patch_path).endswith('.tif'):
        patch_path = str(patch_path) + '.tif'
    
    patch_path = Path(patch_path)
    
    # Construct paths for mask and confidence files
    mask_path = patch_path.parent / (patch_path.stem + '_cl.tif')
    conf_path = patch_path.parent / (patch_path.stem + '_conf.tif')
    
    # Load image (all bands)
    with rasterio.open(patch_path) as src:
        image = src.read()  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
    
    # Load mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # (H, W) - single band
    
    # Load confidence
    with rasterio.open(conf_path) as src:
        confidence = src.read(1)  # (H, W)
    
    return image, mask, confidence


def normalize_sentinel2_bands(image):
    """
    Normalize Sentinel-2 bands using percentile clipping.
    
    Args:
        image: (H, W, 11) - Raw Sentinel-2 image
    
    Returns:
        normalized: (H, W, 11) - Normalized to [0, 1]
    """
    # Sentinel-2 bands are typically 0-10000 digital numbers
    # Use 0.5%-99.5% percentile for robust normalization
    image = image.astype(np.float32)
    
    for band_idx in range(image.shape[2]):
        band = image[:, :, band_idx]
        p_low = np.percentile(band, 0.5)
        p_high = np.percentile(band, 99.5)
        
        # Normalize to [0, 1]
        band_norm = (band - p_low) / (p_high - p_low + 1e-7)
        band_norm = np.clip(band_norm, 0, 1)
        image[:, :, band_idx] = band_norm
    
    return image



class DebrisDataset(Dataset):
    """
    PyTorch Dataset for MARIDA Sentinel-2 and mask pairs.
    Supports all 16 semantic classes (0-15).
    """
    
    def __init__(self, image_paths, normalize=True):
        """
        Args:
            image_paths: List of paths to image TIF files
            normalize: Apply normalization
        """
        self.image_paths = image_paths
        self.normalize = normalize
        
        # Auto-generate mask/confidence paths
        self.mask_paths = []
        for img_path in image_paths:
            base = str(img_path).replace('.tif', '')
            self.mask_paths.append(base + '_cl.tif')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            # Load image and mask
            image, mask, confidence = load_marida_patch(img_path)
            
            # Normalize
            if self.normalize:
                image = normalize_sentinel2_bands(image)
            
            # Convert to torch tensors
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)  # (C, H, W)
            
            # Convert mask to long type for cross-entropy loss
            mask = torch.from_numpy(mask.astype(np.int64)).long()
            
            return image, mask
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy tensors on error
            return torch.zeros((11, 256, 256)), torch.zeros((256, 256), dtype=torch.long)


def create_dataloaders(dataset_dir='Dataset', batch_size=8, 
                      train_split=0.7, val_split=0.15,
                      normalize=True):
    """
    Create train, validation, and test DataLoaders for MARIDA dataset.
    Directly loads from MARIDA patches folder.
    
    Args:
        dataset_dir: Path to Dataset folder
        batch_size: Batch size for DataLoader
        train_split: Proportion for training
        val_split: Proportion for validation
        normalize: Apply normalization
    
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_dir = Path(dataset_dir)
    patches_dir = dataset_dir / 'patches'
    
    # Collect all patch files
    all_patch_files = []
    for tile_dir in patches_dir.iterdir():
        if not tile_dir.is_dir():
            continue
        
        # Find base patch files (not _cl or _conf)
        for tif_file in sorted(tile_dir.glob('*.tif')):
            tif_str = str(tif_file)
            # Skip mask and confidence files
            if '_cl.tif' not in tif_str and '_conf.tif' not in tif_str:
                all_patch_files.append(tif_str)
    
    all_patch_files.sort()
    print(f"Found {len(all_patch_files)} patches total")
    
    # Split data
    n_total = len(all_patch_files)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_files = all_patch_files[:n_train]
    val_files = all_patch_files[n_train:n_train + n_val]
    test_files = all_patch_files[n_train + n_val:]
    
    # Create datasets
    train_dataset = DebrisDataset(train_files, normalize=normalize)
    val_dataset = DebrisDataset(val_files, normalize=normalize)
    test_dataset = DebrisDataset(test_files, normalize=normalize)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_files)} samples")
    print(f"  Val: {len(val_files)} samples")
    print(f"  Test: {len(test_files)} samples")
    
    return train_loader, val_loader, test_loader


def preprocess_dataset(dataset_dir='Dataset'):
    """
    Cache information about the MARIDA dataset.
    """
    dataset_dir = Path(dataset_dir)
    patches_dir = dataset_dir / 'patches'
    
    stats = {
        'total_patches': 0,
        'classes': MARIDA_CLASSES,
        'num_classes': NUM_CLASSES
    }
    
    for tile_dir in patches_dir.iterdir():
        if tile_dir.is_dir():
            num_files = len(list(tile_dir.glob('*.tif'))) // 3  # Divide by 3 (image, mask, conf)
            stats['total_patches'] += num_files
            print(f"  {tile_dir.name}: {num_files} patches")
    
    print(f"\nTotal MARIDA patches: {stats['total_patches']}")
    print(f"Classes: {NUM_CLASSES} (0-15)")
    
    return stats
