"""
Balanced Patch Sampler for Extreme Class Imbalance
Ensures training batches contain minority class patches (Marine Debris, Sargassum, etc.)
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from pathlib import Path


def get_patch_class_weights(dataset, dataset_dir='Dataset', num_classes=16):
    """
    Calculate which patches in the DATASET contain which classes.
    Only calculates weights for patches actually in the dataset.
    
    Args:
        dataset: PyTorch Dataset with image_paths attribute
        dataset_dir: Path to dataset directory
        num_classes: Number of classes
    
    Returns:
        patch_weights: numpy array with weight for each sample in dataset
    """
    # Get the actual image paths from the dataset
    if hasattr(dataset, 'image_paths'):
        image_paths = dataset.image_paths
    else:
        # If dataset is wrapped, try to get base_dataset
        if hasattr(dataset, 'base_dataset'):
            image_paths = dataset.base_dataset.image_paths
        else:
            raise ValueError("Dataset doesn't have image_paths attribute")
    
    print(f"  Calculating weights for {len(image_paths)} patches in this dataset...")
    
    # Class importance scores
    class_importance = {
        0: 0.0,   # Background
        1: 100.0, # Marine Debris
        2: 50.0,  # Dense Sargassum
        3: 50.0,  # Sparse Sargassum
        4: 40.0,  # Natural Organic Material
        5: 30.0,  # Ship
        9: 40.0,  # Foam
        15: 50.0, # Mixed Water
    }
    
    patch_weights = []
    
    try:
        import rasterio
        for i, img_path in enumerate(image_paths):
            if i % 100 == 0 and i > 0:
                print(f"    Analyzed {i}/{len(image_paths)} patches...")
            
            # Get mask path from image path
            mask_path = str(img_path).replace('.tif', '_cl.tif')
            mask_path = mask_path.replace('patches/', 'patches/')  # Ensure correct path
            
            try:
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                    unique_classes = np.unique(mask)
                    
                    patch_weight = sum(
                        class_importance.get(int(c), 5.0)
                        for c in unique_classes
                        if c > 0
                    )
                    
                    if patch_weight == 0:
                        patch_weight = 0.1
                    
                    patch_weights.append(patch_weight)
            except Exception as e:
                # If mask can't be read, give default weight
                patch_weights.append(0.1)
    
    except ImportError:
        print("  Using PIL for mask reading (rasterio not available)...")
        from PIL import Image
        for i, img_path in enumerate(image_paths):
            if i % 100 == 0 and i > 0:
                print(f"    Analyzed {i}/{len(image_paths)} patches...")
            
            mask_path = str(img_path).replace('.tif', '_cl.tif')
            
            try:
                mask = np.array(Image.open(mask_path))
                unique_classes = np.unique(mask)
                
                patch_weight = sum(
                    class_importance.get(int(c), 5.0)
                    for c in unique_classes
                    if c > 0
                )
                
                if patch_weight == 0:
                    patch_weight = 0.1
                
                patch_weights.append(patch_weight)
            except:
                patch_weights.append(0.1)
    
    return np.array(patch_weights, dtype=np.float32)


class BalancedPatchSampler(Sampler):
    """
    Sampler that gives higher probability to patches containing rare classes.
    Only samples from patches actually in the dataset (train/val/test splits).
    """
    
    def __init__(self, dataset, num_samples=None, replacement=True, dataset_dir='Dataset'):
        """
        Args:
            dataset: PyTorch Dataset with image_paths attribute
            num_samples: Number of samples to draw per epoch (default: len(dataset))
            replacement: Whether to sample with replacement
            dataset_dir: Path to dataset directory
        """
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        self.replacement = replacement
        
        # Calculate weights ONLY for patches in this dataset split
        print("\n  📊 Calculating balanced sampler weights for this dataset split...")
        self.weights = torch.from_numpy(
            get_patch_class_weights(dataset=dataset, dataset_dir=dataset_dir, num_classes=16)
        ).float()
        
        print(f"     Max patch weight: {self.weights.max():.2f}")
        print(f"     Mean patch weight: {self.weights.mean():.2f}")
        print(f"     Min patch weight: {self.weights.min():.2f}")
        print(f"     Total samples in dataset: {len(self.weights)}")
        
        # Verify we have correct number of weights
        if len(self.weights) != len(dataset):
            raise ValueError(
                f"Weights mismatch: {len(self.weights)} weights for {len(dataset)} dataset samples. "
                f"Sampler is for different dataset!"
            )
    
    def __iter__(self):
        """Generate sampled indices."""
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples
