"""
Stratified Batch Sampler: Guarantees class representation in every batch.
Ensures every batch has minority class patches, preventing total dropout.
"""

import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader
from collections import defaultdict


class StratifiedByClassSampler(Sampler):
    """
    Batch sampler that guarantees each batch has diverse class representation.
    
    Strategy:
    - Group patches by their dominant class (class with most pixels)
    - Sample each batch to include patches from DIFFERENT classes
    - Ensures model never sees a batch of 100% background
    - Ensures minority classes always appear
    """
    
    def __init__(self, dataset, batch_size=20, num_samples=None):
        """
        Args:
            dataset: Training dataset with (image, mask) pairs
            batch_size: Samples per batch (will try to split evenly across classes)
            num_samples: Total samples to draw (default: len(dataset))
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = num_samples or len(dataset)
        
        # Analyze dataset: assign each patch to its dominant class
        print(f"  🏷️  Analyzing patch class compositions...")
        self.patch_to_dominant_class = {}
        self.class_to_patches = defaultdict(list)
        
        for idx in range(len(dataset)):
            try:
                _, mask = dataset[idx]
                mask = mask.numpy() if torch.is_tensor(mask) else mask
                
                # Find dominant (most common) non-background class
                unique, counts = np.unique(mask, return_counts=True)
                
                # Get non-background classes
                non_bg = [(cls, cnt) for cls, cnt in zip(unique, counts) if cls != 0]
                
                if non_bg:
                    # Dominant minority class
                    dominant_class = max(non_bg, key=lambda x: x[1])[0]
                else:
                    # Pure background
                    dominant_class = 0
                
                self.patch_to_dominant_class[idx] = dominant_class
                self.class_to_patches[dominant_class].append(idx)
                
                if (idx + 1) % 100 == 0:
                    print(f"    Analyzed {idx+1}/{len(dataset)} patches...")
            
            except Exception as e:
                print(f"    Warning: Could not analyze patch {idx}: {e}")
                self.patch_to_dominant_class[idx] = 0
                self.class_to_patches[0].append(idx)
        
        # Print class distribution
        print(f"  Class distribution in dataset:")
        for cls in sorted(self.class_to_patches.keys()):
            count = len(self.class_to_patches[cls])
            pct = 100 * count / len(dataset)
            print(f"    Class {int(cls):2d}: {count:4d} patches ({pct:5.1f}%)")
        
        print(f"  → Each batch will have ~{batch_size} patches from diverse classes")
    
    def __iter__(self):
        """Generate batch indices ensuring class diversity."""
        # Number of batches to create
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            batch = []
            available_patches = list(range(len(self.dataset)))
            
            # Get unique classes in this batch (round-robin through classes)
            classes_in_dataset = list(self.class_to_patches.keys())
            samples_per_class = max(1, self.batch_size // len(classes_in_dataset))
            
            # Sample from each class to fill batch
            for cls in classes_in_dataset:
                class_patches = self.class_to_patches[cls]
                samples_to_draw = min(samples_per_class, len(class_patches))
                
                if samples_to_draw > 0:
                    sampled_indices = np.random.choice(
                        class_patches,
                        size=samples_to_draw,
                        replace=True
                    )
                    batch.extend(sampled_indices)
            
            # Fill remaining slots with random patches
            while len(batch) < self.batch_size:
                batch.append(np.random.randint(0, len(self.dataset)))
            
            # Trim to batch size
            batch = batch[:self.batch_size]
            
            for idx in batch:
                yield idx
    
    def __len__(self):
        return self.num_samples


def create_stratified_batches(train_dataset, batch_size=20):
    """
    Create DataLoader with stratified batch sampling.
    Guarantees each batch has diverse class representation.
    
    Args:
        train_dataset: Training dataset
        batch_size: Samples per batch
    
    Returns:
        DataLoader with stratified sampler
    """
    
    sampler = StratifiedByClassSampler(
        train_dataset,
        batch_size=batch_size,
        num_samples=len(train_dataset)
    )
    
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0
    )
    
    return loader
