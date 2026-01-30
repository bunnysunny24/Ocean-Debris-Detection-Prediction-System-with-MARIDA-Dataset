"""
Test script to verify MARIDA data loading and model forward pass
"""

import torch
import sys
from pathlib import Path

# Test data loading
print("="*70)
print(" TESTING MARIDA DATA LOADING")
print("="*70)

try:
    from data_preprocessing import create_dataloaders, NUM_CLASSES, MARIDA_CLASSES
    
    print(f"\n✓ Imported data_preprocessing")
    print(f"  Num classes: {NUM_CLASSES}")
    print(f"  Classes: {list(MARIDA_CLASSES.values())[:5]}...")
    
    # Try to load data
    print(f"\nLoading MARIDA dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir='Dataset',
        batch_size=2,
        normalize=True
    )
    
    print(f"✓ DataLoaders created successfully")
    
    # Check batch
    print(f"\nTesting batch...")
    images, masks = next(iter(train_loader))
    
    print(f"  Images shape: {images.shape}")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Image dtype: {images.dtype}")
    print(f"  Mask dtype: {masks.dtype}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Mask unique values: {torch.unique(masks).tolist()[:10]}")
    print(f"  Num unique mask values: {len(torch.unique(masks))}")
    
    # Test model forward pass
    print(f"\n" + "="*70)
    print(" TESTING MODEL FORWARD PASS")
    print("="*70)
    
    from advanced_segmentation import EnhancedUNet
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    model = EnhancedUNet(in_channels=11, num_classes=16, pretrained=False).to(device)
    print(f"✓ Model created: {model.__class__.__name__}")
    
    images = images.to(device)
    masks = masks.to(device)
    
    print(f"\nRunning forward pass...")
    outputs = model(images)
    
    print(f"  Outputs shape: {outputs.shape}")
    print(f"  Outputs dtype: {outputs.dtype}")
    print(f"  Outputs range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    
    # Test loss computation
    print(f"\n" + "="*70)
    print(" TESTING LOSS COMPUTATION")
    print("="*70)
    
    from advanced_segmentation import combined_loss_multiclass
    
    print(f"\nComputing combined loss...")
    loss = combined_loss_multiclass(outputs, masks, num_classes=16)
    
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Is NaN: {torch.isnan(loss)}")
    
    print(f"\n✓ ALL TESTS PASSED")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
