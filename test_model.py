"""
Quick test script to diagnose model training and inference issues
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from advanced_segmentation import create_enhanced_model, combined_loss_multiclass
from data_preprocessing import create_dataloaders

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"Device: {device}")
print(f"{'='*70}\n")

# Load data
print("[1] Loading data...")
try:
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir='Dataset',
        batch_size=4
    )
    print(f"✓ Train: {len(train_loader)} batches, Val: {len(val_loader)}, Test: {len(test_loader)}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# Create model
print("\n[2] Creating model...")
try:
    model = create_enhanced_model(in_channels=11, num_classes=16, device=device)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"✗ Error creating model: {e}")
    exit(1)

# Test forward pass
print("\n[3] Testing forward pass...")
try:
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(train_loader))
        images = images.to(device).float()
        masks = masks.to(device).long()
        
        print(f"  Input shape: {images.shape}")
        print(f"  Mask shape: {masks.shape}")
        print(f"  Mask unique values: {torch.unique(masks)}")
        
        outputs = model(images)
        print(f"  Output shape: {outputs.shape}")
        print(f"  Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
        print(f"  Output contains NaN: {torch.isnan(outputs).any()}")
        print(f"  Output contains Inf: {torch.isinf(outputs).any()}")
        
        # Check predictions
        preds = torch.argmax(outputs, dim=1)
        print(f"  Predicted classes: {torch.unique(preds)}")
        print(f"  Prediction distribution: {np.bincount(preds.flatten().cpu().numpy(), minlength=16)}")
        
except Exception as e:
    print(f"✗ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test loss calculation
print("\n[4] Testing loss calculation...")
try:
    loss = combined_loss_multiclass(outputs, masks, num_classes=16)
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Loss is valid: {not torch.isnan(loss) and not torch.isinf(loss)}")
except Exception as e:
    print(f"✗ Error in loss calculation: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test metrics
print("\n[5] Testing metrics calculation...")
try:
    from eval_metrics import SegmentationMetrics
    
    metrics = SegmentationMetrics(num_classes=16)
    
    # Run through full test set
    for images, masks in test_loader:
        images = images.to(device).float()
        masks = masks.to(device).long()
        with torch.no_grad():
            outputs = model(images)
        metrics.update(outputs, masks)
    
    seg_metrics = metrics.get_metrics()
    print(f"  Accuracy: {seg_metrics['accuracy']:.4f}")
    print(f"  Precision: {seg_metrics['precision']:.4f}")
    print(f"  Recall: {seg_metrics['recall']:.4f}")
    print(f"  F1: {seg_metrics['f1']:.4f}")
    print(f"  IoU: {seg_metrics['iou']:.4f}")
    
except Exception as e:
    print(f"✗ Error in metrics: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test on single image
print("\n[6] Testing on single image...")
try:
    # Get a test image
    test_image, test_mask = next(iter(test_loader))
    test_image = test_image[0:1].to(device).float()  # Single image
    
    model.eval()
    with torch.no_grad():
        output = model(test_image)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    print(f"  Input shape: {test_image.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Prediction shape: {pred.shape}")
    print(f"  Predicted classes in output: {np.unique(pred)}")
    print(f"  Class distribution: {np.bincount(pred.flatten(), minlength=16)}")
    
except Exception as e:
    print(f"✗ Error in single image test: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("✓ All tests passed!")
print("="*70)
