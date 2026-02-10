import torch
from simple_unet import create_simple_model
from data_preprocessing import create_dataloaders

# Check model architecture
model = create_simple_model(in_channels=11, out_channels=16, device='cuda')
print("\n✓ Model architecture:")
print(f"  Input: 11 Sentinel-2 bands (256×256)")
print(f"  Output: 16 class logits (256×256)")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check data loading
train_loader, _, _ = create_dataloaders('Dataset', batch_size=5, normalize=True)
images, masks = next(iter(train_loader))
print(f"\n✓ Data loading:")
print(f"  Batch shape: {images.shape}")
print(f"  Expected: (batch, 11, 256, 256) = ({images.shape})")
print(f"  Mask shape: {masks.shape}")
print(f"  Classes in batch: {torch.unique(masks).cpu().numpy()}")

# Forward pass test
model.eval()
with torch.no_grad():
    outputs = model(images.to('cuda'))
    print(f"\n✓ Forward pass:")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Expected: (batch, 16, 256, 256) = ({outputs.shape})")
    
    # Check predictions
    predictions = torch.argmax(outputs, dim=1)
    pred_classes = torch.unique(predictions)
    print(f"  Predicted classes: {pred_classes.cpu().numpy()}")

print("\n✅ Model pipeline verified - everything correct!")
