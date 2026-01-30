"""
Comprehensive Training Script for Ocean Debris Detection
Includes data augmentation, advanced models, evaluation, drift modeling, and post-processing
"""

import os
import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import from custom modules
from data_preprocessing import preprocess_dataset, create_dataloaders, DebrisDataset
from unet_baseline import ModelConfig, create_model, get_optimizer, get_scheduler, combined_loss
from advanced_segmentation import create_enhanced_model, combined_loss_multiclass
from augment_data import AugmentedDebrisDataset
from eval_metrics import evaluate_comprehensive, save_metrics
from drift_simulator import DriftSimulator, OceanCurrentData, WindData, TrajectoryAnalyzer
from postprocess_results import MaskRefiner, Polygonizer, export_results


def train_epoch(model, train_loader, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        Average training loss for the epoch
    """
    model.train()  # Explicitly set to train mode
    total_loss = 0.0
    batch_count = 0
    skip_count = 0
    
    with tqdm(train_loader, desc="Training", leave=False) as pbar:
        for images, masks in pbar:
            images = images.to(device).float()  # Ensure float type
            masks = masks.to(device).long()     # Ensure long type for class labels
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                skip_count += 1
                pbar.update(1)
                continue
            
            loss = combined_loss_multiclass(outputs, masks)  # Multi-class loss for MARIDA (16 classes)
            
            # Check for NaN/Inf before backward
            if torch.isnan(loss) or torch.isinf(loss):
                skip_count += 1
                pbar.update(1)
                continue
            
            # Backward pass with aggressive gradient clipping
            loss.backward()
            
            # Clip gradients before update
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'skip': skip_count})
    
    if batch_count == 0:
        print(f"\n  ⚠ Warning: All {skip_count} batches skipped due to NaN/Inf")
        return 0.0
    return total_loss / batch_count


def validate_epoch(model, val_loader, device):
    """
    Validate for one epoch.
    
    Returns:
        Average validation loss for the epoch
    """
    model.eval()  # Explicitly set to eval mode
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Validation", leave=False) as pbar:
            for images, masks in pbar:
                images = images.to(device).float()  # Ensure float type
                masks = masks.to(device).long()     # Ensure long type for class labels
                
                outputs = model(images)
                # Use multi-class loss for MARIDA dataset (16 classes)
                loss = combined_loss_multiclass(outputs, masks)
                
                # Skip NaN losses
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    batch_count += 1
                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                else:
                    pbar.set_postfix({'val_loss': 'NaN/Inf'})
    
    if batch_count == 0:
        return float('nan')
    return total_loss / batch_count


def train_model_advanced(model, train_loader, val_loader, device, epochs=50, 
                        model_save_path='best_model.pth', use_multiclass=False):
    """
    Advanced training with augmentation and multi-class support.
    
    Args:
        model: U-Net or Enhanced model instance
        train_loader: Augmented training DataLoader
        val_loader: Validation DataLoader
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of epochs to train
        model_save_path: Path to save best model
        use_multiclass: Use multi-class loss
    
    Returns:
        Trained model
    """
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n{'='*70}")
    print(f" STARTING ADVANCED TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {ModelConfig.BATCH_SIZE}")
    print(f"Learning Rate: {ModelConfig.LEARNING_RATE}")
    print(f"Multi-class: {use_multiclass}")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        
        # Skip scheduler step if val_loss is NaN
        if not np.isnan(val_loss):
            scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{epochs}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")
        
        # Checkpoint best model
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        elif np.isnan(val_loss):
            print(f"  ⚠ Val loss is NaN - skipping checkpoint")
            patience_counter += 1
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= ModelConfig.LR_SCHEDULER_PATIENCE * 2:
            print(f"\n⚠ Early stopping triggered (patience: {patience_counter})")
            break
    
    # Load best model if it exists
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"\n✓ Loaded best model from {model_save_path}")
    else:
        print(f"\n⚠ No best model saved - using current model state")
    
    return model


def train_model(model, train_loader, val_loader, device, epochs=50, model_save_path='best_model.pth'):
    """
    Train U-Net model with validation and model checkpointing.
    
    Args:
        model: U-Net model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of epochs to train
        model_save_path: Path to save best model
    
    Returns:
        Trained model
    """
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n{'='*70}")
    print(f" STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {ModelConfig.BATCH_SIZE}")
    print(f"Learning Rate: {ModelConfig.LEARNING_RATE}")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        
        # Step scheduler
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{epochs}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")
        
        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= ModelConfig.LR_SCHEDULER_PATIENCE * 2:
            print(f"\n⚠ Early stopping triggered (patience: {patience_counter})")
            break
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    print(f"\n✓ Loaded best model from {model_save_path}")
    
    return model


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained U-Net model
        test_loader: DataLoader for test data
        device: Device to evaluate on
    
    Returns:
        Average test loss
    """
    model.eval()
    total_loss = 0.0
    
    print(f"\n{'='*70}")
    print(f" EVALUATING ON TEST SET")
    print(f"{'='*70}")
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                total_loss += loss.item()
                
                pbar.set_postfix({'test_loss': f'{loss.item():.4f}'})
    
    test_loss = total_loss / len(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")
    
    return test_loss


def save_model_info(model, save_path='model_info.json'):
    """Save model information and configuration."""
    info = {
        'model_type': 'UNet',
        'in_channels': ModelConfig.IN_CHANNELS,
        'out_channels': ModelConfig.OUT_CHANNELS,
        'initial_filters': ModelConfig.INITIAL_FILTERS,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'config': {
            'batch_size': ModelConfig.BATCH_SIZE,
            'learning_rate': ModelConfig.LEARNING_RATE,
            'num_epochs': ModelConfig.NUM_EPOCHS,
            'confidence_threshold': ModelConfig.CONFIDENCE_THRESHOLD,
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Saved model info to {save_path}")


def main():
    """Main comprehensive training pipeline."""
    
    print(f"\n{'='*70}")
    print(f" OCEAN DEBRIS DETECTION - COMPREHENSIVE TRAINING PIPELINE")
    print(f"{'='*70}\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠ Using CPU (no GPU detected)")
    
    # Paths
    dataset_dir = 'Dataset'
    preprocessed_dir = 'Dataset/preprocessed'
    splits_file = 'Dataset/splits.json'
    results_dir = 'results'
    model_path = 'best_model.pth'
    enhanced_model_path = 'best_model_enhanced.pth'
    
    os.makedirs(results_dir, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Preprocess Dataset
    # ========================================================================
    if not os.path.exists(os.path.join(preprocessed_dir, 'train')):
        print(f"\n[STEP 1/5] PREPROCESSING DATASET")
        print(f"{'-'*70}")
        
        if not os.path.exists(splits_file):
            print(f"⚠ Splits file not found: {splits_file}")
            print(f"Creating default splits...")
            sys.exit(1)
        
        stats = preprocess_dataset(dataset_dir, preprocessed_dir, splits_file)
        print(f"\nPreprocessing complete:")
        for split, count in stats.items():
            print(f"  {split}: {count} patches")
    else:
        print(f"\n[STEP 1/5] PREPROCESSING SKIPPED (data already exists)")
        print(f"{'-'*70}")
    
    # ========================================================================
    # STEP 2: Create Data Loaders with Augmentation
    # ========================================================================
    print(f"\n[STEP 2/5] LOADING MARIDA DATASET (16 Classes)")
    print(f"{'-'*70}")
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_dir='Dataset',
            batch_size=ModelConfig.BATCH_SIZE,
            normalize=True
        )
        print(f"✓ MARIDA dataset loaded successfully")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 3: Train Enhanced Model (ResNeXt + CBAM)
    # ========================================================================
    print(f"\n[STEP 3/5] TRAINING ENHANCED MODEL (ResNeXt + CBAM)")
    print(f"{'-'*70}")
    
    model_enhanced = create_enhanced_model(in_channels=11, num_classes=16, device=device)
    
    model_enhanced = train_model_advanced(
        model_enhanced,
        train_loader,
        val_loader,
        device,
        epochs=ModelConfig.NUM_EPOCHS,
        model_save_path=enhanced_model_path,
        use_multiclass=True
    )
    
    # ========================================================================
    # STEP 4: Evaluate Model
    # ========================================================================
    print(f"\n[STEP 4/5] EVALUATING MODEL")
    print(f"{'-'*70}")
    
    metrics = evaluate_comprehensive(model_enhanced, test_loader, device, num_classes=4)
    save_metrics(metrics, filepath=os.path.join(results_dir, 'evaluation_metrics.json'))
    
    # ========================================================================
    # STEP 5: Post-Processing & Visualization (Example)
    # ========================================================================
    print(f"\n[STEP 5/5] GENERATING EXAMPLE OUTPUTS")
    print(f"{'-'*70}")
    
    # Get sample batch for visualization
    sample_images, _ = next(iter(test_loader))
    sample_images = sample_images.to(device)
    
    with torch.no_grad():
        sample_predictions = model_enhanced(sample_images)
    
    # Process first sample
    pred_logits = sample_predictions[0].cpu().numpy()  # (4, 256, 256)
    
    # Convert to binary mask
    pred_probs = torch.softmax(torch.from_numpy(pred_logits).unsqueeze(0), dim=1)[0]
    plastic_prob = pred_probs[0].numpy()  # Assume class 0 is plastic
    binary_mask = (plastic_prob > 0.5).astype(np.uint8)
    
    # Refine mask
    refined_mask = MaskRefiner.refine_mask(binary_mask)
    
    # Export results
    sample_image = sample_images[0].cpu().numpy()  # (6, 256, 256)
    rgb_image = sample_image[:3]  # Use first 3 bands as RGB
    export_results(refined_mask.astype(float), rgb_image, results_dir)
    
    # ========================================================================
    # STEP 6: Drift Simulation (Example with synthetic data)
    # ========================================================================
    print(f"\n[BONUS] DRIFT SIMULATION (Example)")
    print(f"{'-'*70}")
    
    # Get centroids from mask
    centroids = Polygonizer.get_centroids(refined_mask)
    
    if len(centroids) > 0:
        # Convert pixel coordinates to lat/lon (example for Pacific Ocean)
        initial_positions = [
            (35.5 + y/1000, 139.8 + x/1000) for y, x in centroids
        ]
        
        # Simulate drift with synthetic data
        simulator = DriftSimulator(leeway_coeff=0.03)
        particles = simulator.simulate_drift(
            initial_positions,
            debris_types=['plastic']*len(initial_positions),
            duration_hours=24,
            dt_hours=1.0
        )
        
        # Export trajectories
        TrajectoryAnalyzer.export_trajectory_geojson(
            particles, 
            os.path.join(results_dir, 'drift_trajectories.geojson')
        )
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f" TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Enhanced Model saved to: {enhanced_model_path}")
    print(f"✓ Evaluation metrics saved to: {results_dir}/evaluation_metrics.json")
    print(f"✓ Segmentation results saved to: {results_dir}/")
    print(f"✓ GeoJSON detections saved to: {results_dir}/detections.geojson")
    print(f"✓ Drift trajectories saved to: {results_dir}/drift_trajectories.geojson")
    
    print(f"\nKey Results:")
    if 'segmentation' in metrics:
        seg = metrics['segmentation']
        print(f"  IoU: {seg['iou']:.4f}")
        print(f"  F1 Score: {seg['f1']:.4f}")
        print(f"  Precision: {seg['precision']:.4f}")
        print(f"  Recall: {seg['recall']:.4f}")
    
    print(f"\nNext Steps:")
    print(f"  1. Load model: model.load_state_dict(torch.load('{enhanced_model_path}'))")
    print(f"  2. Integrate real ocean current data (CMEMS)")
    print(f"  3. Add auxiliary classifier for multi-class debris")
    print(f"  4. Deploy on cloud for real-time detection")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
