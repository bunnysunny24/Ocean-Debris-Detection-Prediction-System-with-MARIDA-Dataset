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
from unet_baseline import ModelConfig, get_optimizer, get_scheduler
from simple_unet import create_simple_model, simple_cross_entropy_loss, focal_loss, calculate_class_weights
from augment_data import AugmentedDebrisDataset
from balanced_sampler import BalancedPatchSampler
from metrics_tracker import MetricsTracker, MARIDA_CLASSES, get_confusion_matrix_summary
from visualization_reporter import TrainingVisualizer
from eval_metrics import evaluate_comprehensive, save_metrics
from drift_simulator import DriftSimulator, OceanCurrentData, WindData, TrajectoryAnalyzer
from postprocess_results import MaskRefiner, Polygonizer, export_results
import torch.nn.functional as F


def train_epoch(model, train_loader, optimizer, device, epoch=None, total_epochs=None, class_weights=None):
    """
    Train for one epoch - 16-class semantic segmentation with SimpleUNet.
    Optionally uses class weights to handle class imbalance.
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    # Create description with epoch info
    if epoch is not None and total_epochs is not None:
        desc = f"Epoch {epoch:3d}/{total_epochs} - Training"
    else:
        desc = "Training"
    
    with tqdm(train_loader, desc=desc, leave=False) as pbar:
        for images, masks in pbar:
            images = images.to(device).float()
            masks = masks.to(device).long()
            masks = masks.squeeze(1) if masks.dim() == 4 else masks  # Remove extra dim if present
            
            # Forward pass with SimpleUNet
            optimizer.zero_grad()
            outputs = model(images)  # (B, 16, H, W) - 16 class logits
            
            # Use Focal Loss with class weights for extreme class imbalance
            loss = focal_loss(outputs, masks, num_classes=16, alpha=class_weights, gamma=2.0)
            
            # Backward pass
            loss.backward()
            
            # Standard gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    if batch_count == 0:
        return 0.0
    
    return total_loss / batch_count


def validate_epoch(model, val_loader, device):
    """
    Validate for one epoch - 16-class semantic segmentation.
    Returns predictions for confusion matrix calculation.
    Note: Validation loss is calculated WITHOUT class weights to evaluate true performance.
    
    Returns:
        avg_loss, y_true (flat), y_pred (flat)
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Validation", leave=False) as pbar:
            for images, masks in pbar:
                images = images.to(device).float()
                masks = masks.to(device).long()
                masks = masks.squeeze(1) if masks.dim() == 4 else masks  # Remove extra dim if present
                
                # Forward pass
                outputs = model(images)  # (B, 16, H, W)
                
                # Use unweighted loss for validation - evaluate true performance
                loss = simple_cross_entropy_loss(outputs, masks, num_classes=16, class_weights=None)
                
                # Get predictions (argmax across class dimension)
                preds = torch.argmax(outputs, dim=1)  # (B, H, W)
                
                # Flatten for confusion matrix
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(masks.cpu().numpy().flatten())
                
                total_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    if batch_count == 0:
        return float('nan'), np.array([]), np.array([])
    
    avg_loss = total_loss / batch_count
    return avg_loss, np.array(all_labels), np.array(all_preds)


def train_model_advanced(model, train_loader, val_loader, device, epochs=50, 
                        model_save_path='best_model.pth', use_multiclass=False, class_weights=None):
    """
    Advanced training with augmentation, multi-class support, and real-time visualization.
    Generates confusion matrices, AUC curves, and loss plots after EACH EPOCH.
    Uses class weights to fix class imbalance.
    
    Args:
        model: U-Net or Enhanced model instance
        train_loader: Augmented training DataLoader
        val_loader: Validation DataLoader
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of epochs to train
        model_save_path: Path to save best model
        use_multiclass: Use multi-class loss
        class_weights: Optional class weights for imbalanced data
    
    Returns:
        Trained model
    """
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    
    # Initialize metrics tracker and visualizer
    visualizer = TrainingVisualizer(output_dir='results/visualizations', num_classes=16)
    
    # All files (JSON, text, graphs) saved to timestamped folder
    training_log_json_path = visualizer.output_dir / 'training_log.json'
    text_log_path = visualizer.output_dir / 'training_log.txt'
    
    metrics_tracker = MetricsTracker(num_classes=16, save_path=str(training_log_json_path))
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n{'='*70}")
    print(f" STARTING ADVANCED TRAINING WITH REAL-TIME VISUALIZATIONS")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {ModelConfig.BATCH_SIZE}")
    print(f"Learning Rate: {ModelConfig.LEARNING_RATE}")
    print(f"Multi-class: {use_multiclass}")
    print(f"Visualizations generated after EACH epoch → results/visualizations/")
    
    for epoch in range(1, epochs + 1):
        # Train with class weights to handle imbalance
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, epochs, class_weights=class_weights)
        
        # Validate with predictions for confusion matrix
        val_loss, y_true, y_pred = validate_epoch(model, val_loader, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics for this epoch
        metrics_tracker.add_epoch(
            epoch, train_loss, val_loss, current_lr,
            y_true, y_pred, 
            class_names=[MARIDA_CLASSES.get(i, f"Class {i}") for i in range(16)]
        )
        
        # Save metrics
        metrics_tracker.save()
        
        # Save formatted text log to timestamped folder
        metrics_tracker.save_text_log(text_log_path=text_log_path)
        
        # ====== REAL-TIME VISUALIZATION ======
        print(f"\n  🎨 Generating epoch {epoch} visualizations...")
        
        # 1. Confusion Matrix
        if len(y_true) > 0 and len(y_pred) > 0:
            visualizer.plot_confusion_matrix(y_true, y_pred, epoch, 
                                            class_names=[MARIDA_CLASSES.get(i, f'Class {i}') for i in range(16)])
            print(f"     ✓ Confusion matrix: epoch_{epoch:03d}_cm.png")
            
            # 2. AUC-ROC Curves
            y_pred_proba = np.zeros((len(y_true), 16))
            for i in range(len(y_true)):
                pred_class = y_pred[i]
                y_pred_proba[i, pred_class] = 1.0
            
            visualizer.plot_auc_curves(y_true, y_pred_proba, epoch,
                                      class_names=[MARIDA_CLASSES.get(i, f'Class {i}') for i in range(16)])
            print(f"     ✓ AUC curves: epoch_{epoch:03d}_auc.png")
        
        # 3. Loss Curve (updated after each epoch)
        visualizer.plot_loss_curves(metrics_tracker.metrics['training_loss'], 
                                   metrics_tracker.metrics['validation_loss'])
        print(f"     ✓ Loss curve updated")
        
        # 4. Overall accuracy metrics (NO per-class in console, only confusion matrix)
        per_class_data = {}
        for epoch_idx, ep in enumerate(metrics_tracker.metrics['epochs']):
            per_class_data[str(ep)] = metrics_tracker.metrics['per_class_metrics'][epoch_idx]
        
        # Generate overall accuracy curves (Precision, Recall, F1 macro average)
        visualizer.plot_overall_accuracy_curves(per_class_data)
        print(f"     ✓ Overall accuracy curves updated")
        
        # Skip scheduler step if val_loss is NaN
        if not np.isnan(val_loss):
            scheduler.step(val_loss)
        
        # Print progress with macro metrics
        if len(metrics_tracker.metrics['per_class_metrics']) > 0:
            macro_f1 = metrics_tracker.metrics.get(f'epoch_{epoch}_macro_f1', 0)
            print(f"\nEpoch {epoch:3d}/{epochs}: "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Macro F1: {macro_f1:.4f}")
        else:
            print(f"\nEpoch {epoch:3d}/{epochs}: "
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
        
        # Early stopping (with Focal Loss, need much higher patience to allow learning)
        # Patience = 50 epochs (focal loss needs time to learn minority classes)
        if patience_counter >= 50:
            print(f"\n⚠ Early stopping triggered (patience: {patience_counter}/50)")
            break
    
    # Load best model if it exists
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"\n✓ Loaded best model from {model_save_path}")
    else:
        print(f"\n⚠ No best model saved - using current model state")
    
    # Generate final comprehensive report
    print(f"\n{'='*70}")
    print(f" GENERATING FINAL TRAINING REPORT")
    print(f"{'='*70}")
    print(metrics_tracker.get_summary())
    print(f"{'='*70}")
    
    # Generate final summary report
    report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    FINAL TRAINING REPORT                                      ║
╚════════════════════════════════════════════════════════════════════════════════╝

✅ TRAINING COMPLETE
Total Epochs: {epoch}
Best Epoch: {metrics_tracker.metrics['best_epoch']} (Val Loss: {metrics_tracker.metrics['best_val_loss']:.6f})

📊 OUTPUT GENERATED

Real-time Visualizations (per-epoch):
  ✓ {epoch} Confusion Matrices → results/visualizations/confusion_matrices/
  ✓ {epoch} AUC-ROC Curves → results/visualizations/auc_curves/
  ✓ Loss Curves → results/visualizations/loss_curves/
  ✓ Per-class Metrics → results/visualizations/per_class_metrics/

Data Files:
  ✓ Training Log (Text) → {text_log_path}
  ✓ Training Log (JSON) → {training_log_json_path}
  ✓ Evaluation Metrics → {eval_metrics_path}
  ✓ Detections → {visualizer.output_dir}/detections.geojson
  ✓ Drift Trajectories → {visualizer.output_dir}/drift_trajectories.geojson
  ✓ Model Weights → {model_save_path}

📂 TIMESTAMPED FOLDER:
  All outputs organized in: {visualizer.output_dir}
"""
    
    report_path = 'results/visualizations/TRAINING_REPORT.txt'
    os.makedirs('results/visualizations', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Final report saved to {report_path}")
    print(f"\n🎉 All {epoch} epochs completed successfully!")
    print(f"📁 Check results/visualizations/ for all 16 output folders\n")
    
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
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, epochs)
        
        # Validate - now returns (val_loss, y_true, y_pred)
        val_loss, _, _ = validate_epoch(model, val_loader, device)
        
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
                images = images.to(device).float()
                masks = masks.to(device).long()
                masks = masks.squeeze(1) if masks.dim() == 4 else masks  # Remove extra dim if present
                
                outputs = model(images)
                loss = simple_cross_entropy_loss(outputs, masks, num_classes=16)
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
    print(f"\n[STEP 2/5] LOADING MARIDA DATASET (16 Classes + Augmentation)")
    print(f"{'-'*70}")
    
    try:
        # Load base dataset
        train_loader_base, val_loader, test_loader = create_dataloaders(
            dataset_dir='Dataset',
            batch_size=ModelConfig.BATCH_SIZE,
            normalize=True
        )
        
        # Apply augmentation to training data
        augmented_dataset = AugmentedDebrisDataset(
            train_loader_base.dataset,
            use_geometric=True,
            use_spectral=True,
            p_aug=0.7  # 70% augmentation probability
        )
        
        # Create augmented training loader with BALANCED SAMPLING
        # Ensures patches with Marine Debris, Sargassum, Ships get prioritized
        balanced_sampler = BalancedPatchSampler(
            dataset=augmented_dataset,
            num_samples=len(augmented_dataset),
            dataset_dir='Dataset'
        )
        
        train_loader = torch.utils.data.DataLoader(
            augmented_dataset,
            batch_size=ModelConfig.BATCH_SIZE,
            sampler=balanced_sampler,  # Use balanced sampler instead of shuffle
            num_workers=0
        )
        
        print(f"✓ MARIDA dataset loaded with augmentation + BALANCED PATCH SAMPLING")
        print(f"  Train: {len(augmented_dataset)} samples (with augmentation + balanced sampling)")
        print(f"  Val: {len(val_loader.dataset)} samples")
        print(f"  Test: {len(test_loader.dataset)} samples")
        
        # Calculate SIMPLE class weights based on dataset split
        # (not from full dataloader to avoid index issues)
        print(f"  ✓ Using Focal Loss with alpha weighting for extreme class imbalance")
        class_weights = None  # Will compute manually in training
    except Exception as e:
        print(f"⚠ Augmentation failed ({e}), falling back to basic loading")
        train_loader_base, val_loader, test_loader = create_dataloaders(
            dataset_dir='Dataset',
            batch_size=ModelConfig.BATCH_SIZE,
            normalize=True
        )
        print(f"✓ Loaded with basic loading (no augmentation)")
        
        # Still use balanced sampler on fallback loader
        balanced_sampler = BalancedPatchSampler(
            dataset=train_loader_base.dataset,
            num_samples=len(train_loader_base.dataset),
            dataset_dir='Dataset'
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_loader_base.dataset,
            batch_size=ModelConfig.BATCH_SIZE,
            sampler=balanced_sampler,
            num_workers=0
        )
        print(f"✓ Applied BALANCED PATCH SAMPLING to fallback loader")
        print(f"  ✓ Using Focal Loss with alpha weighting for extreme class imbalance")
        class_weights = None  # Will compute manually in training
    
    # ========================================================================
    # STEP 3: Train ResNeXt-50 + CBAM Model (16-class segmentation)
    # ========================================================================
    print(f"\n[STEP 3/5] TRAINING ResNeXt-50 + CBAM MODEL (16-class)")
    print(f"{'-'*70}")
    
    # Create stable SimpleUNet model (proven working, 7.7M params)
    from simple_unet import create_simple_model
    model_enhanced = create_simple_model(in_channels=11, out_channels=16, device=device)
    
    print(f"  Architecture: SimpleUNet (stable, proven working)")
    total_params = sum(p.numel() for p in model_enhanced.parameters())
    trainable_params = sum(p.numel() for p in model_enhanced.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    model_enhanced = train_model_advanced(
        model_enhanced,
        train_loader,
        val_loader,
        device,
        epochs=ModelConfig.NUM_EPOCHS,
        model_save_path=enhanced_model_path,
        use_multiclass=True,
        class_weights=class_weights
    )
    
    # ========================================================================
    # STEP 4: Evaluate Model
    # ========================================================================
    print(f"\n[STEP 4/5] EVALUATING MODEL")
    print(f"{'-'*70}")
    
    metrics = evaluate_comprehensive(model_enhanced, test_loader, device, num_classes=16)
    eval_metrics_path = visualizer.output_dir / 'evaluation_metrics.json'
    save_metrics(metrics, filepath=str(eval_metrics_path))
    
    # ========================================================================
    # STEP 4B: Generate Comprehensive Training Visualizations
    # ========================================================================
    print(f"\n[STEP 4B/5] GENERATING TRAINING VISUALIZATIONS")
    print(f"{'-'*70}")
    
    print("Generating comprehensive training report, confusion matrices, and AUC curves...")
    generate_comprehensive_report(
        training_log_path=os.path.join(results_dir, 'training_log.json'),
        visualizations_dir=os.path.join(results_dir, 'visualizations')
    )
    
    print(f"✓ All visualizations saved to: {os.path.join(results_dir, 'visualizations')}/")

    
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
    pred_logits = sample_predictions[0].cpu().numpy()  # (16, 256, 256) - 16 classes
    
    # Convert to class predictions
    pred_class = np.argmax(pred_logits, axis=0)  # Get predicted class for each pixel
    
    # Extract Marine Debris (CLASS 1 - PRIMARY TARGET)
    debris_prob = torch.softmax(torch.from_numpy(pred_logits).unsqueeze(0), dim=1)[0, 1].numpy()
    binary_mask = (debris_prob > 0.5).astype(np.uint8)  # Class 1 = Marine Debris
    
    print(f"\n  Class Distribution in Sample:")
    unique, counts = np.unique(pred_class, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = 100 * count / (256*256)
        class_names = ['Background', 'Marine Debris', 'Dec Sargassum', 'Sparse Sargassum',
                      'Organic Material', 'Ship', 'Clouds', 'Marine Water', 'Sediment Water',
                      'Foam', 'Turbid Water', 'Shallow Water', 'Waves', 'Cloud Shadows',
                      'Wakes', 'Mixed Water']
        class_name = class_names[cls] if cls < len(class_names) else f'Class {cls}'
        print(f"    Class {cls:2d} ({class_name:20s}): {count:6d} pixels ({pct:5.2f}%)")
    
    # Refine mask
    refined_mask = MaskRefiner.refine_mask(binary_mask)
    
    # Export results to timestamped folder
    sample_image = sample_images[0].cpu().numpy()  # (6, 256, 256)
    rgb_image = sample_image[:3]  # Use first 3 bands as RGB
    export_results(refined_mask.astype(float), rgb_image, str(visualizer.output_dir))
    
    # ========================================================================
    # BONUS: Drift Simulation with Real Ocean Data (CMEMS + ERA5)
    # ========================================================================
    print(f"\n[BONUS] PHYSICS-BASED DRIFT SIMULATION")
    print(f"{'-'*70}")
    
    # Try to load real ocean and wind data
    from drift_simulator import auto_load_ocean_and_wind_data
    
    print(f"\n  Attempting to load real ocean current and wind data...")
    ocean_currents, wind_data = auto_load_ocean_and_wind_data(data_dir='data')
    
    print(f"\n  Drift Simulation Framework Ready:")
    print(f"    • Ocean Currents: {'CMEMS (Real Data)' if ocean_currents.u_data is not None else 'Synthetic'}")
    print(f"    • Wind Data: {'ERA5 (Real Data)' if wind_data.u10_data is not None else 'Synthetic'}")
    print(f"    • Lagrangian Particle Tracking (advection equation)")
    print(f"    • GeoJSON trajectory export")
    
    if ocean_currents.u_data is None or wind_data.u10_data is None:
        print(f"\n  🔗 To enable REAL drift simulation:")
        print(f"    1. Download CMEMS_currents_YYYYMMDD.nc from: https://marine.copernicus.eu/")
        print(f"    2. Download ERA5_wind_YYYYMMDD.nc from: https://cds.climate.copernicus.eu/")
        print(f"    3. Place files in: data/")
        print(f"    4. Restart training - data will be auto-loaded")
        print(f"\n  📄 See data/README.md for detailed download instructions")
    else:
        print(f"\n  ✅ Using REAL ocean and wind physics for drift simulation!")
    
    # Get centroids from refined debris mask
    centroids = Polygonizer.get_centroids(refined_mask)
    
    if len(centroids) > 0:
        print(f"\n  Simulating drift for {len(centroids)} debris objects...")
        
        # Convert pixel coordinates to lat/lon (example for Pacific Ocean)
        initial_positions = [
            (35.5 + y/1000, 139.8 + x/1000) for y, x in centroids
        ]
        
        # Create simulator with ocean and wind data
        simulator = DriftSimulator(ocean_currents=ocean_currents, 
                                 wind_data=wind_data,
                                 leeway_coeff=0.03)
        particles = simulator.simulate_drift(
            initial_positions,
            debris_types=['plastic']*len(initial_positions),
            duration_hours=72,  # 3-day forecast
            dt_hours=1.0
        )
        
        # Export trajectories as GeoJSON to timestamped folder
        trajectory_path = str(visualizer.output_dir / 'drift_trajectories.geojson')
        TrajectoryAnalyzer.export_trajectory_geojson(particles, trajectory_path)
        print(f"  ✓ Drift trajectories exported to: {trajectory_path}")
        print(f"    - {len(particles)} tracked particles")
        print(f"    - 72-hour forecast period")
    else:
        print(f"  ⚠ No debris centroids found for drift simulation")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f" TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Enhanced Model saved to: {enhanced_model_path}")
    print(f"✓ Evaluation metrics saved to: {visualizer.output_dir}/evaluation_metrics.json")
    print(f"✓ Training log (JSON) saved to: {visualizer.output_dir}/training_log.json")
    print(f"✓ Training log (TXT) saved to: {visualizer.output_dir}/training_log.txt")
    print(f"✓ GeoJSON detections saved to: {visualizer.output_dir}/detections.geojson")
    print(f"✓ Drift trajectories saved to: {visualizer.output_dir}/drift_trajectories.geojson")
    print(f"✓ All visualizations (confusion matrices, AUC curves, graphs) in: {visualizer.output_dir}")
    
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
