"""
Evaluation Metrics Module
Precision, Recall, IoU, F1, Drift accuracy
"""

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import json


# ============================================================================
# PIXEL-LEVEL METRICS (Segmentation)
# ============================================================================

class SegmentationMetrics:
    """Calculate segmentation metrics: Precision, Recall, IoU, F1."""
    
    def __init__(self, num_classes=1, threshold=0.5):
        """
        Args:
            num_classes: Number of output classes
            threshold: Threshold for binary classification (for num_classes=1)
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives
        self.tn = 0  # True negatives
    
    def update(self, pred_logits, target):
        """
        Update metrics with batch.
        
        Args:
            pred_logits: (batch, num_classes, H, W) or (batch, 1, H, W)
            target: (batch, H, W) ground truth
        """
        # Convert logits to predictions
        if self.num_classes == 1:
            # Binary classification
            pred = (torch.sigmoid(pred_logits) > self.threshold).long().squeeze(1)
        else:
            # Multi-class
            pred = torch.argmax(pred_logits, dim=1)
        
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # Calculate TP, FP, FN, TN
        self.tp += np.sum((pred == 1) & (target == 1))
        self.fp += np.sum((pred == 1) & (target == 0))
        self.fn += np.sum((pred == 0) & (target == 1))
        self.tn += np.sum((pred == 0) & (target == 0))
    
    def get_metrics(self):
        """
        Calculate and return all metrics.
        
        Returns:
            dict with precision, recall, f1, iou, accuracy
        """
        # Avoid division by zero
        epsilon = 1e-7
        
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        iou = self.tp / (self.tp + self.fp + self.fn + epsilon)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + epsilon)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'iou': float(iou),
            'accuracy': float(accuracy)
        }


def calculate_metrics_batch(pred_logits, target, num_classes=1, threshold=0.5):
    """
    Calculate metrics for a single batch.
    
    Args:
        pred_logits: (batch, num_classes, H, W) or (batch, 1, H, W)
        target: (batch, H, W)
        num_classes: Number of classes
        threshold: Threshold for binary
    
    Returns:
        dict with metrics
    """
    if num_classes == 1:
        pred = (torch.sigmoid(pred_logits) > threshold).long().squeeze(1)
    else:
        pred = torch.argmax(pred_logits, dim=1)
    
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    # For binary only
    if np.unique(target).max() <= 1:
        precision = precision_score(target, pred, zero_division=0)
        recall = recall_score(target, pred, zero_division=0)
        f1 = f1_score(target, pred, zero_division=0)
        iou = jaccard_score(target, pred, zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou
        }
    
    return {}


# ============================================================================
# DRIFT PREDICTION METRICS
# ============================================================================

class DriftMetrics:
    """Calculate drift forecast accuracy metrics."""
    
    def __init__(self):
        """Initialize drift metrics."""
        self.errors = []  # Geodesic distances in km
    
    def add_error(self, predicted_position, observed_position):
        """
        Add a drift prediction error.
        
        Args:
            predicted_position: (lat, lon) tuple
            observed_position: (lat, lon) tuple
        """
        error_km = self.geodesic_distance(predicted_position, observed_position)
        self.errors.append(error_km)
    
    @staticmethod
    def geodesic_distance(pos1, pos2):
        """
        Calculate geodesic distance between two positions using Haversine formula.
        
        Args:
            pos1: (lat, lon) in degrees
            pos2: (lat, lon) in degrees
        
        Returns:
            Distance in kilometers
        """
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1 = radians(pos1[0]), radians(pos1[1])
        lat2, lon2 = radians(pos2[0]), radians(pos2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        R = 6371  # Earth radius in km
        return R * c
    
    def get_metrics(self):
        """
        Calculate drift forecast skill metrics.
        
        Returns:
            dict with mean_error, median_error, std_error
        """
        if len(self.errors) == 0:
            return {'mean_error_km': 0, 'median_error_km': 0, 'std_error_km': 0}
        
        errors = np.array(self.errors)
        
        return {
            'mean_error_km': float(np.mean(errors)),
            'median_error_km': float(np.median(errors)),
            'std_error_km': float(np.std(errors)),
            'num_predictions': len(errors)
        }


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_comprehensive(model, test_loader, device, num_classes=1):
    """
    Comprehensive evaluation on test set.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Device to evaluate on
        num_classes: Number of output classes
    
    Returns:
        dict with all evaluation metrics
    """
    from tqdm import tqdm
    from unet_baseline import combined_loss
    
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes)
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            total_loss += loss.item()
            
            # Update metrics
            metrics.update(outputs, masks)
    
    # Get final metrics
    seg_metrics = metrics.get_metrics()
    
    return {
        'segmentation': seg_metrics,
        'avg_loss': total_loss / len(test_loader)
    }


def save_metrics(metrics, filepath='evaluation_metrics.json'):
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics: dict with metrics
        filepath: Path to save JSON
    """
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to {filepath}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f" EVALUATION RESULTS")
    print(f"{'='*70}")
    if 'segmentation' in metrics:
        print(f"\nSegmentation Metrics:")
        for key, value in metrics['segmentation'].items():
            print(f"  {key}: {value:.4f}")
    if 'avg_loss' in metrics:
        print(f"\nAverage Test Loss: {metrics['avg_loss']:.4f}")
    print(f"\n{'='*70}\n")
