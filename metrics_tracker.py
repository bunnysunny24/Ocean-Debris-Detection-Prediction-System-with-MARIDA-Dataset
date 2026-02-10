"""
Comprehensive Metrics Tracking for Training
Tracks per-epoch metrics, confusion matrices, per-class performance
"""

import json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score
from pathlib import Path


class MetricsTracker:
    """Track training metrics across epochs"""
    
    def __init__(self, num_classes=16, save_path='results/training_log.json'):
        self.num_classes = num_classes
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'epochs': [],
            'training_loss': [],
            'validation_loss': [],
            'learning_rate': [],
            'per_class_metrics': [],  # List of dicts per epoch
            'confusion_matrices': [],  # List of matrices per epoch
            'best_epoch': None,
            'best_val_loss': float('inf')
        }
    
    def add_epoch(self, epoch, train_loss, val_loss, lr, 
                  y_true, y_pred, class_names=None):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            lr: Learning rate
            y_true: Ground truth labels (flattened)
            y_pred: Predicted labels (flattened)
            class_names: List of class names
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['training_loss'].append(float(train_loss))
        self.metrics['validation_loss'].append(float(val_loss))
        self.metrics['learning_rate'].append(float(lr))
        
        # Track best model
        if val_loss < self.metrics['best_val_loss']:
            self.metrics['best_val_loss'] = float(val_loss)
            self.metrics['best_epoch'] = epoch
        
        # Compute confusion matrix
        if len(y_true) > 0 and len(y_pred) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
            self.metrics['confusion_matrices'].append(cm.tolist())
            
            # Compute per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=list(range(self.num_classes)), 
                average=None, zero_division=0
            )
            
            per_class = {}
            for c in range(self.num_classes):
                class_name = f"class_{c}" if class_names is None else class_names[c]
                per_class[class_name] = {
                    'precision': float(precision[c]),
                    'recall': float(recall[c]),
                    'f1': float(f1[c]),
                    'support': int(support[c])
                }
            
            self.metrics['per_class_metrics'].append(per_class)
            
            # Compute macro-averaged metrics
            macro_precision = np.mean(precision)
            macro_recall = np.mean(recall)
            macro_f1 = np.mean(f1)
            
            # Compute accuracy
            accuracy = accuracy_score(y_true, y_pred)
            
            # Compute per-class AUC (one-vs-rest for each class)
            try:
                # For multiclass, use OvR (One-vs-Rest) AUC
                # Convert to one-hot encoding for each class
                y_true_onehot = np.eye(self.num_classes)[y_true.astype(int)]
                y_pred_probs_soft = np.eye(self.num_classes)[y_pred.astype(int)]
                # Use weighted average of per-class AUCs
                macro_auc = roc_auc_score(y_true_onehot, y_pred_probs_soft, multi_class='ovr', average='macro')
            except:
                macro_auc = 0.0  # If AUC can't be computed, set to 0
            
            self.metrics[f'epoch_{epoch}_macro_precision'] = float(macro_precision)
            self.metrics[f'epoch_{epoch}_macro_recall'] = float(macro_recall)
            self.metrics[f'epoch_{epoch}_macro_f1'] = float(macro_f1)
            self.metrics[f'epoch_{epoch}_accuracy'] = float(accuracy)
            self.metrics[f'epoch_{epoch}_macro_auc'] = float(macro_auc)
    
    def save(self):
        """Save metrics to JSON file"""
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"✓ Metrics saved to {self.save_path}")
    
    def save_text_log(self, text_log_path=None):
        """Save formatted training log as readable text file"""
        if text_log_path is None:
            # Save in same directory as JSON log
            text_log_path = self.save_path.parent / 'training_log.txt'
        
        text_log_path = Path(text_log_path)
        text_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(text_log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING LOG - OCEAN DEBRIS DETECTION (MARIDA 16-CLASS)\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall summary
            f.write("TRAINING SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Epochs: {len(self.metrics['epochs'])}\n")
            f.write(f"Best Epoch: {self.metrics['best_epoch']}\n")
            f.write(f"Best Validation Loss: {self.metrics['best_val_loss']:.6f}\n")
            f.write(f"Initial Train Loss: {self.metrics['training_loss'][0]:.6f}\n")
            f.write(f"Final Train Loss: {self.metrics['training_loss'][-1]:.6f}\n")
            f.write(f"Initial Val Loss: {self.metrics['validation_loss'][0]:.6f}\n")
            f.write(f"Final Val Loss: {self.metrics['validation_loss'][-1]:.6f}\n")
            
            convergence = 'CONVERGING ✓' if self.metrics['validation_loss'][-1] < self.metrics['validation_loss'][0] else 'NOT CONVERGING ✗'
            f.write(f"Status: {convergence}\n\n")
            
            # Per-epoch details
            f.write("PER-EPOCH METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'LR':<12} {'Macro F1':<12}\n")
            f.write("-" * 80 + "\n")
            
            for epoch_idx, epoch in enumerate(self.metrics['epochs']):
                train_loss = self.metrics['training_loss'][epoch_idx]
                val_loss = self.metrics['validation_loss'][epoch_idx]
                lr = self.metrics['learning_rate'][epoch_idx]
                macro_f1 = self.metrics.get(f'epoch_{epoch}_macro_f1', 0.0)
                
                f.write(f"{epoch:<8} {train_loss:<15.6f} {val_loss:<15.6f} {lr:<12.6f} {macro_f1:<12.6f}\n")
            
            f.write("\n")
            
            # Per-class metrics for each epoch
            f.write("PER-CLASS METRICS BY EPOCH\n")
            f.write("-" * 80 + "\n")
            
            for epoch_idx, epoch in enumerate(self.metrics['epochs']):
                if epoch_idx < len(self.metrics['per_class_metrics']):
                    f.write(f"\n{'='*80}\n")
                    f.write(f"EPOCH {epoch}\n")
                    f.write(f"{'='*80}\n")
                    
                    per_class = self.metrics['per_class_metrics'][epoch_idx]
                    f.write(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}\n")
                    f.write("-" * 78 + "\n")
                    
                    for class_name, metrics in per_class.items():
                        precision = metrics['precision']
                        recall = metrics['recall']
                        f1 = metrics['f1']
                        support = metrics['support']
                        
                        f.write(f"{class_name:<30} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<12}\n")
                    
                    # Macro averages
                    f.write("-" * 78 + "\n")
                    macro_precision = self.metrics.get(f'epoch_{epoch}_macro_precision', 0.0)
                    macro_recall = self.metrics.get(f'epoch_{epoch}_macro_recall', 0.0)
                    macro_f1 = self.metrics.get(f'epoch_{epoch}_macro_f1', 0.0)
                    
                    f.write(f"{'MACRO AVERAGE':<30} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF TRAINING LOG\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ Text training log saved to {text_log_path}")
    
    def get_summary(self):
        """Get human-readable summary"""
        if not self.metrics['epochs']:
            return "No metrics recorded yet"
        
        summary = f"""
Metrics Summary
===============
Total Epochs: {len(self.metrics['epochs'])}
Best Epoch: {self.metrics['best_epoch']} (Val Loss: {self.metrics['best_val_loss']:.6f})

Training Progress:
  Initial Train Loss: {self.metrics['training_loss'][0]:.6f}
  Final Train Loss: {self.metrics['training_loss'][-1]:.6f}
  Initial Val Loss: {self.metrics['validation_loss'][0]:.6f}
  Final Val Loss: {self.metrics['validation_loss'][-1]:.6f}

Convergence: {'✓ CONVERGING' if self.metrics['validation_loss'][-1] < self.metrics['validation_loss'][0] else '✗ NOT CONVERGING'}
        """
        return summary


MARIDA_CLASSES = {
    0: 'Background',
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


def get_confusion_matrix_summary(cm, class_names=MARIDA_CLASSES):
    """Create readable confusion matrix summary"""
    summary = "Confusion Matrix:\n"
    summary += f"{'Class':<25} {'Diagonal':<10} {'Total Samples':<15}\n"
    summary += "-" * 50 + "\n"
    
    for i in range(min(len(cm), len(class_names))):
        diagonal = cm[i, i]
        total = cm[i].sum()
        class_name = class_names.get(i, f"Class {i}")
        summary += f"{class_name:<25} {diagonal:<10} {total:<15}\n"
    
    return summary
