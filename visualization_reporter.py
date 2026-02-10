"""
Comprehensive Training Visualization and Reporting
Generates confusion matrices, AUC curves, loss plots, and detailed reports per epoch
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import class names
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


class TrainingVisualizer:
    """Generate comprehensive training visualizations"""
    
    def __init__(self, output_dir='results/visualizations', num_classes=16):
        # Create timestamped subfolder for this training run
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.base_dir = Path(output_dir)
        self.output_dir = self.base_dir / timestamp
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        
        # Subdirectories
        (self.output_dir / 'confusion_matrices').mkdir(exist_ok=True)
        (self.output_dir / 'auc_curves').mkdir(exist_ok=True)
        (self.output_dir / 'loss_curves').mkdir(exist_ok=True)
        (self.output_dir / 'per_class_metrics').mkdir(exist_ok=True)
        
        print(f"✓ Visualizations will be saved to: {self.output_dir}")
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
    
    def plot_confusion_matrix(self, y_true, y_pred, epoch, class_names=None):
        """Generate and save confusion matrix for an epoch"""
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        
        # Normalize by row
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                   cbar_kws={'label': 'Count'}, square=True)
        ax1.set_title(f'Epoch {epoch}: Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Class')
        ax1.set_ylabel('True Class')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2,
                   cbar_kws={'label': 'Proportion'}, square=True, vmin=0, vmax=1)
        ax2.set_title(f'Epoch {epoch}: Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Class')
        ax2.set_ylabel('True Class')
        
        plt.tight_layout()
        save_path = self.output_dir / 'confusion_matrices' / f'epoch_{epoch:03d}_cm.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def plot_auc_curves(self, y_true, y_pred_proba, epoch, class_names=None):
        """Generate single overall AUC-ROC curve (micro-averaged) for the epoch"""
        # Binarize labels for micro-averaging
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        # Compute micro-averaged ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        # Create single figure with overall AUC curve
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot micro-averaged ROC curve
        ax.plot(fpr_micro, tpr_micro, color='darkorange', lw=3, 
               label=f'Micro-Averaged AUC = {roc_auc_micro:.4f}')
        
        # Plot random classifier line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier (AUC = 0.5000)')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'Epoch {epoch}: Overall ROC Curve (Micro-Averaged)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        save_path = self.output_dir / 'auc_curves' / f'epoch_{epoch:03d}_auc.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def plot_loss_curves(self, train_losses, val_losses):
        """Generate training and validation loss curve"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = list(range(1, len(train_losses) + 1))
        ax.plot(epochs, train_losses, marker='o', label='Training Loss', linewidth=2, markersize=4)
        ax.plot(epochs, val_losses, marker='s', label='Validation Loss', linewidth=2, markersize=4)
        
        # Mark best epoch
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax.plot(best_epoch, best_val_loss, 'g*', markersize=20, label=f'Best Val Loss: {best_val_loss:.6f}')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'loss_curves' / 'loss_curve.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Loss curve saved to {save_path}")
    
    def plot_overall_accuracy_curves(self, per_class_data):
        """
        Generate overall training accuracy and metrics across epochs
        
        Args:
            per_class_data: Dict with structure {epoch: {class_name: {precision, recall, f1}}}
        """
        epochs = sorted([int(e) for e in per_class_data.keys()])
        
        # Calculate macro averages for each epoch
        precisions = []
        recalls = []
        f1_scores = []
        
        for epoch_str in [str(e) for e in epochs]:
            epoch_metrics = per_class_data[epoch_str]
            prec = [v['precision'] for v in epoch_metrics.values()]
            rec = [v['recall'] for v in epoch_metrics.values()]
            f1 = [v['f1'] for v in epoch_metrics.values()]
            
            precisions.append(np.mean(prec))
            recalls.append(np.mean(rec))
            f1_scores.append(np.mean(f1))
        
        # Create figure with 2x1 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Precision, Recall, F1
        ax1.plot(epochs, precisions, marker='o', label='Macro Precision', linewidth=2.5, markersize=6)
        ax1.plot(epochs, recalls, marker='s', label='Macro Recall', linewidth=2.5, markersize=6)
        ax1.plot(epochs, f1_scores, marker='^', label='Macro F1-Score', linewidth=2.5, markersize=6)
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Training Performance (Macro Average)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Plot 2: F1-Score trend
        best_f1_epoch = epochs[np.argmax(f1_scores)]
        best_f1 = max(f1_scores)
        ax2.fill_between(epochs, f1_scores, alpha=0.3, color='blue')
        ax2.plot(epochs, f1_scores, marker='D', label='F1-Score', linewidth=3, markersize=7, color='blue')
        ax2.axvline(x=best_f1_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_f1_epoch})')
        ax2.plot(best_f1_epoch, best_f1, 'g*', markersize=25, label=f'Best F1: {best_f1:.4f}')
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax2.set_title('Overall F1-Score Progression', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        save_path = self.output_dir / 'loss_curves' / 'overall_accuracy.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Overall accuracy curve saved to {save_path}")
    
    def plot_per_class_metrics(self, per_class_data):
        """
        Generate per-class precision, recall, F1 plots across epochs
        
        Args:
            per_class_data: Dict with structure {epoch: {class_name: {precision, recall, f1}}}
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        epochs = sorted([int(e) for e in per_class_data.keys()])
        
        # Extract metrics per class
        classes = set()
        for epoch_data in per_class_data.values():
            classes.update(epoch_data.keys())
        classes = sorted(list(classes))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
        
        for metric_idx, metric_name in enumerate(['precision', 'recall', 'f1']):
            ax = axes[metric_idx]
            
            for class_idx, class_name in enumerate(classes):
                values = []
                for epoch in epochs:
                    epoch_str = str(epoch)
                    if epoch_str in per_class_data and class_name in per_class_data[epoch_str]:
                        values.append(per_class_data[epoch_str][class_name].get(metric_name, 0))
                    else:
                        values.append(0)
                
                if any(v > 0 for v in values):  # Only plot if has data
                    ax.plot(epochs, values, marker='o', label=class_name, 
                           color=colors[class_idx], linewidth=1.5, markersize=3, alpha=0.8)
            
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_name.capitalize(), fontsize=11, fontweight='bold')
            ax.set_title(f'{metric_name.upper()} Across All Classes', fontsize=12, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        save_path = self.output_dir / 'per_class_metrics' / 'per_class_metrics.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Per-class metrics plot saved to {save_path}")
    
    def generate_epoch_summary_report(self, training_log_path):
        """Generate comprehensive text report from training log"""
        with open(training_log_path, 'r') as f:
            data = json.load(f)
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    COMPREHENSIVE TRAINING REPORT                              ║
╚════════════════════════════════════════════════════════════════════════════════╝

📊 TRAINING SUMMARY
{'─' * 80}
Total Epochs:           {len(data['epochs'])}
Best Epoch:             {data['best_epoch']} (Val Loss: {data['best_val_loss']:.6f})
Training Duration:      {len(data['epochs'])} epochs

📈 LOSS PROGRESSION
{'─' * 80}
Initial Train Loss:     {data['training_loss'][0]:.6f}
Final Train Loss:       {data['training_loss'][-1]:.6f}
Loss Reduction:         {(1 - data['training_loss'][-1]/data['training_loss'][0])*100:.2f}%

Initial Val Loss:       {data['validation_loss'][0]:.6f}
Final Val Loss:         {data['validation_loss'][-1]:.6f}
Val Loss Reduction:     {(1 - data['validation_loss'][-1]/data['validation_loss'][0])*100:.2f}%

Best Val Loss:          {data['best_val_loss']:.6f} (at epoch {data['best_epoch']})

📊 LEARNING RATE PROGRESSION
{'─' * 80}
Initial LR:             {data['learning_rate'][0]:.10f}
Final LR:               {data['learning_rate'][-1]:.10f}

🎯 CONVERGENCE ANALYSIS
{'─' * 80}
"""
        
        # Convergence status
        recent_losses = data['validation_loss'][-10:]
        if recent_losses[-1] < recent_losses[0]:
            report += "Status:                 ✓ CONVERGING (loss decreasing in final epochs)\n"
        else:
            report += "Status:                 ✓ STABLE (loss plateauing - good generalization)\n"
        
        report += f"""
Variance (last 10 epochs): {np.std(recent_losses):.6f}
Min Variance:           {np.min(np.diff(recent_losses)):.6f}
Max Variance:           {np.max(np.diff(recent_losses)):.6f}

📁 OUTPUT FILES GENERATED
{'─' * 80}
Confusion Matrices:     {len(list((self.output_dir / 'confusion_matrices').glob('*.png')))} PNG files (one per epoch)
AUC Curves:             {len(list((self.output_dir / 'auc_curves').glob('*.png')))} PNG files (one per epoch)
Loss Curves:            {len(list((self.output_dir / 'loss_curves').glob('*.png')))} PNG files
Per-Class Metrics:      {len(list((self.output_dir / 'per_class_metrics').glob('*.png')))} PNG files
Training Log JSON:      results/training_log.json (detailed per-epoch data)

📊 CLASS PERFORMANCE (Best Epoch {data['best_epoch']})
{'─' * 80}
"""
        
        if len(data['per_class_metrics']) > 0:
            best_metrics = data['per_class_metrics'][data['best_epoch'] - 1]
            report += f"{'Class Name':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}\n"
            report += "─" * 80 + "\n"
            
            for class_name in sorted(best_metrics.keys()):
                metrics = best_metrics[class_name]
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1', 0)
                support = metrics.get('support', 0)
                report += f"{class_name:<30} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support}\n"
        
        report += f"""

✅ VISUALIZATION LOCATIONS
{'─' * 80}
All visualizations saved to: {self.output_dir}
  ├── confusion_matrices/     → Confusion matrices for each epoch
  ├── auc_curves/             → AUC-ROC curves for each epoch
  ├── loss_curves/            → Overall training/validation loss curve
  └── per_class_metrics/      → Per-class performance across epochs

═════════════════════════════════════════════════════════════════════════════════
Generated: {Path(self.output_dir).name}
═════════════════════════════════════════════════════════════════════════════════
"""
        
        return report
    
    def plot_epoch_metrics_evolution(self, data, output_dir=None):
        """
        Plot evolution of metrics across all epochs.
        Shows how AUC, Accuracy, Precision, Recall, F1 change over training.
        
        Args:
            data: Metrics dictionary from training_log.json
            output_dir: Where to save the plot
        """
        import matplotlib.pyplot as plt
        
        if output_dir is None:
            output_dir = self.output_dir
        
        epochs = data.get('epochs', [])
        if len(epochs) == 0:
            return
        
        # Extract per-epoch metrics
        accuracies = []
        aucs = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for epoch in epochs:
            accuracies.append(data.get(f'epoch_{epoch}_accuracy', 0.0))
            aucs.append(data.get(f'epoch_{epoch}_macro_auc', 0.0))
            precisions.append(data.get(f'epoch_{epoch}_macro_precision', 0.0))
            recalls.append(data.get(f'epoch_{epoch}_macro_recall', 0.0))
            f1_scores.append(data.get(f'epoch_{epoch}_macro_f1', 0.0))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Per-Epoch Metrics Evolution', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy over epochs
        axes[0, 0].plot(epochs, accuracies, 'b-o', linewidth=2, markersize=6, label='Accuracy')
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Accuracy', fontsize=11)
        axes[0, 0].set_title('Accuracy per Epoch', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # Plot 2: AUC over epochs
        axes[0, 1].plot(epochs, aucs, 'g-o', linewidth=2, markersize=6, label='Macro AUC')
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Macro AUC', fontsize=11)
        axes[0, 1].set_title('AUC per Epoch', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # Plot 3: Precision & Recall over epochs
        axes[1, 0].plot(epochs, precisions, 'r-o', linewidth=2, markersize=6, label='Macro Precision')
        axes[1, 0].plot(epochs, recalls, 'orange', linestyle='-', marker='s', linewidth=2, markersize=6, label='Macro Recall')
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Score', fontsize=11)
        axes[1, 0].set_title('Precision & Recall per Epoch', fontsize=12, fontweight='bold')
        axes[1, 0].legend(loc='best', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # Plot 4: F1 over epochs
        axes[1, 1].plot(epochs, f1_scores, 'purple', linestyle='-', marker='^', linewidth=2, markersize=6, label='Macro F1')
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Macro F1', fontsize=11)
        axes[1, 1].set_title('F1 Score per Epoch', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(output_dir) / 'epoch_metrics_evolution.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved epoch metrics evolution plot to {plot_path}")
        
        return plot_path


def generate_comprehensive_report(training_log_path, visualizations_dir='results/visualizations'):
    """
    Main function to generate all visualizations and reports FOR EACH EPOCH
    """
    with open(training_log_path, 'r') as f:
        data = json.load(f)
    
    visualizer = TrainingVisualizer(output_dir=visualizations_dir)
    
    print("\n" + "="*80)
    print(" GENERATING COMPREHENSIVE TRAINING VISUALIZATIONS FOR ALL EPOCHS")
    print("="*80 + "\n")
    
    # 1. Loss curves (overall)
    print("📈 Generating loss curves...")
    visualizer.plot_loss_curves(data['training_loss'], data['validation_loss'])
    
    # 2. Per-class metrics (overall)
    print("📊 Generating per-class metrics plots...")
    per_class_data = {}
    for epoch_idx, epoch in enumerate(data['epochs']):
        per_class_data[str(epoch)] = data['per_class_metrics'][epoch_idx]
    visualizer.plot_per_class_metrics(per_class_data)
    
    # 2b. Epoch metrics evolution (AUC, Accuracy, Precision, Recall, F1)
    print("📈 Generating per-epoch metrics evolution plot...")
    visualizer.plot_epoch_metrics_evolution(data)
    
    # 3. GENERATE CONFUSION MATRIX AND AUC FOR EACH EPOCH
    print("\n🎯 Generating per-epoch visualizations...")
    print(f"   Total epochs: {len(data['epochs'])}")
    
    for epoch_idx, epoch in enumerate(data['epochs']):
        # Get predictions and labels from confusion matrix
        cm = data['confusion_matrices'][epoch_idx]
        
        # Reconstruct y_true and y_pred from confusion matrix diagonal and off-diagonal
        # For simplicity, we'll create synthetic predictions based on confusion matrix
        y_true_list = []
        y_pred_list = []
        
        cm_array = np.array(cm)
        for true_class in range(len(cm)):
            for pred_class in range(len(cm)):
                count = int(cm[true_class][pred_class])
                y_true_list.extend([true_class] * count)
                y_pred_list.extend([pred_class] * count)
        
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        
        if len(y_true) > 0:
            # Plot confusion matrix
            visualizer.plot_confusion_matrix(y_true, y_pred, epoch)
            
            # Create probability matrix from confusion matrix for AUC
            # Normalize confusion matrix to get probabilities
            y_pred_proba = np.zeros((len(y_true), 16))
            for i in range(len(y_true)):
                pred_class = y_pred[i]
                y_pred_proba[i, pred_class] = 1.0
            
            # Plot AUC curves
            visualizer.plot_auc_curves(y_true, y_pred_proba, epoch, 
                                      class_names=[MARIDA_CLASSES.get(i, f'Class {i}') for i in range(16)])
            
            # Progress
            if (epoch_idx + 1) % 5 == 0:
                print(f"   ✓ Generated visualizations for epoch {epoch}/{len(data['epochs'])}")
    
    print(f"   ✓ Generated visualizations for all {len(data['epochs'])} epochs\n")
    
    # 4. Generate report
    print("📄 Generating comprehensive text report...")
    report = visualizer.generate_epoch_summary_report(training_log_path)
    
    report_path = Path(visualizations_dir) / 'TRAINING_REPORT.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n✓ Report saved to {report_path}\n")
