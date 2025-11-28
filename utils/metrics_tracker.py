"""
Comprehensive Metrics Tracking and Visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from pathlib import Path
import json


class MetricsTracker:
    """Track and visualize all training metrics"""
    
    def __init__(self, num_classes=6, class_names=None, save_dir='outputs'):
        self.num_classes = num_classes
        self.class_names = class_names or [
            'compact_sources', 'elliptical_galaxies', 'fr2_radio_galaxies',
            'radio_loud_agn', 'spiral_galaxies', 'starburst_galaxies'
        ]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Epoch-level metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        
        # Per-class metrics (validation)
        self.val_precisions = {cls: [] for cls in self.class_names}
        self.val_recalls = {cls: [] for cls in self.class_names}
        self.val_f1s = {cls: [] for cls in self.class_names}
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr, per_class_metrics=None):
        """Update metrics for current epoch"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)
        
        # Update per-class metrics
        if per_class_metrics:
            for i, cls in enumerate(self.class_names):
                self.val_precisions[cls].append(per_class_metrics['precision'][i])
                self.val_recalls[cls].append(per_class_metrics['recall'][i])
                self.val_f1s[cls].append(per_class_metrics['f1'][i])
        
        # Update best
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.best_val_loss = val_loss
    
    def plot_training_curves(self):
        """Plot comprehensive training curves"""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        epochs = range(1, len(self.train_losses) + 1)
        
        # 1. Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2, marker='o', markersize=3)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accuracy curves
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='Train', linewidth=2, marker='o', markersize=3)
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
        axes[0, 1].axhline(y=self.best_val_acc, color='g', linestyle='--', 
                          label=f'Best: {self.best_val_acc:.2%} (Epoch {self.best_epoch})', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # 3. Learning rate
        axes[0, 2].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch', fontsize=12)
        axes[0, 2].set_ylabel('Learning Rate', fontsize=12)
        axes[0, 2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Overfitting gap
        gap = [train - val for train, val in zip(self.train_accs, self.val_accs)]
        axes[1, 0].plot(epochs, gap, 'm-', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].fill_between(epochs, gap, 0, where=[g > 0 for g in gap], 
                                alpha=0.3, color='red', label='Overfitting')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Train Acc - Val Acc', fontsize=12)
        axes[1, 0].set_title('Overfitting Monitor', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Loss ratio
        loss_ratio = [train / (val + 1e-8) for train, val in zip(self.train_losses, self.val_losses)]
        axes[1, 1].plot(epochs, loss_ratio, 'c-', linewidth=2)
        axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Balanced')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Train Loss / Val Loss', fontsize=12)
        axes[1, 1].set_title('Loss Ratio (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Summary stats
        axes[1, 2].axis('off')
        summary_text = f"""
        TRAINING SUMMARY
        
        Best Validation Accuracy: {self.best_val_acc:.2%}
        Best Epoch: {self.best_epoch}
        Best Validation Loss: {self.best_val_loss:.4f}
        
        Final Train Acc: {self.train_accs[-1]:.2%}
        Final Val Acc: {self.val_accs[-1]:.2%}
        Final Train Loss: {self.train_losses[-1]:.4f}
        Final Val Loss: {self.val_losses[-1]:.4f}
        
        Total Epochs: {len(epochs)}
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.save_dir / 'training_curves.png'}")
    
    def plot_per_class_metrics(self):
        """Plot per-class performance metrics"""
        if not self.val_f1s[self.class_names[0]]:
            return
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        epochs = range(1, len(self.val_f1s[self.class_names[0]]) + 1)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
        
        for i, (metric_name, metric_dict) in enumerate([
            ('Precision', self.val_precisions),
            ('Recall', self.val_recalls),
            ('F1-Score', self.val_f1s)
        ]):
            for j, cls in enumerate(self.class_names):
                if metric_dict[cls]:
                    axes[i].plot(epochs, metric_dict[cls], 
                               label=cls.replace('_', ' ').title(),
                               linewidth=2, marker='o', markersize=4, color=colors[j])
            
            axes[i].set_xlabel('Epoch', fontsize=12)
            axes[i].set_ylabel(metric_name, fontsize=12)
            axes[i].set_title(f'Validation {metric_name} per Class', fontsize=14, fontweight='bold')
            axes[i].legend(fontsize=9, loc='best')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.save_dir / 'per_class_metrics.png'}")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=[c.replace('_', '\n').title() for c in self.class_names],
                   yticklabels=[c.replace('_', '\n').title() for c in self.class_names],
                   cbar_kws={'label': 'Count'})
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        
        # Normalized
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                   xticklabels=[c.replace('_', '\n').title() for c in self.class_names],
                   yticklabels=[c.replace('_', '\n').title() for c in self.class_names],
                   cbar_kws={'label': 'Percentage'})
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.save_dir / 'confusion_matrix.png'}")
    
    def save_metrics(self):
        """Save all metrics to JSON"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'learning_rates': self.learning_rates,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'per_class': {
                'precision': {k: v for k, v in self.val_precisions.items()},
                'recall': {k: v for k, v in self.val_recalls.items()},
                'f1': {k: v for k, v in self.val_f1s.items()}
            }
        }
        
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Saved: {self.save_dir / 'metrics.json'}")
    
    def save_classification_report(self, y_true, y_pred):
        """Save detailed classification report"""
        report = classification_report(
            y_true, y_pred,
            target_names=[c.replace('_', ' ').title() for c in self.class_names],
            digits=4
        )
        
        with open(self.save_dir / 'classification_report.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Best Epoch: {self.best_epoch}\n")
            f.write(f"Best Val Accuracy: {self.best_val_acc:.4%}\n")
            f.write(f"Best Val Loss: {self.best_val_loss:.4f}\n\n")
            f.write("="*60 + "\n")
            f.write(report)
            f.write("\n" + "="*60 + "\n")
        
        print(f"✓ Saved: {self.save_dir / 'classification_report.txt'}")
        
        return report
