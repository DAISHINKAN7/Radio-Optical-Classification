"""
Comprehensive Model Testing & Analysis
- Detailed metrics
- Confusion matrices
- ROC curves
- Sample predictions
- Error analysis
- Confidence analysis
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import LoTSSDataset
from data.preprocessing import LoTSSPreprocessing
from data.augmentation import CombinedTransform
from models.resnet_model import ResNet34_LoTSS
from models.efficientnet_model import EfficientNetB0_LoTSS
from models.vit_model import ViT_LoTSS
from models.convnext_model import ConvNeXtTiny_LoTSS


class ModelTester:
    """Comprehensive model testing and analysis"""
    
    def __init__(self, model, model_name, checkpoint_path, val_loader, device, output_dir):
        self.model = model
        self.model_name = model_name
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = [
            'Compact Sources',
            'Elliptical Galaxies',
            'FR2 Radio Galaxies',
            'Radio Loud AGN',
            'Spiral Galaxies',
            'Starburst Galaxies'
        ]
        
        # Load checkpoint
        print(f"\n{'='*60}")
        print(f"Loading {model_name}")
        print(f"{'='*60}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"✓ Loaded checkpoint from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Acc: {checkpoint.get('val_acc', 'N/A'):.2%}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        # Storage
        self.results = {
            'predictions': [],
            'labels': [],
            'probabilities': [],
            'correct': [],
            'samples': []
        }
    
    def test(self):
        """Run complete testing pipeline"""
        print(f"\n{'='*60}")
        print(f"TESTING {self.model_name}")
        print(f"{'='*60}\n")
        
        # 1. Collect predictions
        print("1. Collecting predictions...")
        self._collect_predictions()
        
        # 2. Calculate metrics
        print("\n2. Calculating metrics...")
        metrics = self._calculate_metrics()
        
        # 3. Generate visualizations
        print("\n3. Generating visualizations...")
        self._plot_confusion_matrix()
        self._plot_per_class_metrics()
        self._plot_roc_curves()
        self._plot_precision_recall_curves()
        self._plot_confidence_distribution()
        self._plot_sample_predictions()
        self._analyze_errors()
        
        # 4. Save results
        print("\n4. Saving results...")
        self._save_results(metrics)
        
        print(f"\n{'='*60}")
        print(f"TESTING COMPLETE!")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def _collect_predictions(self):
        """Collect all predictions and probabilities"""
        with torch.no_grad():
            for images, labels, metadata in tqdm(self.val_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Store results
                self.results['predictions'].extend(preds.cpu().numpy())
                self.results['labels'].extend(labels.cpu().numpy())
                self.results['probabilities'].extend(probs.cpu().numpy())
                self.results['correct'].extend((preds == labels).cpu().numpy())
                self.results['samples'].extend([
                    {
                        'image': images[i].cpu().numpy(),
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item(),
                        'confidence': probs[i].max().item(),
                        'stem': metadata['stem'][i]
                    }
                    for i in range(len(labels))
                ])
        
        # Convert to numpy
        self.results['predictions'] = np.array(self.results['predictions'])
        self.results['labels'] = np.array(self.results['labels'])
        self.results['probabilities'] = np.array(self.results['probabilities'])
        self.results['correct'] = np.array(self.results['correct'])
    
    def _calculate_metrics(self):
        """Calculate comprehensive metrics"""
        y_true = self.results['labels']
        y_pred = self.results['predictions']
        y_prob = self.results['probabilities']
        
        # Overall accuracy
        accuracy = (y_true == y_pred).mean()
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro/weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'per_class': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'support': support.tolist()
            },
            'macro_avg': {
                'precision': precision_macro,
                'recall': recall_macro,
                'f1': f1_macro
            },
            'weighted_avg': {
                'precision': precision_weighted,
                'recall': recall_weighted,
                'f1': f1_weighted
            }
        }
        
        return metrics
    
    def _plot_confusion_matrix(self):
        """Plot confusion matrix"""
        y_true = self.results['labels']
        y_pred = self.results['predictions']
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_title(f'{self.model_name} - Confusion Matrix (Counts)', 
                         fontsize=14, fontweight='bold')
        
        # Normalized
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Percentage'})
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_title(f'{self.model_name} - Confusion Matrix (Normalized)', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: confusion_matrix.png")
    
    def _plot_per_class_metrics(self):
        """Plot per-class performance metrics"""
        y_true = self.results['labels']
        y_pred = self.results['predictions']
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.model_name} - Per-Class Metrics', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: per_class_metrics.png")
    
    def _plot_roc_curves(self):
        """Plot ROC curves for each class"""
        y_true = self.results['labels']
        y_prob = self.results['probabilities']
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        
        # Plot ROC curve for each class
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{self.model_name} - ROC Curves', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: roc_curves.png")
    
    def _plot_precision_recall_curves(self):
        """Plot Precision-Recall curves"""
        y_true = self.results['labels']
        y_prob = self.results['probabilities']
        
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, color=color, lw=2,
                   label=f'{class_name} (AP = {pr_auc:.3f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'{self.model_name} - Precision-Recall Curves', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: precision_recall_curves.png")
    
    def _plot_confidence_distribution(self):
        """Plot confidence distribution for correct vs incorrect predictions"""
        correct_confidences = [s['confidence'] for s in self.results['samples'] if s['true_label'] == s['pred_label']]
        incorrect_confidences = [s['confidence'] for s in self.results['samples'] if s['true_label'] != s['pred_label']]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
        axes[0].hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        axes[0].set_xlabel('Confidence', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Box plot
        axes[1].boxplot([correct_confidences, incorrect_confidences],
                       labels=['Correct', 'Incorrect'],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
        axes[1].set_ylabel('Confidence', fontsize=12)
        axes[1].set_title('Confidence Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add statistics
        axes[1].text(0.02, 0.98, 
                    f"Correct: μ={np.mean(correct_confidences):.3f}, σ={np.std(correct_confidences):.3f}\n"
                    f"Incorrect: μ={np.mean(incorrect_confidences):.3f}, σ={np.std(incorrect_confidences):.3f}",
                    transform=axes[1].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: confidence_distribution.png")
    
    def _plot_sample_predictions(self, num_samples=20):
        """Plot sample predictions (correct and incorrect)"""
        # Get some correct and incorrect samples
        correct_samples = [s for s in self.results['samples'] if s['true_label'] == s['pred_label']]
        incorrect_samples = [s for s in self.results['samples'] if s['true_label'] != s['pred_label']]
        
        # Sample randomly
        np.random.shuffle(correct_samples)
        np.random.shuffle(incorrect_samples)
        
        samples_to_plot = correct_samples[:num_samples//2] + incorrect_samples[:num_samples//2]
        
        num_cols = 5
        num_rows = (len(samples_to_plot) + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4*num_rows))
        axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes
        
        for idx, sample in enumerate(samples_to_plot[:len(axes)]):
            ax = axes[idx]
            
            # Show optical channel (first channel)
            img = sample['image'][0]  # Optical channel
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            true_class = self.class_names[sample['true_label']]
            pred_class = self.class_names[sample['pred_label']]
            conf = sample['confidence']
            
            color = 'green' if sample['true_label'] == sample['pred_label'] else 'red'
            
            ax.set_title(f"True: {true_class}\nPred: {pred_class}\nConf: {conf:.2%}",
                        fontsize=9, color=color, fontweight='bold')
        
        # Hide remaining axes
        for idx in range(len(samples_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{self.model_name} - Sample Predictions', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_predictions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: sample_predictions.png")
    
    def _analyze_errors(self):
        """Analyze common error patterns"""
        y_true = self.results['labels']
        y_pred = self.results['predictions']
        
        # Confusion pairs
        errors = []
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                errors.append((true_label, pred_label))
        
        # Count error pairs
        from collections import Counter
        error_counts = Counter(errors)
        
        # Plot most common errors
        if error_counts:
            most_common = error_counts.most_common(10)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            labels = [f"{self.class_names[t]} →\n{self.class_names[p]}" 
                     for (t, p), _ in most_common]
            counts = [count for _, count in most_common]
            
            bars = ax.barh(labels, counts, color='coral', edgecolor='black')
            ax.set_xlabel('Number of Errors', fontsize=12)
            ax.set_title(f'{self.model_name} - Most Common Misclassifications', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{int(width)}',
                       ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'error_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: error_analysis.png")
    
    def _save_results(self, metrics):
        """Save all results to JSON and text files"""
        # Save metrics
        with open(self.output_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save classification report
        y_true = self.results['labels']
        y_pred = self.results['predictions']
        
        report = classification_report(y_true, y_pred, 
                                      target_names=self.class_names,
                                      digits=4)
        
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"{self.model_name} - TEST RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4%}\n\n")
            f.write("="*60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n")
            f.write(report)
            f.write("\n" + "="*60 + "\n")
            f.write("PER-CLASS DETAILED METRICS\n")
            f.write("="*60 + "\n\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['per_class']['precision'][i]:.4f}\n")
                f.write(f"  Recall:    {metrics['per_class']['recall'][i]:.4f}\n")
                f.write(f"  F1-Score:  {metrics['per_class']['f1'][i]:.4f}\n")
                f.write(f"  Support:   {metrics['per_class']['support'][i]}\n\n")
        
        print(f"  ✓ Saved: test_metrics.json")
        print(f"  ✓ Saved: classification_report.txt")


def load_model(model_name, checkpoint_path, device):
    """Load appropriate model based on name"""
    models = {
        'resnet': ResNet34_LoTSS(num_classes=6, pretrained=False),
        'efficientnet': EfficientNetB0_LoTSS(num_classes=6, pretrained=False),
        'vit': ViT_LoTSS(num_classes=6),
        'convnext': ConvNeXtTiny_LoTSS(num_classes=6, pretrained=False)
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")
    
    return models[model_name.lower()]


def main():
    parser = argparse.ArgumentParser(description='Test LoTSS Model')
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet', 'efficientnet', 'vit', 'convnext'],
                       help='Model to test')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/beautiful_dataset_v2')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    # Load validation dataset
    preprocessing = LoTSSPreprocessing(target_size=(600, 600))
    val_transform = CombinedTransform(preprocessing, None)
    
    val_dataset = LoTSSDataset(
        args.data_path,
        split='val',
        transform=val_transform,
        val_split=0.2,
        seed=args.seed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load model
    model = load_model(args.model, args.checkpoint, device)
    
    # Create tester
    tester = ModelTester(
        model=model,
        model_name=args.model.capitalize(),
        checkpoint_path=args.checkpoint,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir
    )
    
    # Run testing
    metrics = tester.test()


if __name__ == "__main__":
    main()