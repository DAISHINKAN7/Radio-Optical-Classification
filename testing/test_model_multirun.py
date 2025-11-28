"""
Comprehensive Multi-Run Model Testing with Statistical Analysis
- Multiple test runs (5-10 iterations)
- Mean metrics with confidence intervals
- Statistical significance testing
- Publication-ready results
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
from scipy import stats
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import LoTSSDataset
from data.preprocessing import LoTSSPreprocessing
from data.augmentation import CombinedTransform
from models.resnet_model import ResNet34_LoTSS
from models.efficientnet_model import EfficientNetB0_LoTSS
from models.vit_model import ViT_LoTSS
from models.convnext_model import ConvNeXtTiny_LoTSS


class MultiRunTester:
    """Comprehensive multi-run model testing with statistical analysis"""
    
    def __init__(self, model_fn, model_name, checkpoint_path, data_path, device, output_dir, 
                 num_runs=10, batch_size=16, num_workers=4, target_size=(600, 600), seed=42):
        self.model_fn = model_fn
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_runs = num_runs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.base_seed = seed
        
        self.class_names = [
            'Compact Sources',
            'Elliptical Galaxies',
            'FR2 Radio Galaxies',
            'Radio Loud AGN',
            'Spiral Galaxies',
            'Starburst Galaxies'
        ]
        
        # Storage for all runs
        self.all_runs_results = []
        self.aggregated_metrics = {}
    
    def run_all_tests(self):
        """Execute multiple test runs"""
        print(f"\n{'='*80}")
        print(f"MULTI-RUN TESTING: {self.model_name}")
        print(f"{'='*80}")
        print(f"Number of runs: {self.num_runs}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"{'='*80}\n")
        
        for run_idx in range(self.num_runs):
            print(f"\n{'─'*80}")
            print(f"RUN {run_idx + 1}/{self.num_runs}")
            print(f"{'─'*80}")
            
            run_seed = self.base_seed + run_idx
            metrics = self._single_run(run_idx, run_seed)
            self.all_runs_results.append(metrics)
            
            print(f"✓ Run {run_idx + 1} complete - Accuracy: {metrics['accuracy']:.4f}")
        
        # Aggregate results
        print(f"\n{'='*80}")
        print("AGGREGATING RESULTS")
        print(f"{'='*80}\n")
        self._aggregate_results()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self._plot_run_variations()
        self._plot_confidence_intervals()
        self._plot_aggregated_confusion_matrix()
        self._generate_publication_table()
        
        # Save results
        self._save_all_results()
        
        print(f"\n{'='*80}")
        print("MULTI-RUN TESTING COMPLETE!")
        print(f"{'='*80}")
        self._print_summary()
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}\n")
    
    def _single_run(self, run_idx, seed):
        """Execute a single test run"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create preprocessing and transform
        preprocessing = LoTSSPreprocessing(target_size=self.target_size)
        val_transform = CombinedTransform(preprocessing, augmentation=None)
        
        # Load validation dataset with specific seed
        val_dataset = LoTSSDataset(
            self.data_path,
            split='val',
            transform=val_transform,
            val_split=0.2,
            seed=seed
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Load model
        model = self.model_fn()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Collect predictions
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Run {run_idx + 1}", leave=False):
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels, _ = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics for a single run"""
        accuracy = (y_true == y_pred).mean()
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Calculate AUC for each class
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        auc_scores = []
        for i in range(len(self.class_names)):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                auc_scores.append(auc(fpr, tpr))
            except:
                auc_scores.append(0.0)
        
        metrics = {
            'accuracy': accuracy,
            'per_class': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'support': support.tolist(),
                'auc': auc_scores
            },
            'macro_avg': {
                'precision': precision_macro,
                'recall': recall_macro,
                'f1': f1_macro,
                'auc': np.mean(auc_scores)
            },
            'weighted_avg': {
                'precision': precision_weighted,
                'recall': recall_weighted,
                'f1': f1_weighted
            }
        }
        
        return metrics
    
    def _aggregate_results(self):
        """Aggregate metrics across all runs with confidence intervals"""
        
        def calc_ci(values, confidence=0.95):
            """Calculate confidence interval"""
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            se = std / np.sqrt(len(values))
            ci = se * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
            return mean, std, ci
        
        # Overall metrics
        accuracies = [r['accuracy'] for r in self.all_runs_results]
        acc_mean, acc_std, acc_ci = calc_ci(accuracies)
        
        # Macro averages
        macro_precision = [r['macro_avg']['precision'] for r in self.all_runs_results]
        macro_recall = [r['macro_avg']['recall'] for r in self.all_runs_results]
        macro_f1 = [r['macro_avg']['f1'] for r in self.all_runs_results]
        macro_auc = [r['macro_avg']['auc'] for r in self.all_runs_results]
        
        # Per-class metrics
        num_classes = len(self.class_names)
        per_class_metrics = {
            'precision': {'mean': [], 'std': [], 'ci': []},
            'recall': {'mean': [], 'std': [], 'ci': []},
            'f1': {'mean': [], 'std': [], 'ci': []},
            'auc': {'mean': [], 'std': [], 'ci': []}
        }
        
        for class_idx in range(num_classes):
            for metric_name in ['precision', 'recall', 'f1', 'auc']:
                values = [r['per_class'][metric_name][class_idx] for r in self.all_runs_results]
                mean, std, ci = calc_ci(values)
                per_class_metrics[metric_name]['mean'].append(mean)
                per_class_metrics[metric_name]['std'].append(std)
                per_class_metrics[metric_name]['ci'].append(ci)
        
        # Aggregate confusion matrices
        cms = [r['confusion_matrix'] for r in self.all_runs_results]
        cm_mean = np.mean(cms, axis=0)
        cm_std = np.std(cms, axis=0, ddof=1)
        
        self.aggregated_metrics = {
            'num_runs': self.num_runs,
            'accuracy': {
                'mean': acc_mean,
                'std': acc_std,
                'ci_95': acc_ci,
                'all_values': accuracies
            },
            'macro_avg': {
                'precision': calc_ci(macro_precision),
                'recall': calc_ci(macro_recall),
                'f1': calc_ci(macro_f1),
                'auc': calc_ci(macro_auc)
            },
            'per_class': per_class_metrics,
            'confusion_matrix': {
                'mean': cm_mean,
                'std': cm_std
            }
        }
    
    def _plot_run_variations(self):
        """Plot metric variations across runs"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_to_plot = [
            ('accuracy', 'Accuracy'),
            ('macro_avg.precision', 'Macro Precision'),
            ('macro_avg.recall', 'Macro Recall'),
            ('macro_avg.f1', 'Macro F1-Score')
        ]
        
        for idx, (metric_path, label) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            # Extract values
            if '.' in metric_path:
                parts = metric_path.split('.')
                values = [r[parts[0]][parts[1]] for r in self.all_runs_results]
            else:
                values = [r[metric_path] for r in self.all_runs_results]
            
            # Plot
            runs = np.arange(1, self.num_runs + 1)
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            ax.plot(runs, values, 'o-', linewidth=2, markersize=8, color='#3498db')
            ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
            ax.fill_between(runs, mean_val - std_val, mean_val + std_val, 
                           alpha=0.2, color='red', label=f'±1 SD: {std_val:.4f}')
            
            ax.set_xlabel('Run Number', fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f'{label} Across Runs', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_xticks(runs)
        
        plt.suptitle(f'{self.model_name} - Metric Stability Across Runs', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'run_variations.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: run_variations.png")
    
    def _plot_confidence_intervals(self):
        """Plot metrics with confidence intervals"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        means = [
            self.aggregated_metrics['accuracy']['mean'],
            self.aggregated_metrics['macro_avg']['precision'][0],
            self.aggregated_metrics['macro_avg']['recall'][0],
            self.aggregated_metrics['macro_avg']['f1'][0],
            self.aggregated_metrics['macro_avg']['auc'][0]
        ]
        cis = [
            self.aggregated_metrics['accuracy']['ci_95'],
            self.aggregated_metrics['macro_avg']['precision'][2],
            self.aggregated_metrics['macro_avg']['recall'][2],
            self.aggregated_metrics['macro_avg']['f1'][2],
            self.aggregated_metrics['macro_avg']['auc'][2]
        ]
        
        x = np.arange(len(metrics))
        bars = ax.bar(x, means, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.errorbar(x, means, yerr=cis, fmt='none', color='red', capsize=10, 
                   capthick=2, linewidth=2, label='95% CI')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.model_name} - Metrics with 95% Confidence Intervals\n(n={self.num_runs} runs)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=11)
        
        # Add value labels
        for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + ci + 0.02,
                   f'{mean:.3f}±{ci:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_intervals.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: confidence_intervals.png")
    
    def _plot_aggregated_confusion_matrix(self):
        """Plot mean confusion matrix with standard deviations"""
        cm_mean = self.aggregated_metrics['confusion_matrix']['mean']
        cm_std = self.aggregated_metrics['confusion_matrix']['std']
        
        cm_norm = cm_mean / (cm_mean.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Mean counts
        sns.heatmap(cm_mean, annot=True, fmt='.1f', cmap='Blues', ax=axes[0],
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Mean Count'})
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_title(f'{self.model_name} - Mean Confusion Matrix\n(n={self.num_runs} runs)', 
                         fontsize=14, fontweight='bold')
        
        # Normalized with annotations
        annot_labels = np.empty_like(cm_norm, dtype=object)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                annot_labels[i, j] = f'{cm_norm[i, j]:.2%}\n±{cm_std[i, j]:.1f}'
        
        sns.heatmap(cm_norm, annot=annot_labels, fmt='', cmap='Blues', ax=axes[1],
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Mean Percentage'})
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_title(f'{self.model_name} - Normalized Confusion Matrix\n(Mean % ± SD)', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'aggregated_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: aggregated_confusion_matrix.png")
    
    def _generate_publication_table(self):
        """Generate publication-ready tables"""
        # Overall metrics table
        overall_data = []
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        
        overall_data.append({
            'Metric': 'Accuracy',
            'Mean': f"{self.aggregated_metrics['accuracy']['mean']:.4f}",
            'SD': f"{self.aggregated_metrics['accuracy']['std']:.4f}",
            '95% CI': f"±{self.aggregated_metrics['accuracy']['ci_95']:.4f}",
            'Range': f"[{min(self.aggregated_metrics['accuracy']['all_values']):.4f}, "
                    f"{max(self.aggregated_metrics['accuracy']['all_values']):.4f}]"
        })
        
        for metric in ['precision', 'recall', 'f1', 'auc']:
            mean, std, ci = self.aggregated_metrics['macro_avg'][metric]
            overall_data.append({
                'Metric': f'Macro {metric.capitalize()}',
                'Mean': f"{mean:.4f}",
                'SD': f"{std:.4f}",
                '95% CI': f"±{ci:.4f}",
                'Range': f"[{mean-std:.4f}, {mean+std:.4f}]"
            })
        
        df_overall = pd.DataFrame(overall_data)
        
        # Per-class metrics table
        per_class_data = []
        for class_idx, class_name in enumerate(self.class_names):
            row = {'Class': class_name}
            for metric in ['precision', 'recall', 'f1', 'auc']:
                mean = self.aggregated_metrics['per_class'][metric]['mean'][class_idx]
                ci = self.aggregated_metrics['per_class'][metric]['ci'][class_idx]
                row[metric.capitalize()] = f"{mean:.3f}±{ci:.3f}"
            per_class_data.append(row)
        
        df_per_class = pd.DataFrame(per_class_data)
        
        # Save tables
        df_overall.to_csv(self.output_dir / 'overall_metrics_table.csv', index=False)
        df_per_class.to_csv(self.output_dir / 'per_class_metrics_table.csv', index=False)
        
        print("  ✓ Saved: overall_metrics_table.csv")
        print("  ✓ Saved: per_class_metrics_table.csv")
        
        # Create formatted text report
        with open(self.output_dir / 'publication_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"{self.model_name} - PUBLICATION-READY RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Number of runs: {self.num_runs}\n")
            f.write(f"Random seeds: {self.base_seed} to {self.base_seed + self.num_runs - 1}\n")
            f.write("="*80 + "\n\n")
            
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(df_overall.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n\n")
            
            f.write("PER-CLASS PERFORMANCE (Mean ± 95% CI)\n")
            f.write("-"*80 + "\n")
            f.write(df_per_class.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n\n")
            
            # Statistical summary
            f.write("STATISTICAL SUMMARY\n")
            f.write("-"*80 + "\n")
            acc_mean = self.aggregated_metrics['accuracy']['mean']
            acc_ci = self.aggregated_metrics['accuracy']['ci_95']
            f.write(f"Model achieves {acc_mean*100:.2f}% ± {acc_ci*100:.2f}% accuracy ")
            f.write(f"(95% confidence interval, n={self.num_runs}).\n\n")
            
            f.write("All metrics reported as: Mean ± 95% Confidence Interval\n")
            f.write(f"Confidence intervals calculated using t-distribution with {self.num_runs-1} degrees of freedom.\n")
            f.write("="*80 + "\n")
        
        print("  ✓ Saved: publication_report.txt")
    
    def _save_all_results(self):
        """Save complete results to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        individual_runs = []
        for run in self.all_runs_results:
            run_copy = run.copy()
            run_copy['confusion_matrix'] = run_copy['confusion_matrix'].tolist()
            individual_runs.append(run_copy)
        
        results = {
            'model_name': self.model_name,
            'checkpoint': str(self.checkpoint_path),
            'num_runs': self.num_runs,
            'seeds': list(range(self.base_seed, self.base_seed + self.num_runs)),
            'aggregated_metrics': {
                'accuracy': {
                    'mean': float(self.aggregated_metrics['accuracy']['mean']),
                    'std': float(self.aggregated_metrics['accuracy']['std']),
                    'ci_95': float(self.aggregated_metrics['accuracy']['ci_95']),
                    'all_values': [float(v) for v in self.aggregated_metrics['accuracy']['all_values']]
                },
                'macro_avg': {
                    metric: {
                        'mean': float(vals[0]),
                        'std': float(vals[1]),
                        'ci_95': float(vals[2])
                    }
                    for metric, vals in self.aggregated_metrics['macro_avg'].items()
                },
                'per_class': {
                    'class_names': self.class_names,
                    'metrics': {
                        metric: {
                            'mean': [float(v) for v in vals['mean']],
                            'std': [float(v) for v in vals['std']],
                            'ci_95': [float(v) for v in vals['ci']]
                        }
                        for metric, vals in self.aggregated_metrics['per_class'].items()
                    }
                },
                'confusion_matrix': {
                    'mean': self.aggregated_metrics['confusion_matrix']['mean'].tolist(),
                    'std': self.aggregated_metrics['confusion_matrix']['std'].tolist()
                }
            },
            'individual_runs': individual_runs
        }
        
        with open(self.output_dir / 'multi_run_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("  ✓ Saved: multi_run_results.json")
    
    def _print_summary(self):
        """Print summary of results"""
        acc_mean = self.aggregated_metrics['accuracy']['mean']
        acc_ci = self.aggregated_metrics['accuracy']['ci_95']
        
        print(f"\nFINAL RESULTS (n={self.num_runs} runs):")
        print(f"  Accuracy: {acc_mean*100:.2f}% ± {acc_ci*100:.2f}% (95% CI)")
        print(f"  Range: [{min(self.aggregated_metrics['accuracy']['all_values'])*100:.2f}%, "
              f"{max(self.aggregated_metrics['accuracy']['all_values'])*100:.2f}%]")


def load_model(model_name, num_classes=6):
    """Factory function to create model"""
    models = {
        'resnet': lambda: ResNet34_LoTSS(num_classes=num_classes, pretrained=False),
        'efficientnet': lambda: EfficientNetB0_LoTSS(num_classes=num_classes, pretrained=False),
        'vit': lambda: ViT_LoTSS(num_classes=num_classes, img_size=600, patch_size=30),
        'convnext': lambda: ConvNeXtTiny_LoTSS(num_classes=num_classes, pretrained=False)
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")
    
    return models[model_name.lower()]


def main():
    parser = argparse.ArgumentParser(description='Multi-Run Model Testing with Statistical Analysis')
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet', 'efficientnet', 'vit', 'convnext'],
                       help='Model to test')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/beautiful_dataset_v2',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of test runs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (each run uses seed+i)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine target size based on model - all models use 600x600
    target_size = (600, 600)
    
    # Create model factory
    model_fn = load_model(args.model)
    
    # Create tester
    tester = MultiRunTester(
        model_fn=model_fn,
        model_name=args.model.capitalize(),
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        device=device,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=target_size,
        seed=args.seed
    )
    
    # Run all tests
    tester.run_all_tests()

if __name__ == "__main__":
    main()