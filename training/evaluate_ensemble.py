"""
Evaluate Ensemble Model on LoTSS Dataset
"""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from models.ensemble_model import EnsembleClassifier
from data.dataset import LoTSSDataset
from data.preprocessing import LoTSSPreprocessing
from data.augmentation import CombinedTransform
from utils.metrics_tracker import MetricsTracker


def evaluate_ensemble(ensemble, dataloader, device, class_names):
    """Evaluate ensemble on dataset"""
    ensemble.eval()
    
    all_preds = []
    all_labels = []
    all_eff_preds = []
    all_conv_preds = []
    
    correct = 0
    total = 0
    
    print("\nEvaluating ensemble...")
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluation"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get ensemble prediction
            preds, probs = ensemble.predict(images)
            
            # Get individual predictions for comparison
            individual = ensemble.get_individual_predictions(images)
            
            # Accuracy
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
            # Store predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_eff_preds.extend(individual['efficientnet']['preds'].cpu().numpy())
            all_conv_preds.extend(individual['convnext']['preds'].cpu().numpy())
    
    accuracy = correct / total
    
    return {
        'accuracy': accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'efficientnet_preds': np.array(all_eff_preds),
        'convnext_preds': np.array(all_conv_preds)
    }


def compare_models(results, class_names, output_dir):
    """Compare ensemble vs individual models"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels = results['labels']
    ensemble_preds = results['predictions']
    eff_preds = results['efficientnet_preds']
    conv_preds = results['convnext_preds']
    
    # Calculate accuracies
    ensemble_acc = (ensemble_preds == labels).mean()
    eff_acc = (eff_preds == labels).mean()
    conv_acc = (conv_preds == labels).mean()
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  EfficientNet:  {eff_acc:.2%}")
    print(f"  ConvNeXt:      {conv_acc:.2%}")
    print(f"  Ensemble:      {ensemble_acc:.2%}")
    print(f"  Improvement:   +{(ensemble_acc - max(eff_acc, conv_acc))*100:.2f}%")
    print(f"{'='*60}")
    
    # Detailed comparison
    print("\nDetailed Analysis:")
    
    # Cases where ensemble is correct but individuals are wrong
    eff_wrong_ens_right = (eff_preds != labels) & (ensemble_preds == labels)
    conv_wrong_ens_right = (conv_preds != labels) & (ensemble_preds == labels)
    both_wrong_ens_right = (eff_preds != labels) & (conv_preds != labels) & (ensemble_preds == labels)
    
    print(f"  Ensemble corrects EfficientNet errors: {eff_wrong_ens_right.sum()}")
    print(f"  Ensemble corrects ConvNeXt errors: {conv_wrong_ens_right.sum()}")
    print(f"  Ensemble corrects both models: {both_wrong_ens_right.sum()}")
    
    # Cases where both individuals are correct
    both_correct = (eff_preds == labels) & (conv_preds == labels)
    print(f"  Both models correct: {both_correct.sum()} ({both_correct.mean():.2%})")
    
    # Agreement analysis
    agreement = (eff_preds == conv_preds)
    print(f"\n  Model agreement: {agreement.mean():.2%}")
    print(f"  Disagreement cases: {(~agreement).sum()}")
    
    # Classification reports
    print(f"\n{'='*60}")
    print("ENSEMBLE CLASSIFICATION REPORT")
    print(f"{'='*60}")
    report = classification_report(labels, ensemble_preds, target_names=class_names, digits=4)
    print(report)
    
    # Save report
    with open(output_dir / 'ensemble_report.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("ENSEMBLE MODEL EVALUATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"EfficientNet Accuracy:  {eff_acc:.4%}\n")
        f.write(f"ConvNeXt Accuracy:      {conv_acc:.4%}\n")
        f.write(f"Ensemble Accuracy:      {ensemble_acc:.4%}\n")
        f.write(f"Improvement:            +{(ensemble_acc - max(eff_acc, conv_acc))*100:.2f}%\n\n")
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(report)
    
    # Plot confusion matrix
    plot_comparison_confusion_matrix(labels, ensemble_preds, eff_preds, conv_preds, 
                                     class_names, output_dir)
    
    return {
        'ensemble_acc': ensemble_acc,
        'efficientnet_acc': eff_acc,
        'convnext_acc': conv_acc,
        'improvement': ensemble_acc - max(eff_acc, conv_acc)
    }


def plot_comparison_confusion_matrix(labels, ens_preds, eff_preds, conv_preds, 
                                      class_names, output_dir):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for ax, preds, title in zip(axes, 
                                [eff_preds, conv_preds, ens_preds],
                                ['EfficientNet', 'ConvNeXt', 'Ensemble']):
        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                   xticklabels=[c.replace('_', '\n') for c in class_names],
                   yticklabels=[c.replace('_', '\n') for c in class_names],
                   cbar_kws={'label': 'Percentage'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{title}\nAcc: {(preds == labels).mean():.2%}', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_dir / 'ensemble_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ensemble Model')
    
    # Model paths
    parser.add_argument('--efficientnet_path', type=str, 
                       default='outputs/efficientnet/best_model.pth',
                       help='Path to EfficientNet checkpoint')
    parser.add_argument('--convnext_path', type=str,
                       default='outputs/convnext/best_model.pth',
                       help='Path to ConvNeXt checkpoint')
    
    # Data
    parser.add_argument('--data_path', type=str, default='data/beautiful_dataset_v2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Ensemble settings
    parser.add_argument('--method', type=str, default='weighted',
                       choices=['average', 'weighted', 'vote'],
                       help='Ensemble method')
    parser.add_argument('--output_dir', type=str, default='outputs/ensemble')
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    print("\n" + "="*60)
    print("ENSEMBLE MODEL EVALUATION")
    print("="*60)
    
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
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create ensemble
    ensemble = EnsembleClassifier(
        args.efficientnet_path,
        args.convnext_path,
        num_classes=6,
        method=args.method,
        device=device
    )
    
    # Evaluate
    results = evaluate_ensemble(ensemble, val_loader, device, val_dataset.class_folders)
    
    # Compare models
    comparison = compare_models(results, val_dataset.class_folders, args.output_dir)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Ensemble Accuracy: {comparison['ensemble_acc']:.2%}")
    print(f"Improvement over best: +{comparison['improvement']*100:.2f}%")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()