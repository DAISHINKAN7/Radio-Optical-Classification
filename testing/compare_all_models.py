"""
Compare all trained models side-by-side
Generates comprehensive comparison report
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd


def load_metrics(output_dir):
    """Load test metrics from JSON"""
    metrics_file = Path(output_dir) / 'test_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def compare_models(model_results):
    """Generate comparison visualizations"""
    output_dir = Path('outputs/model_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = [
        'Compact Sources',
        'Elliptical Galaxies',
        'FR2 Radio Galaxies',
        'Radio Loud AGN',
        'Spiral Galaxies',
        'Starburst Galaxies'
    ]
    
    # 1. Overall Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(model_results.keys())
    accuracies = [model_results[m]['accuracy'] * 100 for m in models]
    
    bars = ax.bar(models, accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    plt.close()
    
    # 2. Per-Class F1 Comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(class_names))
    width = 0.2
    
    for i, (model_name, color) in enumerate(zip(models, ['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])):
        f1_scores = model_results[model_name]['per_class']['f1']
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, [f * 100 for f in f1_scores], width, 
               label=model_name, color=color)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1-Score (%)', fontsize=12)
    ax.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_comparison.png', dpi=150)
    plt.close()
    
    # 3. Metrics Heatmap
    metrics_data = []
    for model_name in models:
        row = [
            model_results[model_name]['accuracy'],
            model_results[model_name]['macro_avg']['precision'],
            model_results[model_name]['macro_avg']['recall'],
            model_results[model_name]['macro_avg']['f1']
        ]
        metrics_data.append(row)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.set_yticklabels(models)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(4):
            text = ax.text(j, i, f'{metrics_data[i][j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Model Metrics Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=150)
    plt.close()
    
    # 4. Generate comparison table
    df_data = []
    for model_name in models:
        df_data.append({
            'Model': model_name,
            'Accuracy': f"{model_results[model_name]['accuracy']:.4f}",
            'Precision': f"{model_results[model_name]['macro_avg']['precision']:.4f}",
            'Recall': f"{model_results[model_name]['macro_avg']['recall']:.4f}",
            'F1-Score': f"{model_results[model_name]['macro_avg']['f1']:.4f}"
        })
    
    df = pd.DataFrame(df_data)
    
    # Save as CSV
    df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # Save as text
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*60 + "\n")
        f.write("BEST MODEL: ")
        best_model = models[np.argmax(accuracies)]
        f.write(f"{best_model} ({max(accuracies):.2f}%)\n")
        f.write("="*60 + "\n")
    
    print(f"\n✓ Comparison complete! Results saved to: {output_dir}")
    print(f"\nBest Model: {best_model} ({max(accuracies):.2f}%)")


if __name__ == "__main__":
    # Load results from all models
    model_results = {}
    
    models_to_compare = {
        'ResNet': 'outputs/resnet_test',
        'EfficientNet': 'outputs/efficientnet_test',
        'ViT': 'outputs/vit_test',
        'ConvNeXt': 'outputs/convnext_test'
    }
    
    print("\nLoading model results...")
    for model_name, output_dir in models_to_compare.items():
        metrics = load_metrics(output_dir)
        if metrics:
            model_results[model_name] = metrics
            print(f"  ✓ Loaded {model_name}: {metrics['accuracy']:.2%}")
        else:
            print(f"  ✗ {model_name}: No results found in {output_dir}")
    
    if len(model_results) >= 2:
        compare_models(model_results)
    else:
        print("\n⚠️  Need at least 2 models tested to compare!")
        print("Run testing first with: python testing/test_model.py")