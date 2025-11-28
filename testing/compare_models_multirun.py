"""
Compare Multiple Models with Statistical Significance Testing
- Side-by-side comparison with confidence intervals
- Statistical significance tests (t-tests, ANOVA)
- Publication-ready comparison tables and figures
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats
from itertools import combinations


def load_multi_run_results(output_dir):
    """Load multi-run test results from JSON"""
    results_file = Path(output_dir) / 'multi_run_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def perform_statistical_tests(model_results):
    """Perform statistical significance tests between models"""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*80 + "\n")
    
    model_names = list(model_results.keys())
    
    # Extract accuracy values for each model
    accuracy_data = {
        name: results['aggregated_metrics']['accuracy']['all_values']
        for name, results in model_results.items()
    }
    
    # Perform pairwise t-tests
    print("Pairwise T-Tests (Accuracy):")
    print("-"*80)
    
    pairwise_results = []
    for model1, model2 in combinations(model_names, 2):
        t_stat, p_value = stats.ttest_ind(accuracy_data[model1], accuracy_data[model2])
        
        mean1 = np.mean(accuracy_data[model1])
        mean2 = np.mean(accuracy_data[model2])
        
        significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"{model1} vs {model2}:")
        print(f"  {model1}: {mean1*100:.2f}%")
        print(f"  {model2}: {mean2*100:.2f}%")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f} {significant}")
        print()
        
        pairwise_results.append({
            'Model 1': model1,
            'Model 2': model2,
            'Mean 1 (%)': f"{mean1*100:.2f}",
            'Mean 2 (%)': f"{mean2*100:.2f}",
            'Difference (%)': f"{abs(mean1-mean2)*100:.2f}",
            't-statistic': f"{t_stat:.4f}",
            'p-value': f"{p_value:.4f}",
            'Significance': significant
        })
    
    # Perform ANOVA if more than 2 models
    if len(model_names) > 2:
        print("\nOne-Way ANOVA (Accuracy):")
        print("-"*80)
        f_stat, p_value = stats.f_oneway(*accuracy_data.values())
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Result: Significant differences exist between models (p < 0.05)")
        else:
            print("Result: No significant differences between models (p >= 0.05)")
    
    return pairwise_results


def compare_models_comprehensive(model_results):
    """Generate comprehensive comparison visualizations"""
    output_dir = Path('outputs/multi_run_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_names = list(model_results.keys())
    class_names = model_results[model_names[0]]['aggregated_metrics']['per_class']['class_names']
    
    # Perform statistical tests
    pairwise_results = perform_statistical_tests(model_results)
    
    # 1. Accuracy Comparison with Error Bars
    fig, ax = plt.subplots(figsize=(12, 7))
    
    means = []
    cis = []
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12'][:len(model_names)]
    
    for model_name in model_names:
        acc_metrics = model_results[model_name]['aggregated_metrics']['accuracy']
        means.append(acc_metrics['mean'] * 100)
        cis.append(acc_metrics['ci_95'] * 100)
    
    x = np.arange(len(model_names))
    bars = ax.bar(x, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.errorbar(x, means, yerr=cis, fmt='none', color='red', capsize=10, 
               capthick=2, linewidth=2, label='95% CI')
    
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Model Accuracy Comparison with 95% Confidence Intervals', 
                fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add value labels
    for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
        ax.text(bar.get_x() + bar.get_width()/2., mean + ci + 1.5,
               f'{mean:.2f}%\n±{ci:.2f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison_ci.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ Saved: accuracy_comparison_ci.png")
    
    # 2. Boxplot Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    accuracy_data = [
        [v * 100 for v in model_results[name]['aggregated_metrics']['accuracy']['all_values']]
        for name in model_names
    ]
    
    bp = ax.boxplot(accuracy_data, labels=model_names, patch_artist=True,
                   notch=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Accuracy Distribution Across Runs', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: accuracy_boxplot.png")
    
    # 3. Per-Class F1 Comparison with Error Bars
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(class_names))
    width = 0.8 / len(model_names)
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        per_class = model_results[model_name]['aggregated_metrics']['per_class']['metrics']
        f1_means = [m * 100 for m in per_class['f1']['mean']]
        f1_cis = [c * 100 for c in per_class['f1']['ci_95']]
        
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, f1_means, width, label=model_name, 
                     color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.errorbar(x + offset, f1_means, yerr=f1_cis, fmt='none', 
                   color='black', capsize=3, linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Class', fontsize=13)
    ax.set_ylabel('F1-Score (%)', fontsize=13)
    ax.set_title('Per-Class F1-Score Comparison with 95% CI', 
                fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_comparison_ci.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: f1_comparison_ci.png")
    
    # 4. Comprehensive Metrics Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    heatmap_data = []
    annotations = []
    
    for model_name in model_names:
        agg = model_results[model_name]['aggregated_metrics']
        row = [
            agg['accuracy']['mean'],
            agg['macro_avg']['precision']['mean'],
            agg['macro_avg']['recall']['mean'],
            agg['macro_avg']['f1']['mean'],
            agg['macro_avg']['auc']['mean']
        ]
        heatmap_data.append(row)
        
        annot_row = [
            f"{agg['accuracy']['mean']:.3f}\n±{agg['accuracy']['ci_95']:.3f}",
            f"{agg['macro_avg']['precision']['mean']:.3f}\n±{agg['macro_avg']['precision']['ci_95']:.3f}",
            f"{agg['macro_avg']['recall']['mean']:.3f}\n±{agg['macro_avg']['recall']['ci_95']:.3f}",
            f"{agg['macro_avg']['f1']['mean']:.3f}\n±{agg['macro_avg']['f1']['ci_95']:.3f}",
            f"{agg['macro_avg']['auc']['mean']:.3f}\n±{agg['macro_avg']['auc']['ci_95']:.3f}"
        ]
        annotations.append(annot_row)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_yticklabels(model_names, fontsize=12)
    
    # Add annotations
    for i in range(len(model_names)):
        for j in range(len(metrics_names)):
            ax.text(j, i, annotations[i][j],
                   ha="center", va="center", color="black", 
                   fontsize=9, fontweight='bold')
    
    ax.set_title('Model Metrics Heatmap\n(Mean ± 95% CI)', 
                fontsize=15, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap_ci.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: metrics_heatmap_ci.png")
    
    # 5. Generate Comparison Tables
    # Overall metrics table
    overall_data = []
    for model_name in model_names:
        agg = model_results[model_name]['aggregated_metrics']
        num_runs = model_results[model_name]['num_runs']
        
        overall_data.append({
            'Model': model_name,
            'Runs': num_runs,
            'Accuracy (%)': f"{agg['accuracy']['mean']*100:.2f} ± {agg['accuracy']['ci_95']*100:.2f}",
            'Precision': f"{agg['macro_avg']['precision']['mean']:.3f} ± {agg['macro_avg']['precision']['ci_95']:.3f}",
            'Recall': f"{agg['macro_avg']['recall']['mean']:.3f} ± {agg['macro_avg']['recall']['ci_95']:.3f}",
            'F1-Score': f"{agg['macro_avg']['f1']['mean']:.3f} ± {agg['macro_avg']['f1']['ci_95']:.3f}",
            'AUC': f"{agg['macro_avg']['auc']['mean']:.3f} ± {agg['macro_avg']['auc']['ci_95']:.3f}"
        })
    
    df_overall = pd.DataFrame(overall_data)
    df_overall.to_csv(output_dir / 'comparison_table.csv', index=False)
    print("  ✓ Saved: comparison_table.csv")
    
    # Statistical tests table
    df_pairwise = pd.DataFrame(pairwise_results)
    df_pairwise.to_csv(output_dir / 'statistical_tests.csv', index=False)
    print("  ✓ Saved: statistical_tests.csv")
    
    # 6. Generate comprehensive report
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL METRICS (Mean ± 95% CI)\n")
        f.write("-"*80 + "\n")
        f.write(df_overall.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n\n")
        
        f.write("STATISTICAL SIGNIFICANCE TESTS\n")
        f.write("-"*80 + "\n")
        f.write(df_pairwise.to_string(index=False))
        f.write("\n\n")
        f.write("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant\n")
        f.write("\n" + "="*80 + "\n\n")
        
        # Best model
        best_idx = df_overall['Accuracy (%)'].apply(
            lambda x: float(x.split(' ± ')[0])
        ).idxmax()
        best_model = df_overall.loc[best_idx, 'Model']
        best_acc = df_overall.loc[best_idx, 'Accuracy (%)']
        
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Best Model: {best_model}\n")
        f.write(f"Accuracy: {best_acc}%\n")
        f.write("\n" + "="*80 + "\n")
    
    print("  ✓ Saved: comparison_report.txt")
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Load results from all models
    model_results = {}
    
    models_to_compare = {
        'ResNet': 'outputs/resnet_multirun',
        'EfficientNet': 'outputs/efficientnet_multirun',
        'ViT': 'outputs/vit_multirun',
        'ConvNeXt': 'outputs/convnext_multirun'
    }
    
    print("\n" + "="*80)
    print("LOADING MULTI-RUN RESULTS")
    print("="*80 + "\n")
    
    for model_name, output_dir in models_to_compare.items():
        results = load_multi_run_results(output_dir)
        if results:
            model_results[model_name] = results
            acc = results['aggregated_metrics']['accuracy']
            print(f"  ✓ {model_name}: {acc['mean']*100:.2f}% ± {acc['ci_95']*100:.2f}% "
                  f"(n={results['num_runs']} runs)")
        else:
            print(f"  ✗ {model_name}: No results found in {output_dir}")
    
    if len(model_results) >= 2:
        compare_models_comprehensive(model_results)
    else:
        print("\n⚠️  Need at least 2 models tested to compare!")
        print("Run multi-run testing first with: python testing/test_model_multirun.py")