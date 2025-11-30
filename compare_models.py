"""
Model Comparison for CBIS-DDSM

Compares all trained models on test set performance.
Generates comparison tables and overlaid ROC/PR curves.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Directories
RESULTS_DIR = r"d:\Allen Archive\Allen Archives\NEU_academics\Semester4\ML\Project\CBIS_DDSM_Model_Comparison\results"
COMPARISON_DIR = os.path.join(RESULTS_DIR, "comparison")

os.makedirs(COMPARISON_DIR, exist_ok=True)

# Model configurations
MODELS = {
    'ConvNeXt': 'convnext',
    'DenseNet121': 'densenet121',
    'Swin Transformer': 'swin_transformer',
    'Swin Multi-View': 'swin_transformer_multiview',
    'ResNet18': 'resnet18',
    'ResNet50': 'resnet50',
    'EfficientNet-B0': 'efficientnet',
    'U-Net': 'unet',
    'Custom CNN': 'custom_cnn'
}


def load_metrics(model_name, metrics_path):
    """Load metrics from CSV file"""
    try:
        df = pd.read_csv(metrics_path)
        metrics = df.iloc[0].to_dict()
        print(f"Loaded metrics for {model_name}")
        return metrics
    except Exception as e:
        print(f"Error loading {model_name} metrics: {e}")
        return None


def create_all_models_comparison_table(all_metrics):
    """Create comparison table for all models"""
    
    if not all_metrics:
        return None
    
    # Get all metric names from first model
    first_model = list(all_metrics.keys())[0]
    metric_names = list(all_metrics[first_model].keys())
    
    # Create comparison data
    comparison_data = {'Metric': [metric.replace('_', ' ') for metric in metric_names]}
    
    # Add each model's metrics
    for model_name, metrics in all_metrics.items():
        comparison_data[model_name] = [f"{metrics[metric]:.4f}" for metric in metric_names]
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = os.path.join(COMPARISON_DIR, "all_models_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved all models comparison: {csv_path}")
    
    return df


def create_metrics_comparison_table(convnext_metrics, densenet_metrics):
    """Create and save comparison table (legacy function for backward compatibility)"""
    
    comparison_data = {
        'Metric': [],
        'ConvNeXt': [],
        'DenseNet121': [],
        'Difference': []
    }
    
    for metric in convnext_metrics.keys():
        comparison_data['Metric'].append(metric.replace('_', ' '))
        convnext_val = convnext_metrics[metric]
        densenet_val = densenet_metrics[metric]
        comparison_data['ConvNeXt'].append(f"{convnext_val:.4f}")
        comparison_data['DenseNet121'].append(f"{densenet_val:.4f}")
        diff = densenet_val - convnext_val
        comparison_data['Difference'].append(f"{diff:+.4f}")
    
    df = pd.DataFrame(comparison_data)
    # Legacy: keep the table in memory only, do not write metrics_comparison.csv anymore
    return df


def print_all_models_comparison_table(df):
    """Print formatted comparison table for all models"""
    print("\nAll Models Performance Comparison")
    print("=" * 120)
    
    # Create header
    header = f"{'Metric':<20}"
    for col in df.columns[1:]:  # Skip 'Metric' column
        header += f"{col:<12}"
    print(header)
    print("-" * 120)
    
    # Print each row
    for _, row in df.iterrows():
        row_str = f"{row['Metric']:<20}"
        for col in df.columns[1:]:
            row_str += f"{row[col]:<12}"
        print(row_str)
    
    print("=" * 120)


def print_comparison_table(df):
    """Print formatted comparison table (legacy for 2 models)"""
    print("\nModel Performance Comparison")
    print("-" * 70)
    print(f"{'Metric':<25} {'ConvNeXt':<12} {'DenseNet121':<12} {'Difference':<12}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{row['Metric']:<25} {row['ConvNeXt']:<12} {row['DenseNet121']:<12} {row['Difference']:<12}")
    
    print("-" * 70)


def create_all_models_bar_chart(all_metrics, output_path):
    """Create grouped bar chart comparing all models"""
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC', 'MCC']
    labels = [m.replace('_', ' ') for m in metrics_to_plot]
    
    model_names = list(all_metrics.keys())
    colors = ['#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#17BECF', '#BCBD22', '#7F7F7F']
    
    x = np.arange(len(labels))
    width = 0.10
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, (model_name, metrics) in enumerate(all_metrics.items()):
        values = [metrics[m] for m in metrics_to_plot]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, 
                     color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only show labels for visible bars
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=7, rotation=0)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('All Models Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved all models bar chart: {output_path}")


def create_metrics_bar_chart(convnext_metrics, densenet_metrics, output_path):
    """Create grouped bar chart comparing metrics (legacy function)"""
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC', 'Sensitivity_TPR', 'Specificity_TNR', 'MCC']
    
    convnext_values = [convnext_metrics[m] for m in metrics_to_plot]
    densenet_values = [densenet_metrics[m] for m in metrics_to_plot]
    
    labels = [m.replace('_', ' ') for m in metrics_to_plot]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width/2, convnext_values, width, label='ConvNeXt', color='#FF7F0E', alpha=0.8)
    bars2 = ax.bar(x + width/2, densenet_values, width, label='DenseNet121', color='#2CA02C', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart: {output_path}")


def load_predictions(model_dir):
    """Load saved predictions from model evaluation"""
    pred_file = os.path.join(model_dir, 'predictions.npz')
    if os.path.exists(pred_file):
        data = np.load(pred_file)
        return data['y_true'], data['y_pred_proba']
    return None, None


def create_roc_comparison(output_path):
    """Create overlaid ROC curves for all models"""
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.5)
    
    # Color palette for models
    colors = ['#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#17BECF', '#BCBD22', '#7F7F7F']
    color_idx = 0
    
    # Load and plot each model
    for model_name, model_dir in MODELS.items():
        model_dir_path = os.path.join(RESULTS_DIR, model_dir)
        y_true, y_proba = load_predictions(model_dir_path)
        
        if y_true is not None and y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[color_idx % len(colors)], lw=2.5, 
                   label=f'{model_name} (AUC = {roc_auc:.4f})')
            color_idx += 1
        else:
            print(f"Warning: {model_name} predictions not found")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve Comparison - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC comparison: {output_path}")


def create_pr_comparison(output_path):
    """Create overlaid Precision-Recall curves for all models"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color palette for models
    colors = ['#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#17BECF', '#BCBD22', '#7F7F7F']
    color_idx = 0
    
    # Load and plot each model
    for model_name, model_dir in MODELS.items():
        model_dir_path = os.path.join(RESULTS_DIR, model_dir)
        y_true, y_proba = load_predictions(model_dir_path)
        
        if y_true is not None and y_proba is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
            ax.plot(recall, precision, color=colors[color_idx % len(colors)], lw=2.5, 
                   label=f'{model_name} (AP = {pr_auc:.4f})')
            color_idx += 1
        else:
            print(f"Warning: {model_name} predictions not found")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve Comparison - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR comparison: {output_path}")


def create_summary_report(all_metrics):
    """Create text summary report for all models"""
    
    report_path = os.path.join(COMPARISON_DIR, "comparison_summary.txt")
    
    # Get all metric names from first model
    first_model = list(all_metrics.keys())[0]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC', 'MCC']
    
    with open(report_path, 'w') as f:
        f.write("CBIS-DDSM Model Comparison Report\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Models Compared:\n")
        for i, model in enumerate(all_metrics.keys(), 1):
            f.write(f"  {i}. {model}\n")
        f.write("\n")
        
        f.write("Dataset: CBIS-DDSM Breast Cancer Mammography\n")
        f.write("Task: Binary Classification (Benign vs Malignant)\n\n")
        
        f.write("Performance Comparison:\n")
        f.write("-" * 100 + "\n")
        
        # Header
        header = f"{'Metric':<20}"
        for model in all_metrics.keys():
            header += f"{model:<20}"
        f.write(header + "\n")
        f.write("-" * 100 + "\n")
        
        # Rows
        for metric in metric_names:
            row_str = f"{metric:<20}"
            for model in all_metrics.keys():
                val = all_metrics[model].get(metric, 0)
                row_str += f"{val:<20.4f}"
            f.write(row_str + "\n")
            
        f.write("-" * 100 + "\n\n")
        
        # Best model analysis
        best_f1_model = max(all_metrics.items(), key=lambda x: x[1]['F1'])[0]
        best_auc_model = max(all_metrics.items(), key=lambda x: x[1]['ROC_AUC'])[0]
        best_recall_model = max(all_metrics.items(), key=lambda x: x[1]['Recall'])[0]
        
        f.write("Summary & Recommendations:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Overall (F1 Score): {best_f1_model} ({all_metrics[best_f1_model]['F1']:.4f})\n")
        f.write(f"Best Discriminator (ROC AUC): {best_auc_model} ({all_metrics[best_auc_model]['ROC_AUC']:.4f})\n")
        f.write(f"Best Sensitivity (Recall): {best_recall_model} ({all_metrics[best_recall_model]['Recall']:.4f})\n\n")
        
        f.write("Detailed Analysis:\n")
        
        # Sort models by F1
        sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['F1'], reverse=True)
        
        f.write(f"1. {sorted_models[0][0]} is the top performing model.\n")
        f.write(f"   - F1 Score: {sorted_models[0][1]['F1']:.4f}\n")
        f.write(f"   - ROC AUC: {sorted_models[0][1]['ROC_AUC']:.4f}\n")
        f.write(f"   - Recall: {sorted_models[0][1]['Recall']:.4f}\n")
        
        if len(sorted_models) > 1:
            f.write(f"\n2. {sorted_models[1][0]} is the runner-up.\n")
            f.write(f"   - F1 Score: {sorted_models[1][1]['F1']:.4f}\n")
            
        f.write("\nConclusion:\n")
        if best_f1_model == best_auc_model:
            f.write(f"{best_f1_model} is the clear winner, achieving both the highest F1 score and ROC AUC. ")
        else:
            f.write(f"{best_f1_model} shows the best balance of precision and recall, while {best_auc_model} has the best overall discriminative ability. ")
            
        f.write("For medical screening, high sensitivity (Recall) is crucial to avoid missing malignant cases. ")
        f.write(f"{best_recall_model} achieves the best sensitivity of {all_metrics[best_recall_model]['Recall']:.4f}.\n")
    
    print(f"Saved summary report: {report_path}")


def main():
    print("\nModel Comparison Analysis")
    print("-" * 40)
    
    # Load all available model metrics
    print("\nLoading model metrics...")
    all_metrics = {}
    
    for model_name, model_dir in MODELS.items():
        metrics_path = os.path.join(RESULTS_DIR, model_dir, "test_metrics.csv")
        if os.path.exists(metrics_path):
            metrics = load_metrics(model_name, metrics_path)
            if metrics is not None:
                all_metrics[model_name] = metrics
        else:
            print(f"Skipping {model_name} (not yet trained)")
    
    if len(all_metrics) < 2:
        print("\nError: Need at least 2 trained models for comparison.")
        print("Please train more models first.")
        return
    
    print(f"\nComparing {len(all_metrics)} models: {', '.join(all_metrics.keys())}")
    
    # Create comparison table for ALL models
    print("\nGenerating all models comparison table...")
    all_models_df = create_all_models_comparison_table(all_metrics)
    if all_models_df is not None:
        print_all_models_comparison_table(all_models_df)
    
    # Clean up legacy file if it exists
    legacy_csv = os.path.join(COMPARISON_DIR, "metrics_comparison.csv")
    if os.path.exists(legacy_csv):
        try:
            os.remove(legacy_csv)
            print("Removed legacy comparison file: metrics_comparison.csv")
        except:
            pass
    
    # Create visualizations
    print("\nGenerating comparison visualizations...")
    
    # Bar chart for ALL models
    bar_chart_path = os.path.join(COMPARISON_DIR, "metrics_bar_chart.png")
    create_all_models_bar_chart(all_metrics, bar_chart_path)
    
    # ROC curve comparison
    roc_path = os.path.join(COMPARISON_DIR, "roc_comparison.png")
    create_roc_comparison(roc_path)
    
    # PR curve comparison
    pr_path = os.path.join(COMPARISON_DIR, "pr_comparison.png")
    create_pr_comparison(pr_path)
    
    # Create summary report
    print("\nGenerating summary report...")
    create_summary_report(all_metrics)
    
    print("\nComparison complete!")
    print(f"Results saved to: {COMPARISON_DIR}")
    print("\nGenerated files:")
    print("  - all_models_comparison.csv (metrics table)")
    print("  - metrics_bar_chart.png (metrics visualization)")
    print("  - roc_comparison.png (ROC curves)")
    print("  - pr_comparison.png (PR curves)")
    print("  - comparison_summary.txt (text summary)")


if __name__ == "__main__":
    main()
