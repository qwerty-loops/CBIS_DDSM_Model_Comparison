"""
Visualize All Models Confusion Matrices
Creates a single figure with all 8 model confusion matrices in a grid
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Directories
RESULTS_DIR = r"d:\Allen Archive\Allen Archives\NEU_academics\Semester4\ML\Project\CBIS_DDSM_Model_Comparison\results"
COMPARISON_DIR = os.path.join(RESULTS_DIR, "comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)

# Model configurations (in order for grid)
MODELS = {
    'ConvNeXt': 'convnext',
    'DenseNet121': 'densenet121',
    'Swin Transformer': 'swin_transformer',
    'SparseWin Transformer': 'sparsewin_transformer',
    'ResNet18': 'resnet18',
    'ResNet50': 'resnet50',
    'EfficientNet-B0': 'efficientnet',
    'U-Net': 'unet'
}


def load_predictions(model_dir):
    """Load saved predictions from model evaluation"""
    pred_file = os.path.join(model_dir, 'predictions.npz')
    if os.path.exists(pred_file):
        data = np.load(pred_file)
        return data['y_true'], data['y_pred']
    return None, None


def create_all_confusion_matrices():
    """Create a grid of confusion matrices for all models"""
    
    # Set up the figure with 2 rows x 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Confusion Matrices - All Models Comparison', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Color map
    cmap = 'Blues'
    
    # Track if any model failed to load
    models_loaded = 0
    
    for idx, (model_name, model_dir) in enumerate(MODELS.items()):
        ax = axes[idx]
        model_dir_path = os.path.join(RESULTS_DIR, model_dir)
        
        # Load predictions
        y_true, y_pred = load_predictions(model_dir_path)
        
        if y_true is not None and y_pred is not None:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate metrics for annotation
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                       cbar=False, ax=ax,
                       annot_kws={'size': 14, 'weight': 'bold'},
                       linewidths=2, linecolor='white',
                       square=True)
            
            # Set labels
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}\nAcc: {accuracy:.3f} | Sens: {sensitivity:.3f} | Spec: {specificity:.3f}', 
                        fontsize=12, fontweight='bold', pad=10)
            
            # Set tick labels
            ax.set_xticklabels(['Benign (0)', 'Malignant (1)'], fontsize=10)
            ax.set_yticklabels(['Benign (0)', 'Malignant (1)'], fontsize=10, rotation=0)
            
            models_loaded += 1
            print(f"✓ Loaded {model_name}")
            
        else:
            # Model not found - show empty plot with message
            ax.text(0.5, 0.5, f'{model_name}\n(Not Available)', 
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            print(f"✗ {model_name} predictions not found")
    
    # Add legend/explanation at the bottom
    fig.text(0.5, 0.02, 
             'Confusion Matrix Layout: Rows = True Labels, Columns = Predicted Labels | '
             'TN (top-left), FP (top-right), FN (bottom-left), TP (bottom-right)',
             ha='center', fontsize=11, style='italic', color='gray')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save figure
    output_path = os.path.join(COMPARISON_DIR, 'all_confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved all confusion matrices: {output_path}")
    
    # Also save high-res version
    output_path_hires = os.path.join(COMPARISON_DIR, 'all_confusion_matrices_hires.png')
    plt.savefig(output_path_hires, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved high-res version: {output_path_hires}")
    
    plt.close()
    
    return models_loaded


def create_normalized_confusion_matrices():
    """Create a grid of NORMALIZED confusion matrices (percentages)"""
    
    # Set up the figure with 2 rows x 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Normalized Confusion Matrices - All Models Comparison (Percentages)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Color map
    cmap = 'RdYlGn'  # Red-Yellow-Green for percentages
    
    for idx, (model_name, model_dir) in enumerate(MODELS.items()):
        ax = axes[idx]
        model_dir_path = os.path.join(RESULTS_DIR, model_dir)
        
        # Load predictions
        y_true, y_pred = load_predictions(model_dir_path)
        
        if y_true is not None and y_pred is not None:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Normalize by row (true labels)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Calculate metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # Create heatmap with percentages
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap=cmap, 
                       cbar=False, ax=ax,
                       annot_kws={'size': 14, 'weight': 'bold'},
                       linewidths=2, linecolor='white',
                       square=True, vmin=0, vmax=1)
            
            # Set labels
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}\nAccuracy: {accuracy:.3f}', 
                        fontsize=12, fontweight='bold', pad=10)
            
            # Set tick labels
            ax.set_xticklabels(['Benign (0)', 'Malignant (1)'], fontsize=10)
            ax.set_yticklabels(['Benign (0)', 'Malignant (1)'], fontsize=10, rotation=0)
            
        else:
            # Model not found
            ax.text(0.5, 0.5, f'{model_name}\n(Not Available)', 
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    
    # Add explanation
    fig.text(0.5, 0.02, 
             'Normalized by True Labels (Rows) | Shows percentage of each true class predicted as each class',
             ha='center', fontsize=11, style='italic', color='gray')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save figure
    output_path = os.path.join(COMPARISON_DIR, 'all_confusion_matrices_normalized.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved normalized confusion matrices: {output_path}")
    
    plt.close()


def main():
    print("\n" + "="*60)
    print("Creating Confusion Matrix Visualizations")
    print("="*60 + "\n")
    
    print("Loading predictions from all models...")
    print("-" * 60)
    
    # Create absolute count confusion matrices
    models_loaded = create_all_confusion_matrices()
    
    print(f"\nSuccessfully loaded {models_loaded}/{len(MODELS)} models")
    
    # Create normalized confusion matrices
    print("\nCreating normalized confusion matrices...")
    print("-" * 60)
    create_normalized_confusion_matrices()
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"\nOutput directory: {COMPARISON_DIR}")
    print("\nGenerated files:")
    print("  1. all_confusion_matrices.png (absolute counts)")
    print("  2. all_confusion_matrices_hires.png (high-res version)")
    print("  3. all_confusion_matrices_normalized.png (percentages)")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
