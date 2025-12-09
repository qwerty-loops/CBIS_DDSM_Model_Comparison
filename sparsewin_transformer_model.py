"""
CBIS-DDSM SparseWin Transformer Model

Binary classification of mammography images using SparseWin Transformer architecture
with multi-token attention and sparse attention patterns for improved efficiency.
Evaluates on: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, Sensitivity, Specificity, MCC
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, matthews_corrcoef
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing modules (ENHANCED VERSION with corrected matching)
from data_preprocessing import create_data_loaders, OUTPUT_DIR_ENHANCED

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_CLASSES = 2
MODEL_NAME = "swin_tiny_patch4_window7_224"
OUTPUT_MODEL_DIR = r"d:\Allen Archive\Allen Archives\NEU_academics\Semester4\ML\Project\CBIS_DDSM_Model_Comparison\models"
OUTPUT_RESULTS_DIR = r"d:\Allen Archive\Allen Archives\NEU_academics\Semester4\ML\Project\CBIS_DDSM_Model_Comparison\results\sparsewin_transformer"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)


class SparseMultiTokenAttention(nn.Module):
    """
    Sparse Multi-Token Attention mechanism
    Implements sparse attention patterns to reduce computational complexity
    while maintaining multi-token interaction capabilities
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., 
                 sparse_ratio=0.5, num_tokens=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sparse_ratio = sparse_ratio
        self.num_tokens = num_tokens
        
        # Multi-token query, key, value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Token aggregation for multi-token attention
        self.token_mixer = nn.Linear(dim * num_tokens, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply sparse attention mask (keep top-k connections)
        if self.training and self.sparse_ratio < 1.0:
            # Compute sparsity threshold
            k_sparse = int(N * self.sparse_ratio)
            if k_sparse > 0:
                # Get top-k attention scores
                topk_vals, topk_indices = torch.topk(attn, k=k_sparse, dim=-1)
                # Create sparse mask
                sparse_mask = torch.zeros_like(attn).scatter_(-1, topk_indices, 1.0)
                attn = attn * sparse_mask + (1 - sparse_mask) * (-1e9)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Multi-token aggregation
        if N >= self.num_tokens:
            # Sample multiple tokens for aggregation
            token_indices = torch.linspace(0, N-1, self.num_tokens, dtype=torch.long, device=x.device)
            multi_tokens = x[:, token_indices, :].reshape(B, -1)
            aggregated = self.token_mixer(multi_tokens).unsqueeze(1)
            # Blend with original
            x = x + 0.1 * aggregated
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SparseWinBlock(nn.Module):
    """
    SparseWin Transformer Block with sparse multi-token attention
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., sparse_ratio=0.5, num_tokens=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparseMultiTokenAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
            sparse_ratio=sparse_ratio, num_tokens=num_tokens
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Stochastic depth
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        # Attention with residual
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SparseWinTransformerClassifier(nn.Module):
    """
    SparseWin Transformer model for binary classification
    Uses sparse attention patterns and multi-token attention for efficiency
    """
    
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=2, 
                 pretrained=True, num_sparse_blocks=2, sparse_ratio=0.5, num_tokens=4):
        super(SparseWinTransformerClassifier, self).__init__()
        
        # Load pretrained Swin Transformer as backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features
        
        # Add sparse attention blocks
        self.sparse_blocks = nn.ModuleList([
            SparseWinBlock(
                dim=self.feature_dim,
                num_heads=8,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.1,
                attn_drop=0.1,
                drop_path=0.1,
                sparse_ratio=sparse_ratio,
                num_tokens=num_tokens
            )
            for _ in range(num_sparse_blocks)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(self.feature_dim)
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Reshape for attention blocks (add sequence dimension)
        B = features.shape[0]
        features = features.unsqueeze(1)  # (B, 1, feature_dim)
        
        # Apply sparse attention blocks
        for block in self.sparse_blocks:
            features = block(features)
        
        # Global pooling
        features = features.mean(dim=1)  # (B, feature_dim)
        
        # Normalize and classify
        features = self.norm(features)
        output = self.head(features)
        
        return output


class MetricsCalculator:
    """Calculate all evaluation metrics"""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_pred_proba):
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_true, y_pred_proba),
            'PR_AUC': average_precision_score(y_true, y_pred_proba),
            'Sensitivity_TPR': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Specificity_TNR': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, phase='Test'):
        print(f"\n{phase} Metrics - SparseWin Transformer:")
        for metric_name, value in metrics.items():
            print(f"{metric_name.replace('_', ' ')}: {value:.4f}")


class Visualizer:
    """Create and save visualizations"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, output_path):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix - SparseWin Transformer', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: {output_path}")
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, output_path):
        """Plot and save ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - SparseWin Transformer', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved: {output_path}")
    
    @staticmethod
    def plot_pr_curve(y_true, y_pred_proba, output_path):
        """Plot and save Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve - SparseWin Transformer', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PR curve saved: {output_path}")
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_path):
        """Plot and save training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss', marker='o', markersize=4)
        ax1.plot(val_losses, label='Val Loss', marker='s', markersize=4)
        ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Accuracy plot
        ax2.plot(train_accs, label='Train Accuracy', marker='o', markersize=4)
        ax2.plot(val_accs, label='Val Accuracy', marker='s', markersize=4)
        ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history saved: {output_path}")


class Trainer:
    """Handle model training"""
    
    def __init__(self, model, device, train_loader, val_loader):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Calculate class weights to handle imbalance
        class_weights = self._calculate_class_weights()
        print(f"\nClass weights: {class_weights}")
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        
        # Early stopping
        self.early_stop_patience = 10
        self.epochs_without_improvement = 0
    
    def _calculate_class_weights(self):
        """Calculate class weights based on training data distribution"""
        print("\nCalculating class weights...")
        class_counts = [0, 0]
        
        for _, labels in self.train_loader:
            for label in labels:
                class_counts[label.item()] += 1
        
        total = sum(class_counts)
        # Inverse frequency weighting
        weights = [total / (len(class_counts) * count) for count in class_counts]
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        print(f"Class distribution: Class 0 (Benign): {class_counts[0]}, Class 1 (Malignant): {class_counts[1]}")
        print(f"Computed weights: Class 0: {weights[0]:.4f}, Class 1: {weights[1]:.4f}")
        
        return weights_tensor
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs):
        """Complete training loop"""
        print("\nStarting SparseWin Transformer training")
        print(f"Device: {self.device} | Epochs: {num_epochs} | LR: {LEARNING_RATE}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model and early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                best_model_path = os.path.join(OUTPUT_MODEL_DIR, 'sparsewin_transformer_best.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Best model saved with val_loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epoch(s)")
                
                if self.epochs_without_improvement >= self.early_stop_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best validation loss: {self.best_val_loss:.4f}")
                    break
        
        print("\nTraining complete")


class Evaluator:
    """Handle model evaluation"""
    
    def __init__(self, model, device, test_loader):
        self.model = model
        self.device = device
        self.test_loader = test_loader
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\nEvaluating on test set...")
        
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_pred_proba = np.array(all_probs)
        
        return y_true, y_pred, y_pred_proba


def save_metrics_to_csv(metrics, output_path):
    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)
    print(f"Metrics saved: {output_path}")


def main():
    print("\nSparseWin Transformer Model Training")
    print("-" * 40)
    print("Features: Sparse Multi-Token Attention + Efficient Attention Patterns")
    
    # Create data loaders (LEAKY VERSION)
    print("\nLoading preprocessed data (ENHANCED with corrected matching)...")
    train_loader, val_loader, test_loader = create_data_loaders(
        output_dir=OUTPUT_DIR_ENHANCED,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=4  # Parallel data loading for faster training
    )
    
    # Initialize model
    print(f"\nInitializing SparseWin Transformer (base: {MODEL_NAME})...")
    model = SparseWinTransformerClassifier(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=True,
        num_sparse_blocks=2,
        sparse_ratio=0.5,  # Keep 50% of attention connections
        num_tokens=4  # Multi-token aggregation with 4 tokens
    )
    model = model.to(DEVICE)
    print(f"Model loaded on {DEVICE}")
    print(f"Sparse attention ratio: 0.5 (50% sparsity)")
    print(f"Multi-token attention: 4 tokens")
    
    # Train model
    trainer = Trainer(model, DEVICE, train_loader, val_loader)
    trainer.train(NUM_EPOCHS)
    
    # Save training history
    history_path = os.path.join(OUTPUT_RESULTS_DIR, 'training_history.png')
    visualizer = Visualizer()
    visualizer.plot_training_history(
        trainer.train_losses, trainer.val_losses,
        trainer.train_accs, trainer.val_accs,
        history_path
    )
    
    # Load best model for evaluation
    print("\nLoading best model for final evaluation...")
    best_model_path = os.path.join(OUTPUT_MODEL_DIR, 'sparsewin_transformer_best.pth')
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate on test set
    evaluator = Evaluator(model, DEVICE, test_loader)
    y_true, y_pred, y_pred_proba = evaluator.evaluate()
    
    # Save predictions for comparison
    predictions_path = os.path.join(OUTPUT_RESULTS_DIR, 'predictions.npz')
    np.savez(predictions_path, y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba)
    print(f"Predictions saved: {predictions_path}")
    
    # Calculate metrics
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(y_true, y_pred, y_pred_proba)
    metrics_calc.print_metrics(metrics, phase='Test')
    
    # Save metrics
    metrics_csv_path = os.path.join(OUTPUT_RESULTS_DIR, 'test_metrics.csv')
    save_metrics_to_csv(metrics, metrics_csv_path)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    visualizer = Visualizer()
    
    # Confusion Matrix
    cm_path = os.path.join(OUTPUT_RESULTS_DIR, 'confusion_matrix.png')
    visualizer.plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # ROC Curve
    roc_path = os.path.join(OUTPUT_RESULTS_DIR, 'roc_curve.png')
    visualizer.plot_roc_curve(y_true, y_pred_proba, roc_path)
    
    # PR Curve
    pr_path = os.path.join(OUTPUT_RESULTS_DIR, 'pr_curve.png')
    visualizer.plot_pr_curve(y_true, y_pred_proba, pr_path)
    
    print("\nEvaluation complete!")
    print(f"Results: {OUTPUT_RESULTS_DIR}")
    print(f"Model: {OUTPUT_MODEL_DIR}")


if __name__ == "__main__":
    main()
