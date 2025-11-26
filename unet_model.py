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
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing modules
from data_preprocessing import create_data_loaders, OUTPUT_DIR


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_CLASSES = 2
MODEL_NAME = "unet"
OUTPUT_MODEL_DIR = r"d:\Allen Archive\Allen Archives\NEU_academics\Semester4\ML\Project\CBIS_DDSM_Model_Comparison\models"
OUTPUT_RESULTS_DIR = r"d:\Allen Archive\Allen Archives\NEU_academics\Semester4\ML\Project\CBIS_DDSM_Model_Comparison\results\unet"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetClassifier(nn.Module):
    """U-Net architecture adapted for binary classification
    
    Traditional U-Net is for segmentation, but we adapt it for classification by:
    1. Using the encoder path to extract hierarchical features
    2. Using the decoder path to refine features
    3. Adding global pooling and classification head at the end
    """
    
    def __init__(self, num_classes=2, in_channels=3):
        super(UNetClassifier, self).__init__()
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder (upsampling path)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Classification
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x


class MetricsCalculator:
    """Calculate all evaluation metrics"""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_pred_proba):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
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
        print(f"\n{phase} Metrics - {MODEL_NAME}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name.replace('_', ' ')}: {value:.4f}")


class Visualizer:
    """Create and save visualizations"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, output_path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {MODEL_NAME.upper()}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: {output_path}")
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, output_path):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {MODEL_NAME.upper()}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved: {output_path}")
    
    @staticmethod
    def plot_pr_curve(y_true, y_pred_proba, output_path):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {MODEL_NAME.upper()}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PR curve saved: {output_path}")
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(train_losses, label='Train Loss', marker='o')
        ax1.plot(val_losses, label='Validation Loss', marker='s')
        ax1.set_xlabel('Epoch', fontsize=12); ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend(); ax1.grid(alpha=0.3)
        
        ax2.plot(train_accs, label='Train Accuracy', marker='o')
        ax2.plot(val_accs, label='Validation Accuracy', marker='s')
        ax2.set_xlabel('Epoch', fontsize=12); ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.legend(); ax2.grid(alpha=0.3)
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
        
        class_weights = self._calculate_class_weights()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        self.train_losses = []; self.val_losses = []
        self.train_accs = []; self.val_accs = []
        self.best_val_loss = float('inf')
    
    def _calculate_class_weights(self):
        print("\nCalculating class weights...")
        class_counts = [0, 0]
        for _, labels in self.train_loader:
            for label in labels: class_counts[label.item()] += 1
        total = sum(class_counts)
        weights = [total / (len(class_counts) * count) for count in class_counts]
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        print(f"Class distribution: {class_counts}")
        print(f"Computed weights: {weights}")
        return weights_tensor
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0; correct = 0; total = 0
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        return running_loss / total, correct / total
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss / total, correct / total
    
    def train(self, num_epochs):
        print(f"\nStarting {MODEL_NAME.upper()} training")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss); self.train_accs.append(train_acc)
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss); self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(OUTPUT_MODEL_DIR, f'{MODEL_NAME}_best.pth'))
                print(f"Best model saved.")


class Evaluator:
    """Handle model evaluation"""
    def __init__(self, model, device, test_loader):
        self.model = model; self.device = device; self.test_loader = test_loader
    
    def evaluate(self):
        print("\nEvaluating on test set...")
        self.model.eval()
        all_labels = []; all_preds = []; all_probs = []
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def save_metrics_to_csv(metrics, output_path):
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    print(f"Metrics saved: {output_path}")


def main():
    print(f"\n{MODEL_NAME.upper()} Model Training")
    print("-" * 40)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        OUTPUT_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=0
    )
    
    print(f"\nInitializing {MODEL_NAME.upper()} model...")
    print("Note: U-Net is adapted for classification with encoder-decoder + global pooling")
    model = UNetClassifier(num_classes=NUM_CLASSES, in_channels=3).to(DEVICE)
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = Trainer(model, DEVICE, train_loader, val_loader)
    trainer.train(NUM_EPOCHS)
    
    Visualizer.plot_training_history(
        trainer.train_losses, trainer.val_losses,
        trainer.train_accs, trainer.val_accs,
        os.path.join(OUTPUT_RESULTS_DIR, 'training_history.png')
    )
    
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_MODEL_DIR, f'{MODEL_NAME}_best.pth')))
    
    evaluator = Evaluator(model, DEVICE, test_loader)
    y_true, y_pred, y_pred_proba = evaluator.evaluate()
    
    np.savez(os.path.join(OUTPUT_RESULTS_DIR, 'predictions.npz'), 
             y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba)
    
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(y_true, y_pred, y_pred_proba)
    metrics_calc.print_metrics(metrics, phase='Test')
    save_metrics_to_csv(metrics, os.path.join(OUTPUT_RESULTS_DIR, 'test_metrics.csv'))
    
    visualizer = Visualizer()
    visualizer.plot_confusion_matrix(y_true, y_pred, os.path.join(OUTPUT_RESULTS_DIR, 'confusion_matrix.png'))
    visualizer.plot_roc_curve(y_true, y_pred_proba, os.path.join(OUTPUT_RESULTS_DIR, 'roc_curve.png'))
    visualizer.plot_pr_curve(y_true, y_pred_proba, os.path.join(OUTPUT_RESULTS_DIR, 'pr_curve.png'))
    
    print(f"\nEvaluation complete!\nResults: {OUTPUT_RESULTS_DIR}")

if __name__ == "__main__":
    main()
