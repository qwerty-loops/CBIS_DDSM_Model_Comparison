# CBIS-DDSM Breast Cancer Classification

This project compares deep learning models for binary classification of mammography images from the CBIS-DDSM dataset.

**Task:** Classify mammograms as Benign (0) or Malignant (1)

## Overview

The project includes 8 models trained on 3,103 full mammogram images with corrected SeriesInstanceUID-based matching:

1. ConvNeXt-Tiny
2. DenseNet121  
3. ResNet18
4. ResNet50
5. Swin Transformer
6. SparseWin Transformer
7. EfficientNet-B0
8. U-Net (adapted for classification)

All models use the same enhanced preprocessing pipeline with CLAHE and negative transformation, consistent training procedures, and comprehensive evaluation metrics.

---

## Requirements

### Python version

- Python 3.8 or newer

### Main libraries

```text
numpy
pandas
matplotlib
seaborn
torch
torchvision
timm
Pillow
scikit-learn
tqdm
```

### Install

```bash
pip install numpy pandas matplotlib seaborn torch torchvision timm Pillow scikit-learn tqdm
```

If you have a CUDA-capable GPU, install the matching PyTorch build from the official website. The code will use GPU automatically if it is available.

---

## Project Structure

```
CBIS_DDSM_Model_Comparison/
│
├── data_preprocessing.py          # Enhanced preprocessing with corrected matching
├── convnext_model.py              # ConvNeXt model
├── densenet121_model.py           # DenseNet121 model
├── resnet18_model.py              # ResNet18 model
├── resnet50_model.py              # ResNet50 model
├── swin_transformer_model.py      # Swin Transformer
├── sparsewin_transformer_model.py # SparseWin Transformer
├── efficientnet_model.py          # EfficientNet-B0 model
├── unet_model.py                  # U-Net model
├── compare_models.py              # Model comparison
├── visualize_all_confusion_matrices.py  # Confusion matrix grid visualization
├── run_pipeline.py                # Automated training pipeline
│
├── preprocessed_data_enhanced/    # Enhanced dataset (3,103 images)
│   ├── train_enhanced.csv         # 2,165 images
│   ├── val_enhanced.csv           # 467 images
│   └── test_enhanced.csv          # 471 images
│
├── models/                        # Trained model weights
│   └── [model_name]_best.pth
│
└── results/                       # Evaluation results
    ├── convnext/
    ├── densenet121/
    ├── resnet18/
    ├── resnet50/
    ├── swin_transformer/
    ├── sparsewin_transformer/
    ├── efficientnet/
    ├── unet/
    └── comparison/                # Comparison visualizations
        ├── all_models_comparison.csv
        ├── all_confusion_matrices.png
        ├── metrics_bar_chart.png
        ├── roc_comparison.png
        └── pr_comparison.png
```

---

## Dataset

**Source:** CBIS-DDSM (Curated Breast Imaging Subset of DDSM)

**Enhanced Dataset:**
- 3,103 full mammogram images (MASS + CALCIFICATION cases)
- Matched using SeriesInstanceUID (corrected from previous PatientID matching)
- Train: 2,165 | Val: 467 | Test: 471
- Class distribution: Benign (56%) / Malignant (44%)

**Preprocessing:**
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Negative transformation for mammography
- Data augmentation (flips, rotations, color jitter)
- Resize to 224×224

---

## Quick Start

1. **Preprocessing** (creates enhanced dataset):
   ```bash
   python data_preprocessing.py
   ```

2. **Train models** (individually or use pipeline):
   ```bash
   # Individual
   python swin_transformer_model.py
   
   # Or run all models automatically
   python run_pipeline.py
   ```

3. **Compare results**:
   ```bash
   python compare_models.py
   python visualize_all_confusion_matrices.py
   ```

4. Check `results/comparison/` for visualizations and metrics.

---

## Step-by-Step Usage

### 1. Preprocessing

**Script:** `data_preprocessing.py`

```bash
python data_preprocessing.py
```

**What it does:**
- Loads MASS and CALCIFICATION annotations from CBIS-DDSM
- Matches images using SeriesInstanceUID (corrected approach)
- Scans 10,237 JPEG files and filters for full mammograms
- Creates train/val/test splits (70/15/15)
- Applies CLAHE preprocessing and negative transformation

**Output:** `preprocessed_data_enhanced/`
- `train_enhanced.csv` (2,165 images)
- `val_enhanced.csv` (467 images)  
- `test_enhanced.csv` (471 images)

### 2. Training Models

**Configuration:**
- Image size: 224×224
- Batch size: 16-32 (model dependent)
- Max epochs: 30
- Learning rate: 1e-4
- Optimizer: AdamW (weight_decay: 0.05)
- Loss: CrossEntropyLoss with class weights
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=7

**Anti-overfitting measures:**
- Dropout (0.3-0.5 depending on model)
- Strong weight decay (0.05)
- Early stopping
- Data augmentation

**Each model:**
- Loads from `preprocessed_data_enhanced/`
- Trains with validation monitoring
- Saves best weights to `models/`
- Evaluates on test set
- Saves metrics and visualizations to `results/[model_name]/`

Run any of the following:

#### ConvNeXt

```bash
python convnext_model.py
```

Output:

- `models/convnext_best.pth`
- `results/convnext/` (metrics, plots, predictions)

#### DenseNet121

```bash
python densenet121_model.py
```

Output:

- `models/densenet121_best.pth`
- `results/densenet121/`

#### ResNet18

```bash
python resnet18_model.py
```

Output:

- `models/resnet18_best.pth`
- `results/resnet18/`

#### ResNet50

```bash
python resnet50_model.py
```

Output:

- `models/resnet50_best.pth`
- `results/resnet50/`

#### Swin Transformer

```bash
python swin_transformer_model.py
```

Main points:

- Backbone: `swin_tiny_patch4_window7_224` (ImageNet pretrained)
- Standard single-image classification

Output:

- `models/swin_transformer_best.pth`
- `results/swin_transformer/`

#### EfficientNet-B0

```bash
python efficientnet_model.py
```

Main points:

- Backbone: EfficientNet-B0 (ImageNet pretrained)
- Compound scaling method balancing network depth, width, and resolution
- Efficient architecture with excellent accuracy-to-parameter ratio

Output:

- `models/efficientnet_best.pth`
- `results/efficientnet/`

#### U-Net (Classification)

```bash
python unet_model.py
```

Main points:

- Traditional U-Net architecture adapted for classification
- Encoder-decoder structure with skip connections
- Global pooling + classification head for binary classification
- No pretrained weights (trained from scratch)

Output:

- `models/unet_best.pth`
- `results/unet/`

---

## Models

| Model | Type | Parameters | ROC AUC | Notes |
|-------|------|------------|---------|-------|
| **Swin Transformer** | Transformer | 28M | 0.824 | Best overall performance |
| **SparseWin Transformer** | Transformer | 28M | 0.822 | Sparse attention variant |
| **DenseNet121** | CNN | 8M | 0.817 | Dense connections |
| **ConvNeXt-Tiny** | CNN | 28M | 0.807 | Modern CNN design |
| **ResNet50** | CNN | 25M | 0.805 | Deep residual network |
| **ResNet18** | CNN | 11M | 0.795 | Lightweight baseline |
| **EfficientNet-B0** | CNN | 5M | 0.769 | Efficient scaling |
| **U-Net** | Encoder-Decoder | 31M | 0.621 | Adapted from segmentation |

---

## 📊 Evaluation Metrics

Both models are evaluated using the following comprehensive metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **Precision** | Positive predictive value (TP / (TP + FP)) |
| **Recall** | True positive rate (TP / (TP + FN)) |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC AUC** | Area under the ROC curve |
| **PR AUC** | Area under the Precision-Recall curve |
| **Sensitivity (TPR)** | True Positive Rate, same as Recall |
| **Specificity (TNR)** | True Negative Rate (TN / (TN + FP)) |
| **MCC** | Matthews Correlation Coefficient - balanced measure for imbalanced datasets |

### Why These Metrics?

- **Accuracy** - Overall performance indicator
- **Precision** - Important to minimize false positives (avoid unnecessary biopsies)
- **Recall/Sensitivity** - Critical to catch all malignant cases (minimize false negatives)
- **F1** - Balance between precision and recall
- **ROC AUC** - Model's ability to discriminate between classes
- **PR AUC** - Better for imbalanced datasets than ROC AUC
- **Specificity** - Ability to correctly identify benign cases
- **MCC** - Single metric accounting for all confusion matrix elements

---

## 📈 Visualizations

Each model generates the following visualizations:

### 1. Confusion Matrix
- Shows true positives, false positives, true negatives, and false negatives
- Helps identify specific classification errors
- Saved as: `confusion_matrix.png`

### 2. ROC Curve
- Plots True Positive Rate vs False Positive Rate
- Shows model's discriminative ability across thresholds
- Includes AUC score
- Saved as: `roc_curve.png`

### 3. Precision-Recall Curve
- Plots Precision vs Recall
- Particularly useful for imbalanced datasets
- Includes PR AUC score
- Saved as: `pr_curve.png`

### 4. Training History
- Shows training and validation loss over epochs
- Shows training and validation accuracy over epochs
- Helps identify overfitting or underfitting
- Saved as: `training_history.png`

---

## 📋 Results

Results are saved in the following locations:

### ConvNeXt Results
- **Directory:** `results/convnext/`
- **Metrics CSV:** `test_metrics.csv`
- **Visualizations:**
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `pr_curve.png`
  - `training_history.png`

### DenseNet121 Results
- **Directory:** `results/densenet121/`
- **Metrics CSV:** `test_metrics.csv`
- **Visualizations:**
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `pr_curve.png`
  - `training_history.png`

---

## 🔧 Customization

### Adjusting Hyperparameters

You can modify the following parameters in the model scripts:

```python
# In convnext_model.py or densenet121_model.py

IMAGE_SIZE = 224          # Image dimensions
BATCH_SIZE = 16           # Batch size for training
NUM_EPOCHS = 30           # Number of training epochs
LEARNING_RATE = 1e-4      # Initial learning rate
```

### Data Augmentation

The preprocessing script includes data augmentation for training:
- Random horizontal flip
- Random rotation (±10 degrees)
- Color jitter (brightness and contrast)

You can modify these in `data_preprocessing.py`:

```python
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

---

## 💡 Tips for Best Results

1. **GPU Usage:** Training on GPU significantly speeds up the process. The code automatically detects and uses GPU if available.

2. **Batch Size:** Adjust based on your GPU memory. Reduce if you encounter out-of-memory errors.

3. **Early Stopping:** The code uses ReduceLROnPlateau scheduler and saves the best model based on validation loss.

4. **Data Balance:** Check the class distribution in the preprocessing output. Consider using weighted loss if classes are highly imbalanced.

5. **Evaluation:** Always evaluate on the test set (not used during training) for unbiased performance estimates.

---

## 🐛 Troubleshooting

### Issue: Out of Memory Error
**Solution:** Reduce `BATCH_SIZE` in the model scripts (try 8 or 4).

### Issue: Image Loading Errors
**Solution:** Run preprocessing first to validate all image paths.

### Issue: Slow Training on Windows
**Solution:** Set `num_workers=0` in DataLoader (already configured).

### Issue: CUDA Not Available
**Solution:** Install PyTorch with CUDA support or use CPU (training will be slower).

---

## Key Improvements

**Corrected Image-Label Matching:**
- Fixed SeriesInstanceUID extraction (was using position [1], now uses [2])
- Matches 3,103 images (up from 110 with old approach)
- Verified against reference notebook implementation

**Enhanced Preprocessing:**
- CLAHE for contrast enhancement
- Negative transformation for mammography
- Proper train/val/test splits

**Anti-Overfitting:**
- Dropout layers (0.3-0.5)
- Strong weight decay (0.05)
- Early stopping (patience=7)
- Data augmentation

## Results Summary

All 8 models achieve similar performance (ROC AUC 0.77-0.82), with transformers slightly outperforming CNNs. The close performance suggests the dataset difficulty ceiling has been reached with current approaches.

**Best Model:** Swin Transformer (ROC AUC: 0.824, F1: 0.723)

---

**Project:** CBIS-DDSM Breast Cancer Classification  
**Dataset:** 3,103 full mammogram images  
**Models:** 8 architectures (CNNs + Transformers)  
**Task:** Binary classification (Benign vs Malignant)
