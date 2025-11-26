# CBIS-DDSM Breast Cancer Classification

This project builds and compares several deep learning models for binary classification of mammography images from the CBIS-DDSM (Curated Breast Imaging Subset of DDSM) dataset. The goal is to classify each image as:

- 0 – Benign
- 1 – Malignant

The code handles data preprocessing, training, evaluation, visualization, and model comparison.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Step-by-Step Usage](#step-by-step-usage)
- [Models](#models)
- [Model Comparison](#model-comparison)
- [Evaluation Metrics](#evaluation-metrics)
- [Class Imbalance: What Was Fixed](#class-imbalance-what-was-fixed)
- [Visualizations](#visualizations)
- [Tips and Troubleshooting](#tips-and-troubleshooting)
---

## Overview

The project currently supports six models:

1. ConvNeXt-Tiny
2. DenseNet121
3. ResNet18
4. ResNet50
5. Swin Transformer (single-view)
6. Swin Transformer (multi-view variant)

All models share the same preprocessing pipeline, training loop style, metrics, and output structure. A separate comparison script summarizes and visualizes how they perform side by side.

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
CBIS_DDSM_CNN/
│
├── data_preprocessing.py          # Data preprocessing and cleaning
├── convnext_model.py              # ConvNeXt model
├── densenet121_model.py           # DenseNet121 model
├── resnet18_model.py              # ResNet18 model
├── resnet50_model.py              # ResNet50 model
├── swin_transformer_model.py      # Swin Transformer (single-view)
├── swin_transformer_multiview_model.py  # Swin Transformer (multi-view)
├── compare_models.py              # Model comparison and plots
├── README.md                      # Main documentation (this file)
│
├── preprocessed_data/             # Output from preprocessing (created automatically)
│   ├── train_cleaned.csv
│   ├── val_cleaned.csv
│   ├── test_cleaned.csv
│   └── preprocessing_summary.txt
│
├── models/                        # Trained model weights (created automatically)
│   ├── convnext_best.pth
│   ├── densenet121_best.pth
│   ├── resnet18_best.pth
│   ├── resnet50_best.pth
│   ├── swin_transformer_best.pth
│   └── swin_transformer_multiview_best.pth
│
└── results/                       # Evaluation results (created automatically)
    ├── convnext/
    ├── densenet121/
    ├── resnet18/
    ├── resnet50/
    ├── swin_transformer/
    ├── swin_transformer_multiview/
    └── comparison/                # Cross-model comparison tables and plots
```

---

## Dataset

- Source: CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
- Location on disk (in this project):  
  `D:\Allen Archive\Allen Archives\NEU_academics\Semester4\ML\Project\CBIS_DDSM`

Main pieces you need:

- `split_train.csv`, `split_val.csv`, `split_test.csv` – metadata and labels
- Image folders (JPEG/PNG) referenced by those CSVs

Classes:

- Class 0: Benign
- Class 1: Malignant

---

## Quick Start

1. Run preprocessing (once):
   ```bash
   python data_preprocessing.py
   ```
2. Train one or more models, for example:
   ```bash
   python convnext_model.py
   python densenet121_model.py
   python resnet18_model.py
   python resnet50_model.py
   python swin_transformer_model.py
   python swin_transformer_multiview_model.py
   ```
3. Run the comparison script:
   ```bash
   python compare_models.py
   ```
4. Open the outputs in `results/` and `results/comparison/` to inspect metrics and plots.

---

## Step-by-Step Usage

### 1. Preprocessing

Script: `data_preprocessing.py`

```bash
python data_preprocessing.py
```

What it does:

- Loads train, validation, and test CSV files
- Checks for missing values
- Verifies that image paths exist
- Checks a sample of images for corruption
- Removes duplicate image entries
- Prints class distributions
- Saves cleaned CSVs
- Writes a short summary report

Outputs in `preprocessed_data/`:

- `train_cleaned.csv`
- `val_cleaned.csv`
- `test_cleaned.csv`
- `preprocessing_summary.txt`

### 2. Training and Evaluation – Models

All model scripts share the same basic configuration:

- Image size: 224×224
- Batch size: 16
- Epochs: 30
- Learning rate: 1e-4
- Optimizer: AdamW
- Loss: CrossEntropyLoss with class weights
- Scheduler: ReduceLROnPlateau

Each script:

- Trains on the training set
- Monitors validation loss
- Saves the best model weights
- Evaluates on the test set
- Computes a full set of metrics
- Saves predictions for comparison
- Produces several plots

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

#### Swin Transformer (single-view)

```bash
python swin_transformer_model.py
```

Main points:

- Backbone: `swin_tiny_patch4_window7_224` (ImageNet pretrained)
- Standard single-image classification

Output:

- `models/swin_transformer_best.pth`
- `results/swin_transformer/`

#### Swin Transformer (multi-view)

```bash
python swin_transformer_multiview_model.py
``;

Main points:

- Designed to be extendable to multiple views per patient
- Currently uses a similar backbone and training style as single-view

Output:

- `models/swin_transformer_multiview_best.pth`
- `results/swin_transformer_multiview/`

---

## Models

Short description of each model:

- **ConvNeXt-Tiny** – Modern CNN architecture inspired by Vision Transformers, but purely convolutional.
- **DenseNet121** – CNN with dense connections; good feature reuse and strong gradients.
- **ResNet18 / ResNet50** – Classic residual networks; stable and widely used baselines.
- **Swin Transformer (single-view)** – Vision Transformer with shifted windows, good at capturing both local and global patterns.
- **Swin Transformer (multi-view)** – Variant intended for combining information from multiple views; currently run on single views but keeps the design flexible.

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

## 📚 References

1. **CBIS-DDSM Dataset:** [https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)

2. **ConvNeXt Paper:** Liu, Z., et al. (2022). "A ConvNet for the 2020s." CVPR 2022.

3. **DenseNet Paper:** Huang, G., et al. (2017). "Densely Connected Convolutional Networks." CVPR 2017.

---

## 📝 Notes

- All output directories are created automatically
- Models are saved only when validation loss improves
- All metrics are computed on the test set using the best model checkpoint
- Preprocessing needs to be run only once unless data changes
- Both models use transfer learning with ImageNet pretrained weights

---

## ✅ Checklist

Before running the models, ensure:

- [ ] Dataset is properly organized in the specified directory
- [ ] Python 3.8+ is installed
- [ ] All required packages are installed
- [ ] There is enough disk space for models and results (~500MB)
- [ ] GPU available (optional but recommended)
- [ ] Preprocessing script has been run successfully


**Project:** CBIS-DDSM Breast Cancer Classification  
**Models:** ConvNeXt-Tiny, DenseNet121, ResNet18, ResNet50, Swin Transformer (single/multi-view)  
**Task:** Binary classification (Benign vs Malignant)
