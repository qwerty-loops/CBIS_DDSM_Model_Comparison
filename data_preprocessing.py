"""
CBIS-DDSM Data Preprocessing

Loads, cleans, and prepares the CBIS-DDSM mammography dataset.
Creates cleaned CSV files and PyTorch DataLoaders for model training.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Dataset paths
BASE = r"D:\Allen Archive\Allen Archives\NEU_academics\Semester4\ML\Project\CBIS_DDSM"
CSV_DIR = os.path.join(BASE, "csv")
JPEG_DIR = os.path.join(BASE, "jpeg")
PNG_DIR = os.path.join(BASE, "png")

# Output directory for preprocessed data
OUTPUT_DIR = r"d:\Allen Archive\Allen Archives\NEU_academics\Semester4\ML\Project\CBIS_DDSM_CNN\preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DataPreprocessor:
    """Handles all data preprocessing operations for CBIS-DDSM dataset"""
    
    def __init__(self, base_dir, output_dir):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def load_data(self):
        """Load train, validation, and test CSV files"""
        print("\nLoading data...")
        
        train_path = os.path.join(self.base_dir, "split_train.csv")
        val_path = os.path.join(self.base_dir, "split_val.csv")
        test_path = os.path.join(self.base_dir, "split_test.csv")
        
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}")
        
        return self.train_df, self.val_df, self.test_df
    
    def check_missing_values(self, df, split_name):
        """Check for missing values in dataset"""
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print(f"{split_name}: No missing values")
        else:
            print(f"{split_name} missing values:\n{missing[missing > 0]}")
        return missing
    
    def verify_image_paths(self, df, split_name):
        """Verify that all image paths exist"""
        missing_images = []
        for idx, row in df.iterrows():
            img_path = row['jpg_path']
            if not os.path.exists(img_path):
                missing_images.append(img_path)
        
        if len(missing_images) == 0:
            print(f"{split_name}: All {len(df)} images found")
        else:
            print(f"{split_name}: WARNING - {len(missing_images)} images not found")
            
        return missing_images
    
    def check_image_quality(self, df, split_name, sample_size=100):
        """Check a sample of images for quality issues"""
        print(f"{split_name}: Checking image quality (sample={sample_size})...")
        
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        corrupt_images = []
        
        for idx, row in sample_df.iterrows():
            img_path = row['jpg_path']
            try:
                img = Image.open(img_path)
                img.verify()  # Verify image integrity
                img = Image.open(img_path)  # Reopen after verify
                img_array = np.array(img)
                
                # Check if image is not empty
                if img_array.size == 0:
                    corrupt_images.append(img_path)
                    
            except Exception as e:
                corrupt_images.append(img_path)
                
        if len(corrupt_images) == 0:
            print(f"{split_name}: Images OK")
        else:
            print(f"{split_name}: WARNING - {len(corrupt_images)} corrupt images")
            
        return corrupt_images
    
    def analyze_class_distribution(self):
        """Analyze and display class distribution"""
        print("\nClass Distribution:")
        
        splits = [('Train', self.train_df), ('Validation', self.val_df), ('Test', self.test_df)]
        
        for split_name, df in splits:
            class_counts = df['label'].value_counts()
            total = len(df)
            
            print(f"{split_name}: Benign={class_counts.get(0, 0)} ({class_counts.get(0, 0)/total*100:.1f}%), Malignant={class_counts.get(1, 0)} ({class_counts.get(1, 0)/total*100:.1f}%)")
    
    def remove_duplicates(self, df, split_name):
        """Remove duplicate entries based on image path"""
        original_len = len(df)
        df_cleaned = df.drop_duplicates(subset=['jpg_path'], keep='first')
        removed = original_len - len(df_cleaned)
        
        if removed > 0:
            print(f"{split_name}: Removed {removed} duplicates")
            
        return df_cleaned
    
    def clean_data(self):
        """Perform comprehensive data cleaning"""
        print("\nStarting data cleaning...")
        
        # Check missing values
        for df, name in [(self.train_df, 'Train'), (self.val_df, 'Validation'), (self.test_df, 'Test')]:
            self.check_missing_values(df, name)
        
        # Verify image paths
        for df, name in [(self.train_df, 'Train'), (self.val_df, 'Validation'), (self.test_df, 'Test')]:
            self.verify_image_paths(df, name)
        
        # Check image quality
        for df, name in [(self.train_df, 'Train'), (self.val_df, 'Validation'), (self.test_df, 'Test')]:
            self.check_image_quality(df, name)
        
        # Remove duplicates
        self.train_df = self.remove_duplicates(self.train_df, 'Train')
        self.val_df = self.remove_duplicates(self.val_df, 'Validation')
        self.test_df = self.remove_duplicates(self.test_df, 'Test')
        
        # Analyze class distribution
        self.analyze_class_distribution()
        
        print("\nData cleaning complete.")
    
    def save_cleaned_data(self):
        """Save cleaned dataframes to CSV"""
        train_path = os.path.join(self.output_dir, "train_cleaned.csv")
        val_path = os.path.join(self.output_dir, "val_cleaned.csv")
        test_path = os.path.join(self.output_dir, "test_cleaned.csv")
        
        self.train_df.to_csv(train_path, index=False)
        self.val_df.to_csv(val_path, index=False)
        self.test_df.to_csv(test_path, index=False)
        
        print(f"\nSaved cleaned data to {self.output_dir}")
    
    def create_summary_report(self):
        """Create a summary report of preprocessing"""
        report = {
            'Dataset': 'CBIS-DDSM Breast Cancer Mammography',
            'Total_Samples': len(self.train_df) + len(self.val_df) + len(self.test_df),
            'Train_Samples': len(self.train_df),
            'Val_Samples': len(self.val_df),
            'Test_Samples': len(self.test_df),
            'Train_Class0': int((self.train_df['label'] == 0).sum()),
            'Train_Class1': int((self.train_df['label'] == 1).sum()),
            'Val_Class0': int((self.val_df['label'] == 0).sum()),
            'Val_Class1': int((self.val_df['label'] == 1).sum()),
            'Test_Class0': int((self.test_df['label'] == 0).sum()),
            'Test_Class1': int((self.test_df['label'] == 1).sum()),
        }
        
        report_path = os.path.join(self.output_dir, "preprocessing_summary.txt")
        with open(report_path, 'w') as f:
            f.write("CBIS-DDSM Preprocessing Summary\n\n")
            for key, value in report.items():
                f.write(f"{key.replace('_', ' ')}: {value}\n")
        
        print(f"Summary saved: {report_path}")
        return report


class CBISDDSMDataset(Dataset):
    """Custom PyTorch Dataset for CBIS-DDSM images"""
    
    def __init__(self, csv_path, transform=None, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_size = image_size
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row['jpg_path']
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return image, label


def get_data_transforms(image_size=224):
    # Training with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation and test (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(output_dir, batch_size=32, image_size=224, num_workers=4):
    print("\nCreating data loaders...")
    
    train_transform, val_transform = get_data_transforms(image_size)
    
    train_dataset = CBISDDSMDataset(
        csv_path=os.path.join(output_dir, "train_cleaned.csv"),
        transform=train_transform,
        image_size=image_size
    )
    
    val_dataset = CBISDDSMDataset(
        csv_path=os.path.join(output_dir, "val_cleaned.csv"),
        transform=val_transform,
        image_size=image_size
    )
    
    test_dataset = CBISDDSMDataset(
        csv_path=os.path.join(output_dir, "test_cleaned.csv"),
        transform=val_transform,
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def main():
    print("\nCBIS-DDSM Data Preprocessing")
    print("-" * 40)
    
    preprocessor = DataPreprocessor(BASE, OUTPUT_DIR)
    
    preprocessor.load_data()
    
    preprocessor.clean_data()
    
    preprocessor.save_cleaned_data()
    
    preprocessor.create_summary_report()
    
    print("\nTesting data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        OUTPUT_DIR,
        batch_size=32,
        image_size=224,
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    images, labels = next(iter(train_loader))
    print(f"Batch: {images.shape}, dtype: {images.dtype}")
    print(f"Labels: {labels.shape}, dtype: {labels.dtype}")
    
    print("\nPreprocessing complete!")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
