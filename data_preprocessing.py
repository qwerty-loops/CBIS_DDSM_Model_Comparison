"""
CBIS-DDSM Data Preprocessing

Loads, cleans, and prepares the CBIS-DDSM mammography dataset.
Creates cleaned CSV files and PyTorch DataLoaders for model training.

Features:
- Standard preprocessing: Uses existing split files (110 images)
- Enhanced preprocessing: Processes full CBIS-DDSM dataset (5000+ images)
  by matching all JPEG images with annotations from CSV files
  Includes both MASS and CALCIFICATION cases
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
import re
import glob
from sklearn.model_selection import train_test_split
import cv2
import warnings
warnings.filterwarnings('ignore')

# Dataset paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "Images")
CSV_DIR = os.path.join(BASE, "csv")
JPEG_DIR = os.path.join(BASE, "jpeg")
PNG_DIR = os.path.join(BASE, "png")

# Output directories
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "preprocessed_data")
OUTPUT_DIR_ENHANCED = os.path.join(SCRIPT_DIR, "preprocessed_data_enhanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_ENHANCED, exist_ok=True)

# Enhanced dataset paths
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
BASE_ENHANCED = os.path.join(PARENT_DIR, "CBIS_DDSM")
CSV_DIR_ENHANCED = os.path.join(BASE_ENHANCED, "csv")
JPEG_DIR_ENHANCED = os.path.join(BASE_ENHANCED, "jpeg")


class DataPreprocessor:
    """Handles all data preprocessing operations for CBIS-DDSM dataset"""
    
    def __init__(self, base_dir, output_dir):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def load_data(self, use_enhanced=True):
        """Load train, validation, and test CSV files
        
        Args:
            use_enhanced: If True, tries to load enhanced dataset (3000+ images)
                         If False or enhanced not found, uses original dataset (110 images)
        """
        print("\nLoading data...")
        
        # Try to use enhanced dataset first
        # Get the script directory (CBIS_DDSM_Model_Comparison)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        enhanced_dir = os.path.join(script_dir, "preprocessed_data_enhanced")
        
        if use_enhanced and os.path.exists(enhanced_dir):
            train_path = os.path.join(enhanced_dir, "train_enhanced.csv")
            val_path = os.path.join(enhanced_dir, "val_enhanced.csv")
            test_path = os.path.join(enhanced_dir, "test_enhanced.csv")
            
            if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
                print("  Using ENHANCED dataset (5000+ images)")
                self.train_df = pd.read_csv(train_path)
                self.val_df = pd.read_csv(val_path)
                self.test_df = pd.read_csv(test_path)
                
                print(f"  Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}")
                return self.train_df, self.val_df, self.test_df
        
        print("  Using ORIGINAL dataset (110 images)")
        train_path = os.path.join(self.base_dir, "split_train.csv")
        val_path = os.path.join(self.base_dir, "split_val.csv")
        test_path = os.path.join(self.base_dir, "split_test.csv")
        
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        
        print("  Updating image paths...")
        self._fix_image_paths(self.train_df)
        self._fix_image_paths(self.val_df)
        self._fix_image_paths(self.test_df)
        
        print(f"  Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}")
        
        return self.train_df, self.val_df, self.test_df
    
    def _fix_image_paths(self, df):
        """Update image paths to match current directory structure"""
        def update_path(row):
            old_path = row['jpg_path']
            if pd.isna(old_path): return old_path
            
            filename = os.path.basename(old_path)
            
            uid = row.get('SeriesInstanceUID', '')
            if not uid and 'jpeg' in str(old_path):
                parts = str(old_path).replace('\\', '/').split('/')
                try:
                    idx = parts.index('jpeg')
                    uid = parts[idx+1]
                except (ValueError, IndexError):
                    pass
            
            if uid:
                new_path = os.path.join(self.base_dir, "jpeg", str(uid), filename)
                return new_path
            return old_path

        if 'jpg_path' in df.columns:
            df['jpg_path'] = df.apply(update_path, axis=1)
    
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
                img.verify()
                img = Image.open(img_path)
                img_array = np.array(img)
                
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
        
        for df, name in [(self.train_df, 'Train'), (self.val_df, 'Validation'), (self.test_df, 'Test')]:
            self.check_missing_values(df, name)
        
        for df, name in [(self.train_df, 'Train'), (self.val_df, 'Validation'), (self.test_df, 'Test')]:
            self.verify_image_paths(df, name)
        
        for df, name in [(self.train_df, 'Train'), (self.val_df, 'Validation'), (self.test_df, 'Test')]:
            self.check_image_quality(df, name)
        
        self.train_df = self.remove_duplicates(self.train_df, 'Train')
        self.val_df = self.remove_duplicates(self.val_df, 'Validation')
        self.test_df = self.remove_duplicates(self.test_df, 'Test')
        
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


class EnhancedDataPreprocessor:
    """Enhanced preprocessing to create dataset with 5000+ images instead of 110
    Includes both MASS and CALCIFICATION cases"""
    
    @staticmethod
    def extract_series_uid(path):
        """Extract SeriesInstanceUID from annotation file path
        
        Example path: 'Mass-Training_P_00001_LEFT_CC/[StudyUID]/[SeriesUID]/000000.dcm'
        Path structure: [FolderName]/[StudyInstanceUID]/[SeriesInstanceUID]/[filename]
        Returns: SeriesInstanceUID (the third part, index 2)
        """
        if pd.isna(path) or not isinstance(path, str):
            return None
        parts = path.split('/')
        if len(parts) >= 3:
            return parts[2]  # SeriesInstanceUID is the THIRD part (index 2)
        return None
    
    @staticmethod
    def load_and_match():
        """Load all CBIS-DDSM data and match images with labels"""
        print("\nLoading CBIS-DDSM data...")
        
        ann_files = {
            "mass_train": "mass_case_description_train_set.csv",
            "mass_test": "mass_case_description_test_set.csv",
            "calc_train": "calc_case_description_train_set.csv",
            "calc_test": "calc_case_description_test_set.csv",
        }
        
        dfs = []
        for name, fname in ann_files.items():
            df = pd.read_csv(os.path.join(CSV_DIR_ENHANCED, fname))
            dfs.append(df)
            print(f"  {name}: {len(df)} annotations")
        
        ann = pd.concat(dfs, ignore_index=True)
        
        ann = ann.rename(columns={
            "image file path": "image_file_path",
            "cropped image file path": "cropped_image_file_path",
        })
        
        patho_map = {"MALIGNANT":1, "BENIGN":0, "BENIGN_WITHOUT_CALLBACK":0}
        ann["label"] = ann["pathology"].astype(str).str.upper().map(patho_map)
        
        # Extract SeriesInstanceUID from annotation paths (like notebook approach)
        ann["full_series_uid"] = ann["image_file_path"].apply(EnhancedDataPreprocessor.extract_series_uid)
        ann["crop_series_uid"] = ann["cropped_image_file_path"].apply(EnhancedDataPreprocessor.extract_series_uid)
        
        print(f"\nTotal annotations: {len(ann)}")
        
        # Debug: Show sample extracted UIDs
        sample_uids = ann["full_series_uid"].dropna().head(3).tolist()
        print(f"Sample extracted UIDs: {sample_uids[:1] if sample_uids else 'None'}")
        
        print("Scanning JPEG files...")
        rows = []
        for sid in os.listdir(JPEG_DIR_ENHANCED):
            sdir = os.path.join(JPEG_DIR_ENHANCED, sid)
            if os.path.isdir(sdir):
                for jp in glob.glob(os.path.join(sdir, "*.jpg")):
                    rows.append({"SeriesInstanceUID": sid, "jpg_path": jp})
        
        jpeg_df = pd.DataFrame(rows)
        print(f"Found {len(jpeg_df)} JPEG files")
        
        meta = pd.read_csv(os.path.join(CSV_DIR_ENHANCED, "meta.csv"))
        dinfo = pd.read_csv(os.path.join(CSV_DIR_ENHANCED, "dicom_info.csv"))
        
        meta["SeriesInstanceUID"] = meta["SeriesInstanceUID"].astype(str)
        jpeg_df = jpeg_df.merge(meta[["SeriesInstanceUID","SeriesDescription"]], on="SeriesInstanceUID", how="left")
        
        # Keep PatientID for reference but don't use for matching
        if "PatientID" in dinfo.columns:
            dinfo_clean = dinfo[["SeriesInstanceUID", "PatientID"]].drop_duplicates()
            jpeg_df = jpeg_df.merge(dinfo_clean, on="SeriesInstanceUID", how="left")
        
        print("Image types:")
        for t, c in jpeg_df["SeriesDescription"].value_counts().items():
            print(f"  {t}: {c}")
        
        print("Matching images with labels by SeriesInstanceUID...")
        
        # Filter for full mammogram images
        full_imgs = jpeg_df[jpeg_df["SeriesDescription"]=="full mammogram images"].copy()
        print(f"  Full mammogram images available: {len(full_imgs)}")
        
        # Prepare annotation data with SeriesInstanceUID and labels
        ann_full = ann[["full_series_uid", "label", "patient_id"]].dropna(subset=["full_series_uid"]).drop_duplicates(subset=["full_series_uid"])
        print(f"  Annotations with valid UIDs: {len(ann_full)}")
        
        # Debug: Check if any UIDs match
        jpeg_uids = set(full_imgs["SeriesInstanceUID"].unique())
        ann_uids = set(ann_full["full_series_uid"].unique())
        common_uids = jpeg_uids.intersection(ann_uids)
        print(f"  Common UIDs found: {len(common_uids)}")
        
        if len(common_uids) == 0:
            print("\n⚠ WARNING: No matching UIDs found!")
            print(f"  Sample JPEG UID: {list(jpeg_uids)[:1]}")
            print(f"  Sample Ann UID:  {list(ann_uids)[:1]}")
        
        # Match by SeriesInstanceUID (like the notebook)
        full_matched = full_imgs.merge(ann_full, left_on="SeriesInstanceUID", right_on="full_series_uid", how="inner")
        print(f"  Full mammograms matched: {len(full_matched)}")
        
        combined = full_matched.drop_duplicates(subset=['jpg_path'])
        
        print(f"\nTotal images: {len(combined)} (Benign: {(combined['label']==0).sum()}, Malignant: {(combined['label']==1).sum()})")
        
        return combined
    
    @staticmethod
    def create_splits(df):
        """Create patient-level train/val/test splits"""
        print("\nCreating splits...")
        
        df["patient"] = df["PatientID"].str.extract(r'(P_\d{5})')[0].fillna(df["PatientID"])
        
        patients = df.groupby("patient")["label"].max().reset_index()
        
        train_ids, tmp_ids = train_test_split(
            patients["patient"], test_size=0.30, random_state=42,
            stratify=patients["label"]
        )
        val_ids, test_ids = train_test_split(tmp_ids, test_size=0.5, random_state=42)
        
        train_df = df[df["patient"].isin(set(train_ids))]
        val_df = df[df["patient"].isin(set(val_ids))]
        test_df = df[df["patient"].isin(set(test_ids))]
        
        print(f"Train: {len(train_df)} images")
        print(f"Val: {len(val_df)} images")
        print(f"Test: {len(test_df)} images")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def run_enhanced_preprocessing():
        """Run enhanced preprocessing to create 3000+ image dataset"""
        print("\nCBIS-DDSM Enhanced Preprocessing")
        
        if not os.path.exists(CSV_DIR_ENHANCED) or not os.path.exists(JPEG_DIR_ENHANCED):
            print(f"\nError: Enhanced dataset directory not found at {BASE_ENHANCED}")
            print("Please ensure the full CBIS-DDSM dataset is available.")
            return False
        
        combined = EnhancedDataPreprocessor.load_and_match()
        
        train, val, test = EnhancedDataPreprocessor.create_splits(combined)
        
        train.to_csv(os.path.join(OUTPUT_DIR_ENHANCED, "train_enhanced.csv"), index=False)
        val.to_csv(os.path.join(OUTPUT_DIR_ENHANCED, "val_enhanced.csv"), index=False)
        test.to_csv(os.path.join(OUTPUT_DIR_ENHANCED, "test_enhanced.csv"), index=False)
        
        print(f"\nDataset created: {len(combined)} images (increased from 110)")
        print(f"Saved to: {OUTPUT_DIR_ENHANCED}")
        
        return True


def apply_clahe_and_negative(image_np, target_size=(224, 224)):
    """
    Apply CLAHE and Negative Transformation (from notebooks)
    Optimized for mammogram mass detection
    
    Args:
        image_np: numpy array (H, W, 3) in RGB format
        target_size: tuple (height, width) for resizing
    
    Returns:
        PIL Image with CLAHE and negative transformation applied
    """
    image = cv2.resize(image_np, (target_size[1], target_size[0]))
    
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    image_lab = cv2.merge((l_channel, a_channel, b_channel))
    
    image = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
    
    image = 255 - image
    
    return Image.fromarray(image.astype(np.uint8))


class CBISDDSMDataset(Dataset):
    """Custom PyTorch Dataset for CBIS-DDSM images with CLAHE + Negative preprocessing"""
    
    def __init__(self, csv_path, transform=None, image_size=224, use_clahe=True):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_size = image_size
        self.use_clahe = use_clahe
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = row['jpg_path']
        
        if self.use_clahe:
            image_np = np.array(Image.open(img_path).convert('RGB'))
            image = apply_clahe_and_negative(image_np, target_size=(self.image_size, self.image_size))
        else:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return image, label


def get_data_transforms(image_size=224, use_clahe=True):
    """
    Get data transforms for training and validation.
    Note: Resizing is handled in apply_clahe_and_negative() when use_clahe=True
    """
    if use_clahe:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
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
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return train_transform, val_transform


def create_data_loaders(output_dir, batch_size=32, image_size=224, num_workers=4, use_clahe=True):
    """
    Create PyTorch DataLoaders with optional CLAHE + Negative preprocessing.
    
    Args:
        output_dir: Directory containing preprocessed CSV files
        batch_size: Batch size for DataLoader
        image_size: Target image size (height, width)
        num_workers: Number of workers for data loading
        use_clahe: If True, applies CLAHE + Negative transformation (like notebooks)
    """
    print("\nCreating data loaders...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    enhanced_dir = os.path.join(script_dir, "preprocessed_data_enhanced")
    
    if os.path.exists(os.path.join(enhanced_dir, "train_enhanced.csv")):
        print("  Using ENHANCED dataset (full mammograms, MASS + CALC CASES)")
        if use_clahe:
            print("  Applying CLAHE + Negative Transformation")
        train_csv = os.path.join(enhanced_dir, "train_enhanced.csv")
        val_csv = os.path.join(enhanced_dir, "val_enhanced.csv")
        test_csv = os.path.join(enhanced_dir, "test_enhanced.csv")
    else:
        print("  Using standard dataset")
        train_csv = os.path.join(output_dir, "train_cleaned.csv")
        val_csv = os.path.join(output_dir, "val_cleaned.csv")
        test_csv = os.path.join(output_dir, "test_cleaned.csv")
    
    train_transform, val_transform = get_data_transforms(image_size, use_clahe=use_clahe)
    
    train_dataset = CBISDDSMDataset(
        csv_path=train_csv,
        transform=train_transform,
        image_size=image_size,
        use_clahe=use_clahe
    )
    
    val_dataset = CBISDDSMDataset(
        csv_path=val_csv,
        transform=val_transform,
        image_size=image_size,
        use_clahe=use_clahe
    )
    
    test_dataset = CBISDDSMDataset(
        csv_path=test_csv,
        transform=val_transform,
        image_size=image_size,
        use_clahe=use_clahe
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
    print("Running enhanced preprocessing (MASS + CALC cases)...")
    
    success = EnhancedDataPreprocessor.run_enhanced_preprocessing()
    
    if not success:
        print("\nEnhanced preprocessing failed.")
        print("Please ensure the full CBIS-DDSM dataset is available.")
        return
    
    print("\nPreprocessing complete!")
    print(f"Output: {OUTPUT_DIR_ENHANCED}")
    print("Files: train_enhanced.csv, val_enhanced.csv, test_enhanced.csv")


if __name__ == "__main__":
    main()
