"""
CBIS-DDSM Data Preprocessing

Loads, cleans, and prepares the CBIS-DDSM mammography dataset.
Creates cleaned CSV files and PyTorch DataLoaders for model training.

Features:
- Standard preprocessing: Uses existing split files (110 images)
- Enhanced preprocessing: Processes full CBIS-DDSM dataset (3000+ images)
  by matching all JPEG images with annotations from CSV files
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
import warnings
warnings.filterwarnings('ignore')

# Dataset paths - Using relative paths for portability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "Images")
CSV_DIR = os.path.join(BASE, "csv")
JPEG_DIR = os.path.join(BASE, "jpeg")
PNG_DIR = os.path.join(BASE, "png")

# Output directory for preprocessed data
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "preprocessed_data")
OUTPUT_DIR_ENHANCED = os.path.join(SCRIPT_DIR, "preprocessed_data_enhanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_ENHANCED, exist_ok=True)

# Enhanced dataset paths (for 3000+ images)
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
                print("  ✓ Using ENHANCED dataset (3000+ images)")
                self.train_df = pd.read_csv(train_path)
                self.val_df = pd.read_csv(val_path)
                self.test_df = pd.read_csv(test_path)
                
                print(f"  Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}")
                return self.train_df, self.val_df, self.test_df
        
        # Fallback to original dataset
        print("  ⚠ Using ORIGINAL dataset (110 images)")
        train_path = os.path.join(self.base_dir, "split_train.csv")
        val_path = os.path.join(self.base_dir, "split_val.csv")
        test_path = os.path.join(self.base_dir, "split_test.csv")
        
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        
        # Fix paths to match current location
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
            
            # Get filename from old path
            filename = os.path.basename(old_path)
            
            # Get SeriesInstanceUID if available to construct path
            uid = row.get('SeriesInstanceUID', '')
            if not uid and 'jpeg' in str(old_path):
                # Try to extract UID from path if not in column
                # Path structure: .../jpeg/UID/filename
                parts = str(old_path).replace('\\', '/').split('/')
                try:
                    idx = parts.index('jpeg')
                    uid = parts[idx+1]
                except (ValueError, IndexError):
                    pass
            
            if uid:
                # Construct new path: BASE/jpeg/UID/filename
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


class EnhancedDataPreprocessor:
    """Enhanced preprocessing to create dataset with 3000+ images instead of 110"""
    
    @staticmethod
    def norm_key(s):
        """Normalize for matching"""
        s = str(s).strip().replace("–","-").replace("—","-")
        return re.sub(r'\s+','', re.sub(r'[_-]+', lambda m: m.group()[0], s)).upper()
    
    @staticmethod
    def load_and_match():
        """Load all CBIS-DDSM data and match images with labels"""
        print("\n" + "="*70)
        print("LOADING CBIS-DDSM DATA (ENHANCED MODE)")
        print("="*70)
        
        # Load annotations
        ann_files = {
            "calc_train": "calc_case_description_train_set.csv",
            "calc_test": "calc_case_description_test_set.csv",
            "mass_train": "mass_case_description_train_set.csv",
            "mass_test": "mass_case_description_test_set.csv",
        }
        
        dfs = []
        for name, fname in ann_files.items():
            df = pd.read_csv(os.path.join(CSV_DIR_ENHANCED, fname))
            dfs.append(df)
            print(f"  ✓ {name}: {len(df)} annotations")
        
        ann = pd.concat(dfs, ignore_index=True)
        
        # Rename columns
        ann = ann.rename(columns={
            "image file path": "image_file_path",
            "cropped image file path": "cropped_image_file_path",
        })
        
        # Create labels
        patho_map = {"MALIGNANT":1, "BENIGN":0, "BENIGN_WITHOUT_CALLBACK":0}
        ann["label"] = ann["pathology"].astype(str).str.upper().map(patho_map)
        
        # Extract series keys
        ann["full_key"] = ann["image_file_path"].apply(lambda s: str(s).split("/")[0] if "/" in str(s) else "").apply(EnhancedDataPreprocessor.norm_key)
        ann["crop_key"] = ann["cropped_image_file_path"].apply(lambda s: str(s).split("/")[0] if "/" in str(s) else "").apply(EnhancedDataPreprocessor.norm_key)
        
        print(f"\nTotal annotations: {len(ann)}")
        
        # Scan JPEG directory
        print("\nScanning JPEG files...")
        rows = []
        for sid in os.listdir(JPEG_DIR_ENHANCED):
            sdir = os.path.join(JPEG_DIR_ENHANCED, sid)
            if os.path.isdir(sdir):
                for jp in glob.glob(os.path.join(sdir, "*.jpg")):
                    rows.append({"SeriesInstanceUID": sid, "jpg_path": jp})
        
        jpeg_df = pd.DataFrame(rows)
        print(f"Found {len(jpeg_df)} JPEG files")
        
        # Load metadata
        meta = pd.read_csv(os.path.join(CSV_DIR_ENHANCED, "meta.csv"))
        dinfo = pd.read_csv(os.path.join(CSV_DIR_ENHANCED, "dicom_info.csv"))
        
        # Merge metadata
        meta["SeriesInstanceUID"] = meta["SeriesInstanceUID"].astype(str)
        jpeg_df = jpeg_df.merge(meta[["SeriesInstanceUID","SeriesDescription"]], on="SeriesInstanceUID", how="left")
        
        if "PatientID" in dinfo.columns:
            dinfo_clean = dinfo[["SeriesInstanceUID", "PatientID"]].drop_duplicates()
            jpeg_df = jpeg_df.merge(dinfo_clean, on="SeriesInstanceUID", how="left")
        
        jpeg_df["pid_norm"] = jpeg_df["PatientID"].astype(str).apply(EnhancedDataPreprocessor.norm_key)
        
        print("\nImage types:")
        for t, c in jpeg_df["SeriesDescription"].value_counts().items():
            print(f"  {t}: {c}")
        
        # Match images
        print("\nMatching images with labels...")
        
        # Full mammograms
        full_imgs = jpeg_df[jpeg_df["SeriesDescription"]=="full mammogram images"].copy()
        ann_full = ann[["full_key", "label"]].dropna().drop_duplicates()
        full_matched = full_imgs.merge(ann_full, left_on="pid_norm", right_on="full_key", how="inner")
        print(f"  ✓ Full mammograms: {len(full_matched)}")
        
        # Cropped images
        crop_imgs = jpeg_df[jpeg_df["SeriesDescription"]=="cropped images"].copy()
        ann_crop = ann[["crop_key", "label"]].dropna().drop_duplicates()
        crop_matched = crop_imgs.merge(ann_crop, left_on="pid_norm", right_on="crop_key", how="inner")
        print(f"  ✓ Cropped images: {len(crop_matched)}")
        
        # Combine
        combined = pd.concat([full_matched, crop_matched], ignore_index=True)
        combined = combined.drop_duplicates(subset=['jpg_path'])
        
        print("\n" + "="*70)
        print(f"COMBINED: {len(combined)} IMAGES")
        print("="*70)
        print(f"Benign: {(combined['label']==0).sum()}, Malignant: {(combined['label']==1).sum()}")
        
        return combined
    
    @staticmethod
    def create_splits(df):
        """Create patient-level train/val/test splits"""
        print("\nCreating patient-level splits...")
        
        # Extract patient ID
        df["patient"] = df["PatientID"].str.extract(r'(P_\d{5})')[0].fillna(df["PatientID"])
        
        # Patient-level split
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
        print("\n" + "#"*70)
        print("CBIS-DDSM ENHANCED PREPROCESSING")
        print("#"*70)
        
        # Check if enhanced data directory exists
        if not os.path.exists(CSV_DIR_ENHANCED) or not os.path.exists(JPEG_DIR_ENHANCED):
            print(f"\nError: Enhanced dataset directory not found at {BASE_ENHANCED}")
            print("Please ensure the full CBIS-DDSM dataset is available.")
            return False
        
        # Load and match
        combined = EnhancedDataPreprocessor.load_and_match()
        
        # Create splits
        train, val, test = EnhancedDataPreprocessor.create_splits(combined)
        
        # Save
        train.to_csv(os.path.join(OUTPUT_DIR_ENHANCED, "train_enhanced.csv"), index=False)
        val.to_csv(os.path.join(OUTPUT_DIR_ENHANCED, "val_enhanced.csv"), index=False)
        test.to_csv(os.path.join(OUTPUT_DIR_ENHANCED, "test_enhanced.csv"), index=False)
        
        print(f"\n✓ Saved to: {OUTPUT_DIR_ENHANCED}")
        print(f"\nDATASET INCREASED FROM 110 TO {len(combined)} IMAGES!")
        print("#"*70 + "\n")
        
        return True


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
    
    # Run enhanced preprocessing by default (3000+ images)
    print("\nRunning Enhanced Preprocessing (3000+ images)...")
    print("="*70)
    
    success = EnhancedDataPreprocessor.run_enhanced_preprocessing()
    
    if not success:
        print("\nEnhanced preprocessing failed.")
        print("Please ensure the full CBIS-DDSM dataset is available.")
        return
    
    print("\n" + "="*70)
    print("Preprocessing complete!")
    print("="*70)
    print(f"Output: {OUTPUT_DIR_ENHANCED}")
    print(f"\nDataset files created:")
    print(f"  - train_enhanced.csv")
    print(f"  - val_enhanced.csv")
    print(f"  - test_enhanced.csv")


if __name__ == "__main__":
    main()
