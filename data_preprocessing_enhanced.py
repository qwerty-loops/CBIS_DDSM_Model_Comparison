"""Enhanced CBIS-DDSM Preprocessing - Increases dataset from 110 to 3000+ images"""

import os, re, glob, pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = os.path.join(PARENT_DIR, "CBIS_DDSM")
CSV_DIR = os.path.join(BASE, "csv")
JPEG_DIR = os.path.join(BASE, "jpeg")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "preprocessed_data_enhanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def norm_key(s):
    """Normalize for matching"""
    s = str(s).strip().replace("–","-").replace("—","-")
    return re.sub(r'\s+','', re.sub(r'[_-]+', lambda m: m.group()[0], s)).upper()

def load_and_match():
    print("\n" + "="*70)
    print("LOADING CBIS-DDSM DATA")
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
        df = pd.read_csv(os.path.join(CSV_DIR, fname))
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
    ann["full_key"] = ann["image_file_path"].apply(lambda s: str(s).split("/")[0] if "/" in str(s) else "").apply(norm_key)
    ann["crop_key"] = ann["cropped_image_file_path"].apply(lambda s: str(s).split("/")[0] if "/" in str(s) else "").apply(norm_key)
    
    print(f"\nTotal annotations: {len(ann)}")
    
    # Scan JPEG directory
    print("\nScanning JPEG files...")
    rows = []
    for sid in os.listdir(JPEG_DIR):
        sdir = os.path.join(JPEG_DIR, sid)
        if os.path.isdir(sdir):
            for jp in glob.glob(os.path.join(sdir, "*.jpg")):
                rows.append({"SeriesInstanceUID": sid, "jpg_path": jp})
    
    jpeg_df = pd.DataFrame(rows)
    print(f"Found {len(jpeg_df)} JPEG files")
    
    # Load metadata
    meta = pd.read_csv(os.path.join(CSV_DIR, "meta.csv"))
    dinfo = pd.read_csv(os.path.join(CSV_DIR, "dicom_info.csv"))
    
    # Merge metadata
    meta["SeriesInstanceUID"] = meta["SeriesInstanceUID"].astype(str)
    jpeg_df = jpeg_df.merge(meta[["SeriesInstanceUID","SeriesDescription"]], on="SeriesInstanceUID", how="left")
    
    if "PatientID" in dinfo.columns:
        dinfo_clean = dinfo[["SeriesInstanceUID", "PatientID"]].drop_duplicates()
        jpeg_df = jpeg_df.merge(dinfo_clean, on="SeriesInstanceUID", how="left")
    
    jpeg_df["pid_norm"] = jpeg_df["PatientID"].astype(str).apply(norm_key)
    
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

def create_splits(df):
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

def main():
    print("\n" + "#"*70)
    print("CBIS-DDSM ENHANCED PREPROCESSING")
    print("#"*70)
    
    # Load and match
    combined = load_and_match()
    
    # Create splits
    train, val, test = create_splits(combined)
    
    # Save
    train.to_csv(os.path.join(OUTPUT_DIR, "train_enhanced.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, "val_enhanced.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test_enhanced.csv"), index=False)
    
    print(f"\n✓ Saved to: {OUTPUT_DIR}")
    print(f"\nDATASET INCREASED FROM 110 TO {len(combined)} IMAGES!")
    print("#"*70 + "\n")

if __name__ == "__main__":
    main()