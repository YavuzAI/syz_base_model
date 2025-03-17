# data_loader.py

import os
import pandas as pd
import numpy as np
import pydicom
from sklearn.model_selection import train_test_split, StratifiedKFold
# Import your custom metadata processing function; ensure it is defined in process_wmetadata_image.py


def read_image_df(data_dir):
    """
    Recursively scan the data_dir and create a DataFrame that stores the image file paths
    and the label, where the label is the folder name.
    """
    data = []
    for root, dirs, files in os.walk(data_dir):
        # The folder name (last part of root) is used as the label
        label = os.path.basename(root)
        # Skip folders that are not labels (optional: you can add a check if labels are only '0' and '1')
        if label not in ['0', '1']:
            continue
        for file in files:
            if file.lower().endswith('.dcm'):
                file_path = os.path.join(root, file)
                data.append({
                    "file_path": file_path,
                    "label": int(label)  # convert label to integer if needed
                })
    return pd.DataFrame(data)

def process_window_value(window_value):
    if isinstance(window_value, pydicom.multival.MultiValue):
        return list(window_value) 
    else:
        return [float(window_value)] 

def read_metadata_df(data_dir):
    """
    Recursively scan the data_dir and create a DataFrame that stores the file path and extracted metadata.
    """
    metadata_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path)
                    # Extract all the relevant metadata fields
                    metadata = {
                        "file_path": file_path,
                        "SliceThickness": float(ds.SliceThickness),
                        "SamplesPerPixel": float(ds.SamplesPerPixel),
                        "Photometric Interpretation": ds.PhotometricInterpretation,
                        "PixelSpacing": process_window_value(ds.PixelSpacing),
                        "BitsAllocated": float(ds.BitsAllocated),
                        "BitsStored": float(ds.BitsStored),
                        "HighBit": float(ds.HighBit),
                        "PixelRepresentation": float(ds.PixelRepresentation),
                        "WindowCenter":  process_window_value(ds.WindowCenter),
                        "WindowWidth": process_window_value(ds.WindowWidth),
                        "RescaleIntercept": float(ds.RescaleIntercept),
                        "RescaleSlope": float(ds.RescaleSlope),
                        "RescaleType": ds.RescaleType
                    }
                    metadata_list.append(metadata)
                except Exception as e:
                    print(f"Error reading metadata from {file_path}: {e}")
    return pd.DataFrame(metadata_list)


def split_dataset(df):
    """
    Split DataFrame into training (70%), validation (20%), and test (10%) sets.
    Stratification is applied based on the 'label' column.
    """
    # First, extract the test set (10%)
    train_val, test = train_test_split(df, test_size=0.10, stratify=df['label'], random_state=42)
    # Then split train_val into train (70%) and validation (20%)
    # Since train_val is 90%, a 22.22% split of it yields approximately 20% of the full dataset.
    train, val = train_test_split(train_val, test_size=0.2222, stratify=train_val['label'], random_state=42)
    return train, val, test

def get_stratified_kfold(df, n_splits=5):
    """
    Return stratified k-fold indices for cross validation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = df.index.values
    y = df['label'].values
    return list(skf.split(X, y))
