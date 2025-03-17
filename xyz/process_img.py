import numpy as np
import pandas as pd
import pydicom
import cv2 

def apply_window(hu_image, center, width):
    min_value = center - width // 2
    max_value = center + width // 2
    windowed = np.clip(hu_image, min_value, max_value)
    # Normalize to 0-1 range
    normalized = (windowed - min_value) / width
    return normalized

def preprocess_with_metadata_from_df(dicom_path, metadata_df, target_shape=(512, 512)):
    """
    Reads a DICOM file, applies windowing transformations, and resizes to a fixed shape.
    
    Args:
        dicom_path (str): Path to the DICOM file.
        metadata_df (pd.DataFrame): DataFrame containing metadata for the images.
        target_shape (tuple): Desired (height, width) for the output image.

    Returns:
        np.ndarray: Processed image with shape (512, 512, 4).
    """
    # Find the metadata row for this specific file
    metadata_row = metadata_df[metadata_df['file_path'] == dicom_path]

    if metadata_row.empty:
        raise ValueError(f"No metadata found for {dicom_path} in the DataFrame")

    # Extract metadata values from the DataFrame
    rescale_slope = metadata_row['RescaleSlope'].values[0]
    rescale_intercept = metadata_row['RescaleIntercept'].values[0]
    slice_thickness = metadata_row['SliceThickness'].values[0]
    pixel_spacing = metadata_row['PixelSpacing'].values[0][0]  # Extract first value if it's a list

    # Load DICOM and get pixel array
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array

    # Convert to Hounsfield Units using rescale values
    hu_image = pixel_array * float(rescale_slope) + float(rescale_intercept)

    # Apply windowing for different medical image views
    brain_window = apply_window(hu_image, center=40, width=80)
    subdural_window = apply_window(hu_image, center=80, width=200)
    stroke_window = apply_window(hu_image, center=50, width=50)

    # Create a resolution map
    resolution_map = np.ones_like(brain_window) * (slice_thickness / pixel_spacing)

    # Resize each channel to (512, 512) before stacking
    brain_window_resized = cv2.resize(brain_window, target_shape, interpolation=cv2.INTER_AREA)
    subdural_window_resized = cv2.resize(subdural_window, target_shape, interpolation=cv2.INTER_AREA)
    stroke_window_resized = cv2.resize(stroke_window, target_shape, interpolation=cv2.INTER_AREA)
    resolution_map_resized = cv2.resize(resolution_map, target_shape, interpolation=cv2.INTER_AREA)

    # Stack all channels together to maintain (512, 512, 4) shape
    multichannel_input = np.stack([
        brain_window_resized,
        subdural_window_resized,
        stroke_window_resized,
        resolution_map_resized
    ], axis=-1)

    return multichannel_input