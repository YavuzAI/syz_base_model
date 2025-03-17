## This file is an outline of the overall pipeline. Each module is self‐contained and together they create the following workflow:

### Data Loading & Preprocessing:
    • Recursively scan a given folder for DICOM files.
    • For each file, use the pydicom library to read the image and extract all metadata fields (e.g., SOP Class UID, Slice Thickness, Pixel Spacing, etc.).
    • Process each image with your custom metadata-based function (preprocess_with_metadata).
    • Store the processed image, label, and metadata in a Pandas DataFrame.
    • Finally, split the DataFrame into training (70%), validation (20%), and test (10%) sets.

### Model Architecture:
    • Create a placeholder CNN model that accepts images with the required input shape.
    • Define the final output layer to support two classes (0 and 1) with an appropriate loss function (here, we use sparse categorical cross-entropy) and an optimizer.

### Training Functions:
• Implement a training function that uses TensorFlow’s ImageDataGenerator to perform data augmentation only on the training data.
• Train the model using the training generator and validate on non‐augmented validation data.

### Evaluation & Cross Validation:
• Define functions to compute accuracy, precision, recall, F1-score, ROC, AUC, and generate a confusion matrix.
• Provide an example of how to use StratifiedKFold for cross validation (handled separately for now).

### Visualization Utilities:
• Create functions to plot the confusion matrix and training curves (accuracy and loss over epochs).

### Main Notebook Integration:
• Use a Jupyter notebook (main.ipynb) to import these modules, create the DataFrame, split the data, build the model, run training, and visualize results.




