import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, test_images, test_labels):
    """
    Evaluate the model on the test data and compute metrics:
    accuracy, precision, recall, F1-score, ROC, AUC, and the confusion matrix.
    """
    preds = model.predict(test_images)
    preds_labels = preds.argmax(axis=1)
    accuracy = accuracy_score(test_labels, preds_labels)
    precision = precision_score(test_labels, preds_labels, average='weighted')
    recall = recall_score(test_labels, preds_labels, average='weighted')
    f1 = f1_score(test_labels, preds_labels, average='weighted')
    # For ROC and AUC, assuming binary classification. If using more classes, adjust accordingly.
    roc_auc = roc_auc_score(test_labels, preds[:, 1]) if preds.shape[1] == 2 else None
    cm = confusion_matrix(test_labels, preds_labels)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }

def cross_validate_model(model_func, df, kfold_splits, batch_size=8, epochs=10):
    """
    Perform cross validation using the provided model function and stratified k-fold indices.
    Returns a list of accuracy scores for each fold.
    """
    metrics_list = []
    for fold, (train_idx, val_idx) in enumerate(kfold_splits):
        print(f"Training fold {fold + 1}")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        # Extract images and labels from the DataFrame
        import numpy as np
        train_images = np.stack(train_df['image'].values)
        train_labels = train_df['label'].values
        val_images = np.stack(val_df['image'].values)
        val_labels = val_df['label'].values
        
        model = model_func()
        history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                            validation_data=(val_images, val_labels), verbose=0)
        preds = model.predict(val_images)
        preds_labels = preds.argmax(axis=1)
        acc = np.mean(preds_labels == val_labels)
        metrics_list.append(acc)
    return metrics_list
