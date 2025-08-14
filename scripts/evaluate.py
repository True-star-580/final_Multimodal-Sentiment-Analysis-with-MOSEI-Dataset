import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)

def calc_mae(preds, labels):
    """
    Calculate Mean Absolute Error (MAE) between predictions and labels.
    
    Args:
        preds (np.ndarray): Predicted values.
        labels (np.ndarray): Ground truth values.
    
    Returns:
        float: Mean absolute error.
    """
    # Check for NaN values and handle them
    if np.any(np.isnan(preds)) or np.any(np.isnan(labels)):
        logger.warning("NaN values detected in predictions or labels for MAE calculation")
        # Remove NaN values
        mask = ~(np.isnan(preds) | np.isnan(labels))
        if np.sum(mask) == 0:
            logger.error("All values are NaN, returning NaN for MAE")
            return float('nan')
        preds = preds[mask]
        labels = labels[mask]
    
    return mean_absolute_error(labels, preds)

def calc_correlation(preds, labels):
    """
    Calculate Pearson correlation coefficient between predictions and labels.
    
    Args:
        preds (np.ndarray): Predicted values.
        labels (np.ndarray): Ground truth values.
    
    Returns:
        float: Pearson correlation value.
    """
    # Check for NaN values
    if np.any(np.isnan(preds)) or np.any(np.isnan(labels)):
        logger.warning("NaN values detected in predictions or labels for correlation calculation")
        mask = ~(np.isnan(preds) | np.isnan(labels))
        if np.sum(mask) == 0:
            return float('nan')
        preds = preds[mask]
        labels = labels[mask]
    
    if np.std(preds) == 0 or np.std(labels) == 0:
        return 0.0
    
    return np.corrcoef(preds, labels)[0, 1]

def calc_binary_accuracy(preds, labels, threshold=0):
    """
    Calculate binary classification accuracy based on a threshold.
    
    Args:
        preds (np.ndarray): Predicted values.
        labels (np.ndarray): Ground truth values.
        threshold (float): Threshold to binarize predictions and labels.
    
    Returns:
        float: Binary classification accuracy.
    """
    # Check for NaN values
    if np.any(np.isnan(preds)) or np.any(np.isnan(labels)):
        logger.warning("NaN values detected in predictions or labels for binary accuracy calculation")
        mask = ~(np.isnan(preds) | np.isnan(labels))
        if np.sum(mask) == 0:
            return float('nan')
        preds = preds[mask]
        labels = labels[mask]
    
    binary_preds = (preds > threshold).astype(int)
    binary_labels = (labels > threshold).astype(int)
    
    return accuracy_score(binary_labels, binary_preds)

def calc_f1(preds, labels, threshold=0):
    """
    Calculate binary F1-score based on a threshold.
    
    Args:
        preds (np.ndarray): Predicted values.
        labels (np.ndarray): Ground truth values.
        threshold (float): Threshold to binarize predictions and labels.
    
    Returns:
        float: Binary F1-score.
    """
    # Check for NaN values
    if np.any(np.isnan(preds)) or np.any(np.isnan(labels)):
        logger.warning("NaN values detected in predictions or labels for F1 calculation")
        mask = ~(np.isnan(preds) | np.isnan(labels))
        if np.sum(mask) == 0:
            return float('nan')
        preds = preds[mask]
        labels = labels[mask]
    
    binary_preds = (preds > threshold).astype(int)
    binary_labels = (labels > threshold).astype(int)
    
    return f1_score(binary_labels, binary_preds, zero_division=0)

def calc_multiclass_metrics(preds, labels):
    """
    Calculate multiclass accuracy and weighted F1-score based on 7 sentiment classes (-3 to +3).
    
    Args:
        preds (np.ndarray): Predicted values.
        labels (np.ndarray): Ground truth values.
    
    Returns:
        dict: Dictionary containing 'multiclass_acc' and 'multiclass_f1'.
    """
    # Check for NaN values
    if np.any(np.isnan(preds)) or np.any(np.isnan(labels)):
        logger.warning("NaN values detected in predictions or labels for multiclass metrics")
        mask = ~(np.isnan(preds) | np.isnan(labels))
        if np.sum(mask) == 0:
            return {"multiclass_acc": float('nan'), "multiclass_f1": float('nan')}
        preds = preds[mask]
        labels = labels[mask]
    
    # Round to nearest integer and clip to [-3, 3] range
    rounded_preds = np.round(preds).clip(-3, 3)
    rounded_labels = np.round(labels).clip(-3, 3)
    
    # Convert to 7 classes (0-6 for -3 to +3)
    preds_classes = (rounded_preds + 3).astype(int)
    labels_classes = (rounded_labels + 3).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(labels_classes, preds_classes)
    f1 = f1_score(labels_classes, preds_classes, average="weighted", zero_division=0)
    
    return {
        "multiclass_acc": acc,
        "multiclass_f1": f1
    }

def get_predictions(model, dataloader, device):
    """
    Run inference on the provided dataloader and collect predictions and labels.
    
    Args:
        model (torch.nn.Module): Trained model for inference.
        dataloader (torch.utils.data.DataLoader): Dataloader to evaluate.
        device (str): Device to perform inference on ('cuda', 'cpu', etc.).
    
    Returns:
        tuple: (np.ndarray of predictions, np.ndarray of labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            if isinstance(batch, dict):
                # Multimodal data
                inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
                labels = batch["label"].to(device)
            else:
                # Unimodal data
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Check for NaN in outputs
            if torch.any(torch.isnan(outputs)):
                logger.warning("NaN detected in model outputs during prediction")
                # Replace NaN with zeros or skip this batch
                outputs = torch.nan_to_num(outputs, nan=0.0)
            
            # Collect predictions and labels
            preds = outputs.squeeze().cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    # Concatenate batch results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return all_preds, all_labels

def evaluate_mosei(model, dataloader, device):
    """
    Evaluate model on CMU-MOSEI dataset using all relevant metrics.
    
    Args:
        model (torch.nn.Module): Trained model for evaluation.
        dataloader (torch.utils.data.DataLoader): Evaluation dataloader.
        device (str): Computation device.
    
    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    all_preds, all_labels = get_predictions(model, dataloader, device)
    
    # Check if we have any valid predictions
    if len(all_preds) == 0 or np.all(np.isnan(all_preds)):
        logger.error("All predictions are NaN or empty")
        return {
            "mae": float('nan'),
            "corr": float('nan'),
            "binary_acc": float('nan'),
            "binary_f1": float('nan'),
            "multiclass_acc": float('nan'),
            "multiclass_f1": float('nan')
        }
    
    # Calculate metrics
    mae = calc_mae(all_preds, all_labels)
    corr = calc_correlation(all_preds, all_labels)
    acc = calc_binary_accuracy(all_preds, all_labels)
    f1 = calc_f1(all_preds, all_labels)
    
    # Multi-class metrics
    multiclass_metrics = calc_multiclass_metrics(all_preds, all_labels)
    
    # Combine all metrics
    metrics = {
        "mae": mae,
        "corr": corr,
        "binary_acc": acc,
        "binary_f1": f1,
        **multiclass_metrics
    }
    
    return metrics

def compute_metrics(model, dataloader, device):
    """
    Alias for evaluate_mosei to maintain compatibility with evaluation script.
    
    Args:
        model (torch.nn.Module): Trained model for evaluation.
        dataloader (torch.utils.data.DataLoader): Evaluation dataloader.
        device (str): Computation device.
    
    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    return evaluate_mosei(model, dataloader, device)

def log_metrics(metrics, split, epoch=None):
    """
    Log evaluation metrics to the logger.
    
    Args:
        metrics (dict): Dictionary of evaluation metrics.
        split (str): Dataset split ('train', 'val', 'test').
        epoch (int, optional): Epoch number (for training logs).
    """
    epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
    logger.info(f"{epoch_str}{split.capitalize()} metrics:")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  Correlation: {metrics['corr']:.4f}")
    logger.info(f"  Binary Accuracy: {metrics['binary_acc']:.4f}")
    logger.info(f"  Binary F1: {metrics['binary_f1']:.4f}")
    logger.info(f"  7-class Accuracy: {metrics['multiclass_acc']:.4f}")
    logger.info(f"  7-class F1: {metrics['multiclass_f1']:.4f}")
