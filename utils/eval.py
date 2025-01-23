import numpy as np
from scipy.stats import sem, t
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

###### Eval ######
def evaluate_model(y_true, y_pred_proba, model_name="Model"):
    """
    Evaluate the model's performance using various metrics: AUC, AUPR, Average F1.

    Args:
    - y_true (np.ndarray): Ground truth labels (binary: 0 or 1).
    - y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
    - model_name (str): Optional name for the model, used in print statements.

    Returns:
    - auc (float): Area Under the ROC Curve (AUC).
    - aupr (float): Area Under the Precision-Recall Curve (AUPR).
    - average_F1 (float): Average F1 score calculated over all thresholds.
    """
    # Compute AUC (Area Under the ROC Curve) and AUPR (Area Under the Precision-Recall Curve)
    print(f"Calculating AUC and AUPR for {model_name}...")
    auc = roc_auc_score(y_true, y_pred_proba)
    aupr = average_precision_score(y_true, y_pred_proba)

    # Compute the precision-recall curve and F1 score for each threshold
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 score for each point on the precision-recall curve
    print("Calculating Average F1 score over thresholds...")
    numerator = 2 * recall_curve * precision_curve
    denom = recall_curve + precision_curve
    F1_curve = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))  # Avoid division by zero
    average_F1 = np.mean(F1_curve)  # Average F1 score over all thresholds

    # Print the evaluation results
    print(f'{model_name} - AUC: {auc:.4f}')
    print(f'{model_name} - AUPR: {aupr:.4f}')
    print(f'{model_name} - Average F1: {average_F1:.4f}')

    return auc, aupr, average_F1

def calculate_ci(data, confidence=0.95):
    """
    Calculate the confidence interval for a given dataset.
    
    Parameters:
        data (list or np.array): The data points.
        confidence (float): The confidence level (default is 0.95).

    Returns:
        tuple: Mean and confidence interval range (lower_bound, upper_bound).
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std_err = sem(data)
    margin_of_error = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin_of_error, mean + margin_of_error
