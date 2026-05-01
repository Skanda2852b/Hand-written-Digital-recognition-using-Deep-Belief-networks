"""
Evaluation metrics: accuracy, classification report, confusion matrix.
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate(classifier, X, y, dataset_name="dataset"):
    """
    Predict and compute accuracy on the given dataset.

    Parameters
    ----------
    classifier : SupervisedDBNClassification
        Trained DBN model.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True labels.
    dataset_name : str
        Human-readable name for logging.

    Returns
    -------
    y_pred : np.ndarray
        Predicted labels.
    accuracy : float
        Classification accuracy (0–1).
    """
    y_pred = np.array(classifier.predict(X))
    accuracy = accuracy_score(y, y_pred)
    print(f"[eval] {dataset_name:10s} accuracy = {accuracy * 100:.2f} %")
    return y_pred, accuracy


def full_report(y_true, y_pred, digits=4):
    """
    Print a detailed sklearn classification report.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    digits : int
        Number of decimal places.

    Returns
    -------
    str
        The classification report string.
    """
    report = classification_report(y_true, y_pred, digits=digits)
    print("[report] Classification report (test set):")
    print("-" * 60)
    print(report)
    return report


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.

    Parameters
    ----------
    y_true, y_pred : np.ndarray

    Returns
    -------
    np.ndarray
        10×10 confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    print("[cm] Confusion matrix computed.")
    return cm
