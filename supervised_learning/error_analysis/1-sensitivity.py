#!/usr/bin/env python3
"""
Module that calculates sensitivity (recall) for each class
from a confusion matrix.
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Parameters
    ----------
    confusion : numpy.ndarray
        Confusion matrix of shape (classes, classes) where rows represent
        the correct labels and columns represent the predicted labels.

    Returns
    -------
    numpy.ndarray
        Array of shape (classes,) containing the sensitivity of each class.
    """
    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # False negatives are the sum of each row minus the true positives
    false_negatives = np.sum(confusion, axis=1) - true_positives

    # Sensitivity = TP / (TP + FN)
    sensitivity_scores = true_positives / (true_positives + false_negatives)

    return sensitivity_scores
