#!/usr/bin/env python3
"""
Module that calculates the specificity
for each class in a confusion matrix.
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class
    in a confusion matrix.

    Parameters:
    confusion (numpy.ndarray): confusion matrix of shape
        (classes, classes) where rows represent correct labels
        and columns represent predicted labels.

    Returns:
    numpy.ndarray: specificity of each class.
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives
    total = np.sum(confusion)
    true_negatives = total - (
        true_positives + false_positives + false_negatives
    )
    actual_negatives = true_negatives + false_positives
    return true_negatives / actual_negatives
