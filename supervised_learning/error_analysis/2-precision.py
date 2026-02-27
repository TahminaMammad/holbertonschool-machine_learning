#!/usr/bin/env python3
"""
Module that calculates the precision
for each class in a confusion matrix.
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class
    in a confusion matrix.

    Parameters:
    confusion (numpy.ndarray): confusion matrix of shape
        (classes, classes) where rows represent correct labels
        and columns represent predicted labels.

    Returns:
    numpy.ndarray: precision of each class.
    """
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    return true_positives / predicted_positives
