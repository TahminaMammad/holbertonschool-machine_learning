#!/usr/bin/env python3
"""Module that calculates the sensitivity for each class in a confusion matrix."""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Parameters:
    confusion (numpy.ndarray): A confusion matrix of shape (classes, classes)
                               where rows represent correct labels and
                               columns represent predicted labels.

    Returns:
    numpy.ndarray: A 1D array containing the sensitivity for each class.
    """
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    return true_positives / actual_positives
