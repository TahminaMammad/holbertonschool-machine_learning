#!/usr/bin/env python3
"""
Module that calculates the F1 score
for each class in a confusion matrix.
"""

import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class
    in a confusion matrix.

    Parameters:
    confusion (numpy.ndarray): confusion matrix of shape
        (classes, classes) where rows represent correct labels
        and columns represent predicted labels.

    Returns:
    numpy.ndarray: F1 score of each class.
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)
    return 2 * (prec * sens) / (prec + sens)
