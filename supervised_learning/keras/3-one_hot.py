#!/usr/bin/env python3
"""Converts a label vector into a one-hot matrix"""

import numpy as np


def one_hot(labels, classes=None):
    """converts a label vector into a one-hot matrix

    labels: numpy.ndarray of shape (m,) containing numeric class labels
    classes: number of classes

    Returns: one-hot matrix of shape (m, classes)
    """

    if classes is None:
        classes = np.max(labels) + 1

    m = labels.shape[0]

    one_hot_matrix = np.zeros((m, classes))
    one_hot_matrix[np.arange(m), labels] = 1

    return one_hot_matrix
