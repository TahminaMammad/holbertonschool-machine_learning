#!/usr/bin/env python3
"""One-hot encode a numeric label vector"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector Y into a one-hot matrix
    Y: numpy.ndarray of shape (m,) with numeric class labels
    classes: maximum number of classes
    Returns: one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    try:
        one_hot[Y, np.arange(m)] = 1
    except IndexError:
        return None
    return one_hot
