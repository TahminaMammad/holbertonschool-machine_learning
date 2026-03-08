#!/usr/bin/env python3
"""One-hot decode a one-hot encoded matrix"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot encoded numpy.ndarray into a vector of labels.
    one_hot: numpy.ndarray of shape (classes, m)
    Returns: numpy.ndarray of shape (m,) with numeric labels, or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    # Use argmax along axis 0 to get the class index for each example
    return np.argmax(one_hot, axis=0)
