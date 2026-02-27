#!/usr/bin/env python3
"""
Module that creates a confusion matrix for classification tasks.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Parameters
    ----------
    labels : numpy.ndarray
        One-hot array of shape (m, classes) containing the correct labels.
    logits : numpy.ndarray
        One-hot array of shape (m, classes) containing the predicted labels.

    Returns
    -------
    numpy.ndarray
        Confusion matrix of shape (classes, classes) where rows represent
        the correct labels and columns represent the predicted labels.
    """
    # Convert one-hot vectors to class indices
    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    # Populate confusion matrix
    for t, p in zip(true_labels, pred_labels):
        confusion[t, p] += 1

    return confusion
