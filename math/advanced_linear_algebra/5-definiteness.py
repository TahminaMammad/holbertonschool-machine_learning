#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix
"""

import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a square numpy matrix

    Args:
        matrix (numpy.ndarray): square matrix

    Returns:
        str: Positive definite, Positive semi-definite,
             Negative semi-definite, Negative definite, Indefinite, or None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.size == 0:
        return None

    # Must be symmetric for definiteness
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    pos = np.all(eigenvalues > 0)
    pos_semi = np.all(eigenvalues >= 0) and np.any(eigenvalues == 0)
    neg = np.all(eigenvalues < 0)
    neg_semi = np.all(eigenvalues <= 0) and np.any(eigenvalues == 0)

    if pos:
        return "Positive definite"
    elif pos_semi:
        return "Positive semi-definite"
    elif neg:
        return "Negative definite"
    elif neg_semi:
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None
