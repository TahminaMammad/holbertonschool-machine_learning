#!/usr/bin/env python3
"""Calculates the PDF of a Gaussian distribution."""


import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian.

    Args:
        X (numpy.ndarray): shape (n, d) containing data points
        m (numpy.ndarray): shape (d,) containing the mean
        S (numpy.ndarray): shape (d, d) containing covariance matrix

    Returns:
        P (numpy.ndarray): shape (n,) containing PDF values
        or None on failure
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2):
        return None

    if (not isinstance(m, np.ndarray) or
            len(m.shape) != 1):
        return None

    if (not isinstance(S, np.ndarray) or
            len(S.shape) != 2):
        return None

    n, d = X.shape

    if (m.shape[0] != d or
            S.shape != (d, d)):
        return None

    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)

        A = 1 / np.sqrt(((2 * np.pi) ** d) * det)

        diff = X - m

        exponent = -0.5 * np.sum((diff @ inv) * diff, axis=1)

        P = A * np.exp(exponent)

        return np.maximum(P, 1e-300)

    except Exception:
        return None
