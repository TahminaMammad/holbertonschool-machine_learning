#!/usr/bin/env python3
"""
PCA module (dimensionality reduction version)
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA and reduces X to ndim dimensions.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Input dataset (assumed centered or raw)
    ndim : int
        Target number of dimensions

    Returns
    -------
    T : numpy.ndarray of shape (n, ndim)
        Transformed dataset
    """
    n = X.shape[0]

    # Center data (important for correctness)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Covariance matrix
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    # Eigen decomposition
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Sort descending by eigenvalues
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]

    # Take top ndim components
    W = eig_vecs[:, :ndim]

    # Project data
    T = np.dot(X_centered, W)

    return T
