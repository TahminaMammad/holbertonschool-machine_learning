#!/usr/bin/env python3
"""
PCA module
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset and returns projection matrix W
    that preserves `var` fraction of variance.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Dataset (assumed centered)
    var : float
        Fraction of variance to preserve

    Returns
    -------
    W : numpy.ndarray of shape (d, nd)
        Projection matrix
    """
    n = X.shape[0]

    # Covariance matrix
    cov = np.dot(X.T, X) / (n - 1)

    # Eigen decomposition (cov is symmetric)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Sort in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # Cumulative variance ratio
    total_variance = np.sum(eig_vals)
    cum_var = np.cumsum(eig_vals) / total_variance

    # FIX: correct component selection
    nd = 0
    for i in range(len(cum_var)):
        nd = i + 1
        if cum_var[i] >= var:
            break

    # Projection matrix
    W = eig_vecs[:, :nd]

    return W
