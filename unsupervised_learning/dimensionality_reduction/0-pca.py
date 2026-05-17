#!/usr/bin/env python3
"""
PCA module
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset and returns the weight matrix W.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        The dataset (assumed centered with mean 0 per feature)
    var : float
        Fraction of variance to retain

    Returns
    -------
    W : numpy.ndarray of shape (d, nd)
        Projection matrix that retains `var` variance
    """
    n = X.shape[0]

    # Covariance matrix (d x d)
    cov = np.dot(X.T, X) / (n - 1)

    # Eigen decomposition (symmetric matrix)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # Compute cumulative variance ratio
    total_variance = np.sum(eig_vals)
    cum_variance = np.cumsum(eig_vals) / total_variance

    # Select number of components
    nd = np.searchsorted(cum_variance, var) + 1

    # Projection matrix
    W = eig_vecs[:, :nd]

    return W
