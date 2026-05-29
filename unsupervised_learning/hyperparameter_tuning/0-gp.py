#!/usr/bin/env python3
"""Gaussian Process module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor.

        Args:
            X_init (numpy.ndarray): shape (t, 1),
                inputs already sampled.
            Y_init (numpy.ndarray): shape (t, 1),
                outputs of the black-box function.
            l (float): length parameter for the kernel.
            sigma_f (float): standard deviation
                for the output function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix.

        Args:
            X1 (numpy.ndarray): shape (m, 1)
            X2 (numpy.ndarray): shape (n, 1)

        Returns:
            numpy.ndarray: covariance kernel matrix
                of shape (m, n)
        """
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
            np.sum(X2**2, axis=1) - 2 * np.matmul(X1, X2.T)

        return (self.sigma_f ** 2) * np.exp(
            -0.5 / (self.l ** 2) * sqdist
        )
