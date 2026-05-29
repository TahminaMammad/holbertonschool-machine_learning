#!/usr/bin/env python3
"""Gaussian Process module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize the Gaussian process.

        Args:
            X_init (numpy.ndarray): shape (t, 1),
                sampled input points.
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
        Calculate the covariance kernel matrix.

        Args:
            X1 (numpy.ndarray): shape (m, 1)
            X2 (numpy.ndarray): shape (n, 1)

        Returns:
            numpy.ndarray: covariance matrix.
        """
        sqdist = (
            np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            + np.sum(X2 ** 2, axis=1)
            - 2 * np.matmul(X1, X2.T)
        )

        return (self.sigma_f ** 2) * np.exp(
            -0.5 * sqdist / (self.l ** 2)
        )

    def predict(self, X_s):
        """
        Predict the mean and variance of points.

        Args:
            X_s (numpy.ndarray): shape (s, 1),
                points to predict.

        Returns:
            mu (numpy.ndarray): shape (s,),
                predicted means.
            sigma (numpy.ndarray): shape (s,),
                predicted variances.
        """
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(K)

        mu_s = K_s.T @ K_inv @ self.Y
        cov_s = K_ss - K_s.T @ K_inv @ K_s

        mu = mu_s.reshape(-1)
        sigma = np.diag(cov_s)

        return mu, sigma
