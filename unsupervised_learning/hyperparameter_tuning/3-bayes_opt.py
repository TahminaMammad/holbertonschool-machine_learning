#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Initialize Bayesian Optimization.

        Args:
            f (function): black-box function to optimize.
            X_init (numpy.ndarray): sampled input points.
            Y_init (numpy.ndarray): outputs of sampled points.
            bounds (tuple): (min, max) bounds of search space.
            ac_samples (int): number of acquisition samples.
            l (float): length parameter for kernel.
            sigma_f (float): standard deviation of output.
            xsi (float): exploration-exploitation factor.
            minimize (bool): True for minimization,
                False for maximization.
        """
        self.f = f

        self.gp = GP(X_init, Y_init, l, sigma_f)

        self.X_s = np.linspace(
            bounds[0],
            bounds[1],
            ac_samples
        ).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize
