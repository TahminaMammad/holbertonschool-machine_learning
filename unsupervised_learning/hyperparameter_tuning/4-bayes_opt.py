#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization."""

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Initialize Bayesian Optimization.

        Args:
            f (function): black-box function.
            X_init (numpy.ndarray): sampled inputs.
            Y_init (numpy.ndarray): sampled outputs.
            bounds (tuple): search space bounds.
            ac_samples (int): number of acquisition samples.
            l (float): kernel length parameter.
            sigma_f (float): kernel standard deviation.
            xsi (float): exploration-exploitation factor.
            minimize (bool): determines minimization
                or maximization.
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

    def acquisition(self):
        """
        Calculate the next best sample location.

        Uses the Expected Improvement acquisition function.

        Returns:
            X_next (numpy.ndarray): next best sample point.
            EI (numpy.ndarray): expected improvement values.
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            optimum = np.min(self.gp.Y)
            improvement = optimum - mu - self.xsi
        else:
            optimum = np.max(self.gp.Y)
            improvement = mu - optimum - self.xsi

        with np.errstate(divide='warn'):
            Z = improvement / sigma

            EI = improvement * norm.cdf(Z)
            EI += sigma * norm.pdf(Z)

            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
