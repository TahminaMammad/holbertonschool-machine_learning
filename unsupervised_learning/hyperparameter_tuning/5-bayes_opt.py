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
            X_init (numpy.ndarray): sampled input points.
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
            X_next (numpy.ndarray): next sample point.
            EI (numpy.ndarray): expected improvement.
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
            EI = EI + sigma * norm.pdf(Z)

            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimize the black-box function.

        Args:
            iterations (int): maximum number
                of iterations.

        Returns:
            X_opt (numpy.ndarray): optimal point.
            Y_opt (numpy.ndarray): optimal value.
        """
        for i in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.isclose(self.gp.X, X_next)):
                break

            Y_next = self.f(X_next)

            self.gp.update(X_next, Y_next)

        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        X_opt = self.gp.X[index]
        Y_opt = self.gp.Y[index]

        return X_opt, Y_opt
