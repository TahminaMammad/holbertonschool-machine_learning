#!/usr/bin/env python3
"""Module that defines a single neuron for binary classification"""

import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize the neuron

        Parameters:
        nx (int): number of input features
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weights vector
        self.W = np.random.randn(1, nx)

        # Bias
        self.b = 0

        # Activated output
        self.A = 0
