#!/usr/bin/env python3
"""
Defines a class DeepNeuralNetwork for binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Represents a deep neural network.
    """

    def __init__(self, nx, layers):
        """
        Class constructor.
        Args:
            nx: number of input features.
            layers: list of nodes in each layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # He et al. initialization (The only allowed loop)
        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            # Layer index starts at 1
            l_idx = i + 1
            # Input size for the first layer is nx, otherwise it's layers[i-1]
            prev_size = nx if i == 0 else layers[i - 1]

            # He initialization: standard normal * sqrt(2 / n_prev)
            he_init = np.sqrt(2 / prev_size)
            self.weights[f"W{l_idx}"] = (np.random.randn(layers[i], prev_size) *
                                         he_init)
            self.weights[f"b{l_idx}"] = np.zeros((layers[i], 1))
