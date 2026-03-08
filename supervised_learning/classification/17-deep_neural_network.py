#!/usr/bin/env python3
"""
Defines a class DeepNeuralNetwork with private attributes.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Represents a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Class constructor.
        Args:
            nx: number of input features.
            layers: list containing the number of nodes in each layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialization logic (Only one loop)
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            l_idx = i + 1
            # Input size: nx if first layer, else previous layer nodes
            prev_size = nx if i == 0 else layers[i - 1]

            # He et al. initialization
            he_init = np.sqrt(2 / prev_size)
            self.__weights[f"W{l_idx}"] = (
                np.random.randn(layers[i], prev_size) * he_init
            )
            self.__weights[f"b{l_idx}"] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the dictionary of intermediary values"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the dictionary of weights and biases"""
        return self.__weights
