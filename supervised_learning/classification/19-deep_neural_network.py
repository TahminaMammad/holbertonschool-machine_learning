#!/usr/bin/env python3
"""
Defines a class DeepNeuralNetwork that defines a deep neural network
performing binary classification, including forward propagation and
cost calculation.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Represents a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Class constructor
        Args:
            nx: number of input features
            layers: list of nodes in each layer
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

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            l_idx = i + 1
            prev_size = nx if i == 0 else layers[i - 1]

            # He et al. initialization
            he_init = np.sqrt(2 / prev_size)
            self.__weights[f"W{l_idx}"] = (
                np.random.randn(layers[i], prev_size) * he_init
            )
            self.__weights[f"b{l_idx}"] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """
        Performs forward propagation
        Args:
            X: numpy.ndarray of shape (nx, m) with input data
        Returns:
            Output of the neural network and the cache
        """
        self.__cache['A0'] = X
        A_prev = X

        for layer in range(1, self.__L + 1):
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]
            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache[f"A{layer}"] = A
            A_prev = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression
        Args:
            Y: numpy.ndarray of shape (1, m) with correct labels
            A: numpy.ndarray of shape (1, m) with activated output
        Returns:
            Cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost
