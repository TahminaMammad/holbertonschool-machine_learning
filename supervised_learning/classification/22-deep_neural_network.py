#!/usr/bin/env python3
"""
DeepNeuralNetwork training implementation
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialization logic (One loop used)
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            l_idx = i + 1
            prev_size = nx if i == 0 else layers[i - 1]
            he_init = np.sqrt(2 / prev_size)
            self.__weights[f"W{l_idx}"] = (
                np.random.randn(layers[i], prev_size) * he_init
            )
            self.__weights[f"b{l_idx}"] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """cache getter"""
        return self.__cache

    @property
    def weights(self):
        """weights getter"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation"""
        self.__cache["A0"] = X
        # Internal loop for layer iteration
        for i in range(self.__L):
            w = self.__weights[f"W{i + 1}"]
            b = self.__weights[f"b{i + 1}"]
            prev_a = self.__cache[f"A{i}"]
            z = np.matmul(w, prev_a) + b
            self.__cache[f"A{i + 1}"] = 1 / (1 + np.exp(-z))
        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """Calculates cost of the model"""
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return (1 / m) * np.sum(loss)

    def evaluate(self, X, Y):
        """Evaluates predictions"""
        a, _ = self.forward_prop(X)
        cost = self.cost(Y, a)
        prediction = np.where(a >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent"""
        m = Y.shape[1]
        # Backwards loop through layers
        for i in range(self.__L, 0, -1):
            a_curr = cache[f"A{i}"]
            a_prev = cache[f"A{i - 1}"]
            w_key = f"W{i}"
            b_key = f"b{i}"

            if i == self.__L:
                dz = a_curr - Y
            else:
                w_next = self.__weights[f"W{i + 1}"]
                dz = np.matmul(w_next.T, dz) * (a_curr * (1 - a_curr))

            dw = (1 / m) * np.matmul(dz, a_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            self.__weights[w_key] -= alpha * dw
            self.__weights[b_key] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network
        Args:
            X: input data (nx, m)
            Y: correct labels (1, m)
            iterations: number of training passes
            alpha: learning rate
        Returns:
            Evaluation of training data after training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (float, int)):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # The only allowed loop for training iterations
        for i in range(iterations):
            A_final, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)
