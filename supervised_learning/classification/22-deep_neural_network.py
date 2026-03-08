#!/usr/bin/env python3
"""
Deep Neural Network for binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network
        nx: number of input features
        layers: list representing the number of nodes in each layer
        """
        # (Assume your previous initialization code from task 21 is here)

    def forward_prop(self, X):
        """
        Forward propagation through the network
        """
        # (Already implemented in task 21)

    def cost(self, Y, A):
        """
        Compute cost using logistic regression
        """
        # (Already implemented in task 21)

    def evaluate(self, X, Y):
        """
        Evaluate predictions
        """
        # (Already implemented in task 21)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent
        """
        # (Already implemented in task 21)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network
        """
        # --- Validate inputs ---
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # --- Training loop (only one allowed) ---
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        # --- Return final evaluation ---
        return self.evaluate(X, Y)
