#!/usr/bin/env python3
"""
Defines a class NeuralNetwork that performs binary classification
including forward propagation and cost calculation.
"""
import numpy as np


class NeuralNetwork:
    """
    Represents a neural network with one hidden layer.
    """

    def __init__(self, nx, nodes):
        """
        Class constructor.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        Uses the sigmoid activation function for both layers.
        """
        # Hidden layer: Z1 = W1X + b1, A1 = sigmoid(Z1)
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        # Output layer: Z2 = W2A1 + b2, A2 = sigmoid(Z2)
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Args:
            Y: numpy.ndarray (1, m) with correct labels.
            A: numpy.ndarray (1, m) with activated output.
        Returns:
            The cost.
        """
        m = Y.shape[1]
        # Using 1.0000001 - A to avoid division by zero/log(0) errors
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1 / m) * np.sum(loss)
        return cost
