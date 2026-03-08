#!/usr/bin/env python3
"""
Defines a class NeuralNetwork that performs binary classification.
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
        """
        # Hidden layer
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        # Output layer
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        """
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1 / m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        Args:
            X: numpy.ndarray (nx, m) containing input data.
            Y: numpy.ndarray (1, m) containing correct labels.
        Returns:
            The prediction (ndarray) and the cost (float).
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        # Convert probabilities to binary 0 or 1
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost
