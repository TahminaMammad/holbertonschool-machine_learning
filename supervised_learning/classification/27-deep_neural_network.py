#!/usr/bin/env python3
"""27-deep_neural_network.py: DeepNeuralNetwork for multiclass classification"""
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    # ... [keep __init__, save, load, gradient_descent, train methods from 26] ...

    def forward_prop(self, X):
        """Forward propagation using sigmoid for hidden layers, softmax for output"""
        self.__cache["A0"] = X
        for i in range(self.L):
            W = self.__weights[f"W{i + 1}"]
            b = self.__weights[f"b{i + 1}"]
            A_prev = self.__cache[f"A{i}"]
            Z = np.dot(W, A_prev) + b

            if i == self.L - 1:
                # Output layer: softmax for multiclass
                t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = t / np.sum(t, axis=0, keepdims=True)
            else:
                # Hidden layers: sigmoid
                A = 1 / (1 + np.exp(-Z))
            self.__cache[f"A{i + 1}"] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Compute cost using categorical cross-entropy"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate the network’s predictions"""
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        cost = self.cost(Y, A)
        return A, cost
