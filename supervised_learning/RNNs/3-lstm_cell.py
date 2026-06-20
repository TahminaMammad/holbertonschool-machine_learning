#!/usr/bin/env python3
"""
LSTM Cell implementation
"""

import numpy as np


class LSTMCell:
    """
    Represents a Long Short-Term Memory (LSTM) cell
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))

        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))

        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))

        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))

        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev (ndarray): shape (m, h), previous hidden state
            c_prev (ndarray): shape (m, h), previous cell state
            x_t (ndarray): shape (m, i), data input for the cell

        Returns:
            h_next (ndarray): next hidden state
            c_next (ndarray): next cell state
            y (ndarray): output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        u = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)

        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_tilde

        o = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)
        h_next = o * np.tanh(c_next)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp_y = np.exp(
            y_linear - np.max(y_linear, axis=1, keepdims=True)
        )
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, c_next, y

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))
