#!/usr/bin/env python3
"""
GRU Cell implementation
"""

import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit (GRU) cell
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))

        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))

        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev (ndarray): shape (m, h), previous hidden state
            x_t (ndarray): shape (m, i), data input for the cell

        Returns:
            h_next (ndarray): next hidden state
            y (ndarray): output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde

        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp_y = np.exp(
            y_linear - np.max(y_linear, axis=1, keepdims=True)
        )
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))
