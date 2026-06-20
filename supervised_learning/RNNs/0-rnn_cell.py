#!/usr/bin/env python3
"""
RNN Cell implementation
"""

import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """
        # Weights for hidden state + input
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        # Weights for output
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
        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state using tanh activation
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        # Compute raw output
        y_linear = np.matmul(h_next, self.Wy) + self.by

        # Apply softmax activation
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y

