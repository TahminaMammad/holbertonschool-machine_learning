#!/usr/bin/env python3
"""RNN Cell"""

import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """
        Class constructor.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (ndarray): Previous hidden state of shape (m, h).
            x_t (ndarray): Input data of shape (m, i).

        Returns:
            h_next (ndarray): Next hidden state.
            y (ndarray): Output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(
            np.matmul(concat, self.Wh) + self.bh
        )

        y_linear = np.matmul(h_next, self.Wy) + self.by

        exp_y = np.exp(
            y_linear - np.max(
                y_linear,
                axis=1,
                keepdims=True
            )
        )
        y = exp_y / np.sum(
            exp_y,
            axis=1,
            keepdims=True
        )

        return h_next, y
