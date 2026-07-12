#!/usr/bin/env python3
"""Module that defines a self-attention layer."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates attention for machine translation."""

    def __init__(self, units):
        """Initialize the self-attention layer.

        Args:
            units: Number of hidden units in the alignment model.
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculate the context vector and attention weights.

        Args:
            s_prev: Previous decoder hidden state with shape
                (batch, units).
            hidden_states: Encoder outputs with shape
                (batch, input_seq_len, units).

        Returns:
            context: Context vector with shape (batch, units).
            weights: Attention weights with shape
                (batch, input_seq_len, 1).
        """
        s_prev = tf.expand_dims(s_prev, axis=1)

        score = self.V(
            tf.nn.tanh(
                self.W(s_prev) + self.U(hidden_states)
            )
        )

        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
