#!/usr/bin/env python3
"""RNN decoder module for machine translation."""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decodes a target sequence using attention and a GRU."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNN decoder.

        Args:
            vocab: Size of the output vocabulary.
            embedding: Dimensionality of the embedding vectors.
            units: Number of hidden units in the GRU.
            batch: Batch size.
        """
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Perform the forward pass of the decoder.

        Args:
            x: Previous target word with shape (batch, 1).
            s_prev: Previous decoder hidden state.
            hidden_states: Outputs from the encoder.

        Returns:
            y: Output vocabulary scores.
            s: New decoder hidden state.
        """
        context, _ = self.attention(s_prev, hidden_states)

        x = self.embedding(x)
        context = tf.expand_dims(context, axis=1)
        x = tf.concat([context, x], axis=-1)

        outputs, s = self.gru(x, initial_state=s_prev)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)

        return y, s
