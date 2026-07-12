#!/usr/bin/env python3
"""RNN encoder module for machine translation."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encodes an input sequence using an embedding layer and a GRU."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNN encoder.

        Args:
            vocab: Size of the input vocabulary.
            embedding: Dimensionality of the embedding vectors.
            units: Number of hidden units in the GRU.
            batch: Batch size.
        """
        super().__init__()

        self.batch = batch
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

    def initialize_hidden_state(self):
        """Return an initial hidden state containing zeros."""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Perform the forward pass of the encoder.

        Args:
            x: Tensor of input word indices.
            initial: Tensor containing the initial hidden state.

        Returns:
            A tuple containing all outputs and the final hidden state.
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
