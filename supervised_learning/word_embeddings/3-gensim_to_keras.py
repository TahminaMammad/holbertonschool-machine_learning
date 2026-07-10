#!/usr/bin/env python3
"""Converts a Gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Convert a Gensim Word2Vec model to a trainable Keras Embedding layer.

    Args:
        model: A trained Gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer containing the Word2Vec weights.
    """
    vectors = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        input_dim=vectors.shape[0],
        output_dim=vectors.shape[1],
        trainable=True
    )

    embedding.build((None,))
    embedding.set_weights([vectors])

    return embedding
