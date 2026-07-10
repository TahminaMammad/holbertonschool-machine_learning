#!/usr/bin/env python3
"""Converts a Gensim Word2Vec model to a Keras Embedding layer."""

import keras


def gensim_to_keras(model):
    """
    Convert a Gensim Word2Vec model to a trainable Keras layer.

    Args:
        model: A trained Gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer.
    """
    vectors = model.wv.vectors

    embedding = keras.layers.Embedding(
        input_dim=vectors.shape[0],
        output_dim=vectors.shape[1],
        trainable=True
    )

    embedding.build((1,))
    embedding.set_weights([vectors])

    return embedding
