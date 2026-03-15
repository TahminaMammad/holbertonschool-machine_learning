#!/usr/bin/env python3
"""Creates a batch normalization layer for a neural network in TensorFlow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.

    prev: tensor, activated output of the previous layer
    n: number of nodes in the layer to be created
    activation: activation function applied to the output

    Returns: tensor of the activated output for the layer
    """
    # Dense layer with VarianceScaling initializer
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg')
    )(prev)

    # Batch normalization layer
    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,
        scale=True
    )(dense)

    # Apply activation function
    output = activation(batch_norm)

    return output
