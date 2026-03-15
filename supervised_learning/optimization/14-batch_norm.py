#!/usr/bin/env python3
"""
Batch Normalization Layer Creation Module
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function to be applied on the output

    Returns:
        A tensor of the activated output for the layer
    """
    # Dense layer with VarianceScaling initializer
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg')
    )(prev)

    # Batch normalization with gamma=1, beta=0, epsilon=1e-7
    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,
        scale=True,
        beta_initializer=tf.keras.initializers.Zeros(),
        gamma_initializer=tf.keras.initializers.Ones()
    )(dense)

    # Apply activation function
    return activation(batch_norm)
