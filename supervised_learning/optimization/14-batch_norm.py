#!/usr/bin/env python3
"""
Module to create a batch normalization layer in TensorFlow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow

    Args:
        prev: the activated output of the previous layer
        n: the number of nodes in the layer to be created
        activation: the activation function that should be used on the output
                    of the layer

    Returns:
        A tensor of the activated output for the layer
    """
    # 1. Setup the Kernel Initializer
    # mode='fan_avg' ensures weights are scaled based on input/output size
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # 2. Create the Dense layer
    # IMPORTANT: use_bias=False because BatchNormalization provides 'beta'
    # activation=None because we normalize BEFORE activating
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )
    Z = dense(prev)

    # 3. Create the Batch Normalization layer
    # gamma_init default is 'ones', beta_init default is 'zeros'
    batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-7)
    Z_norm = batch_norm(Z)

    # 4. Apply activation if it exists
    if activation is None:
        return Z_norm

    return activation(Z_norm)
