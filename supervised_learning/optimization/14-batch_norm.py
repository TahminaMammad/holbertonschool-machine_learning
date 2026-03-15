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
        activation: the activation function to be used on the output
                    of the layer

    Returns:
        A tensor of the activated output for the layer
    """
    # 1. Initialize weights using VarianceScaling
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # 2. Create the base Dense layer
    # We set use_bias=False because Batch Norm has its own beta parameter
    # that acts as a bias.
    model = tf.keras.layers.Dense(units=n,
                                  kernel_initializer=init,
                                  use_bias=False)
    Z = model(prev)

    # 3. Create the Batch Normalization layer
    # gamma_initializer='ones' and beta_initializer='zeros' are defaults
    batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-7)

    # 4. Apply normalization to the unactivated output
    Z_norm = batch_norm(Z)

    # 5. Apply the activation function
    if activation is None:
        return Z_norm

    return activation(Z_norm)
