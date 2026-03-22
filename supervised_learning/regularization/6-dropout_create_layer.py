#!/usr/bin/env python3
"""
Module to create a layer with dropout using TensorFlow
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function for the new layer
        keep_prob: probability that a node will be kept
        training: boolean indicating whether the model is in training mode

    Returns:
        The output of the new layer
    """
    # Initialize weights using VarianceScaling as per requirements
    init = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode=("fan_avg")
    )

    # Create the Dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )

    # Pass the previous output through the dense layer
    output = layer(prev)

    # Create the Dropout layer
    # rate = 1 - keep_prob (TF uses drop probability, not keep probability)
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)

    # Apply dropout to the output of the dense layer
    # training=training ensures dropout is only applied during training
    return dropout(output, training=training)
