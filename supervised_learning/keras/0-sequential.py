#!/usr/bin/env python3
"""Builds a neural network using Keras"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library

    Args:
        nx (int): number of input features
        layers (list): number of nodes in each layer
        activations (list): activation functions for each layer
        lambtha (float): L2 regularization parameter
        keep_prob (float): probability of keeping a node during dropout

    Returns:
        model: the keras sequential model
    """

    model = K.models.Sequential()

    for i in range(len(layers)):

        if i == 0:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.L2(lambtha),
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.L2(lambtha)
            ))

        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
