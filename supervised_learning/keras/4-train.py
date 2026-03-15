#!/usr/bin/env python3
"""Trains a keras model using mini-batch gradient descent"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent

    network: the model to train
    data: numpy.ndarray of shape (m, nx) containing the input data
    labels: one-hot numpy.ndarray of shape (m, classes) containing labels
    batch_size: size of batch used for mini-batch gradient descent
    epochs: number of passes through the data
    verbose: determines if output should be printed during training
    shuffle: determines whether to shuffle the data each epoch

    Returns: the History object generated after training
    """

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
