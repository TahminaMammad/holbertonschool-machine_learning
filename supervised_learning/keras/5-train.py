#!/usr/bin/env python3
"""Trains a keras model using mini-batch gradient descent with validation"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent

    network: the model to train
    data: numpy.ndarray of shape (m, nx)
    labels: one-hot numpy.ndarray of shape (m, classes)
    batch_size: size of the batch
    epochs: number of epochs
    validation_data: tuple (X_val, Y_val) for validation
    verbose: determines if output should be printed
    shuffle: determines whether to shuffle batches

    Returns: the History object generated after training
    """

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )

    return history
