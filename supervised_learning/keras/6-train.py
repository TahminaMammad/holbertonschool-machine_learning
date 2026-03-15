#!/usr/bin/env python3
"""Trains a keras model using mini-batch gradient descent with
validation and optional early stopping"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent

    network: the model to train
    data: numpy.ndarray of shape (m, nx)
    labels: one-hot numpy.ndarray of shape (m, classes)
    batch_size: size of mini-batches
    epochs: number of epochs
    validation_data: tuple (X_val, Y_val)
    early_stopping: whether to apply early stopping
    patience: patience for early stopping
    verbose: training verbosity
    shuffle: whether to shuffle data

    Returns: History object
    """

    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )

    return history
