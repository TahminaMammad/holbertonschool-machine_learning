#!/usr/bin/env python3
"""Trains a keras model with optional validation, early stopping, and learning rate decay"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent with
    optional validation, early stopping, and learning rate decay

    network: the model to train
    data: numpy.ndarray of shape (m, nx)
    labels: one-hot numpy.ndarray of shape (m, classes)
    batch_size: size of mini-batches
    epochs: number of epochs
    validation_data: tuple (X_val, Y_val)
    early_stopping: whether to apply early stopping
    patience: patience for early stopping
    learning_rate_decay: whether to apply inverse time decay
    alpha: initial learning rate
    decay_rate: decay rate
    verbose: training verbosity
    shuffle: whether to shuffle data

    Returns: History object
    """

    callbacks = []

    # Early stopping
    if early_stopping and validation_data is not None:
        es = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(es)

    # Learning rate decay
    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch, lr):
            new_lr = alpha / (1 + decay_rate * epoch)
            print(f"Epoch {epoch + 1}: LearningRateScheduler setting learning rate to {new_lr}.")
            return new_lr

        lr_decay = K.callbacks.LearningRateScheduler(schedule=scheduler, verbose=0)
        callbacks.append(lr_decay)

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
