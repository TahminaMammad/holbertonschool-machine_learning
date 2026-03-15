#!/usr/bin/env python3
"""
Module to train a model with learning rate decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional
    early stopping and learning rate decay.
    """
    callbacks = []

    # Handle Learning Rate Decay
    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            """Calculates inverse time decay for the current epoch"""
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callbacks.append(lr_decay)

    # Handle Early Stopping
    if early_stopping and validation_data:
        stop = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(stop)

    # Train the model
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
