#!/usr/bin/env python3
"""
Module to train a model and save the best iteration
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a model with optional early stopping, learning rate decay,
    and checkpointing to save the best model based on validation loss.
    """
    callbacks = []

    # Handle Learning Rate Decay
    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            """Inverse time decay function"""
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callbacks.append(lr_decay)

    # Handle Early Stopping
    if early_stopping and validation_data:
        stop = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(stop)

    # Handle Saving the Best Model
    if save_best and validation_data and filepath:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        callbacks.append(checkpoint)

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
