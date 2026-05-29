#!/usr/bin/env python3
"""Bayesian Optimization using GPyOpt."""

import numpy as np
import GPyOpt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import os


def build_model(lr, units, dropout, l2_reg, batch_size):
    """Build and compile a simple neural network."""
    model = models.Sequential([
        layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            input_shape=(20,)
        ),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(params):
    """Training function for BO."""
    lr = float(params[:, 0])
    units = int(params[:, 1])
    dropout = float(params[:, 2])
    l2_reg = float(params[:, 3])
    batch_size = int(params[:, 4])

    # dummy dataset (replace with real dataset if needed)
    x_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 2, 1000)
    x_val = np.random.randn(200, 20)
    y_val = np.random.randint(0, 2, 200)

    model = build_model(lr, units, dropout, l2_reg, batch_size)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    checkpoint_name = (
        f"model_lr{lr:.4f}_u{units}_d{dropout:.2f}"
        f"_l2{l2_reg:.4f}_b{batch_size}.keras"
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_name,
        save_best_only=True,
        monitor='val_loss'
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop, checkpoint]
    )

    best_val_acc = max(history.history['val_accuracy'])

    # we minimize => return negative accuracy
    return -best_val_acc


def run_bayes_opt():
    """Run Bayesian Optimization."""

    bounds = [
        {'name': 'lr', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
        {'name': 'units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
        {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'l2', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128)}
    ]

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=train_model,
        domain=bounds,
        acquisition_type='EI'
    )

    optimizer.run_optimization(max_iter=30)

    # plot convergence
    optimizer.plot_convergence()
    plt.savefig("convergence.png")

    # save report
    with open("bayes_opt.txt", "w") as f:
        f.write(str(optimizer.get_evaluations()))
        f.write("\n\nBest parameters:\n")
        f.write(str(optimizer.X[np.argmin(optimizer.Y)]))


if __name__ == "__main__":
    run_bayes_opt()
