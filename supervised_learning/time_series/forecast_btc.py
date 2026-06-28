#!/usr/bin/env python3
"""
Forecast Bitcoin prices using an LSTM network.

This script:
- Loads the preprocessed Bitcoin dataset.
- Creates sequences of the previous 24 hours.
- Splits the data into training, validation, and testing sets.
- Builds an LSTM model.
- Trains the model using tf.data.Dataset.
- Evaluates the model.
- Saves the trained model.
"""

import numpy as np
import pandas as pd
import tensorflow as tf


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume_(BTC)",
    "Volume_(Currency)",
    "Weighted_Price"
]

SEQUENCE_LENGTH = 1440
BATCH_SIZE = 64
EPOCHS = 20


def load_data(filename):
    """
    Load the processed dataset.

    Args:
        filename (str): CSV filename.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    return pd.read_csv(filename)


def create_sequences(data, sequence_length):
    """
    Create input/output sequences.

    Args:
        data (DataFrame): Processed data.
        sequence_length (int): Number of previous
            timesteps used as input.

    Returns:
        tuple:
            X (numpy.ndarray)
            y (numpy.ndarray)
    """
    values = data[FEATURES].values

    X = []
    y = []

    for i in range(len(values) - sequence_length - 1):
        X.append(values[i:i + sequence_length])

        # Close price is index 3
        y.append(values[i + sequence_length][3])

    return np.array(X), np.array(y)


def split_data(X, y):
    """
    Split data into train, validation,
    and test sets.

    Returns:
        tuple
    """
    total = len(X)

    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_valid = X[train_end:valid_end]
    y_valid = y[train_end:valid_end]

    X_test = X[valid_end:]
    y_test = y[valid_end:]

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    )


def make_dataset(X, y, training=False):
    """
    Convert arrays into tf.data.Dataset.

    Args:
        X (ndarray): Inputs.
        y (ndarray): Targets.
        training (bool): Whether to shuffle.

    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if training:
        dataset = dataset.shuffle(10000)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def build_model():
    """
    Build the LSTM model.

    Returns:
        tf.keras.Model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(SEQUENCE_LENGTH, len(FEATURES))
        ),

        tf.keras.layers.LSTM(
            64,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(32),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model
def main():
    """
    Load the data, train the model,
    evaluate it, and save it.
    """
    print("Loading processed dataset...")
    data = load_data("btc_processed.csv")

    print("Creating sequences...")
    X, y = create_sequences(
        data,
        SEQUENCE_LENGTH
    )

    print("Splitting dataset...")
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test
    ) = split_data(X, y)

    print("Building tf.data datasets...")

    train_ds = make_dataset(
        X_train,
        y_train,
        training=True
    )

    valid_ds = make_dataset(
        X_valid,
        y_valid
    )

    test_ds = make_dataset(
        X_test,
        y_test
    )

    print("Building model...")
    model = build_model()

    model.summary()

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    print("Training...")

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=[callback]
    )

    print("\nEvaluating...")

    loss, mae = model.evaluate(
        test_ds,
        verbose=2
    )

    print(f"\nTest MSE : {loss:.6f}")
    print(f"Test MAE : {mae:.6f}")

    model.save("btc_forecast.keras")

    print("\nModel saved as btc_forecast.keras")


if __name__ == "__main__":
    main()
