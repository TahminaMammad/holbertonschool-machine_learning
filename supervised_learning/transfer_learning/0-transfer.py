#!/usr/bin/env python3
"""
Transfer learning script for CIFAR-10 classification
"""

from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Preprocesses CIFAR-10 data

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3)
        Y: numpy.ndarray of shape (m,)

    Returns:
        X_p, Y_p
    """
    X_p = K.applications.mobilenet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    # Load dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Input layer
    inputs = K.Input(shape=(32, 32, 3))

    # Resize to 224x224
    x = K.layers.Lambda(
        lambda image: K.backend.resize_images(image, 7, 7, "bilinear")
    )(inputs)

    # Load pretrained model
    base_model = K.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze most layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model(x, training=False)

    # Custom classifier
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)

    # Compile
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=10,
        batch_size=64
    )

    # Save model
    model.save("cifar10.h5")
