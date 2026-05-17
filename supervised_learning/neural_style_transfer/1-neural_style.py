#!/usr/bin/env python3
"""Module that defines the NST class for neural style transfer."""

import numpy as np
import tensorflow as tf


class NST:
    """Class that performs tasks for neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1',
                    'block5_conv1']

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        """
        Initialize the NST instance.

        Args:
            style_image (np.ndarray): style reference image
            content_image (np.ndarray): content reference image
            alpha (float): content weight
            beta (float): style weight
        """

        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError(
                "alpha must be a non-negative number"
            )

        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError(
                "beta must be a non-negative number"
            )

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescale an image so that its pixel values are between
        0 and 1 and its largest side is 512 pixels.

        Args:
            image (np.ndarray): image to scale

        Returns:
            tf.Tensor: scaled image tensor
        """

        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        if h > w:
            new_h = 512
            new_w = int((w * 512) / h)
        else:
            new_w = 512
            new_h = int((h * 512) / w)

        image = tf.convert_to_tensor(image, dtype=tf.float32)

        image = tf.image.resize(
            image,
            (new_h, new_w),
            method=tf.image.ResizeMethod.BICUBIC
        )

        image = image / 255.0

        image = tf.clip_by_value(image, 0.0, 1.0)

        image = tf.expand_dims(image, axis=0)

        return image

    def load_model(self):
        """
        Create the model used to calculate cost.
        """

        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg.trainable = False

        outputs = []

        x = vgg.input

        for layer in vgg.layers[1:]:

            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding
                )

            x = layer(x)

            if layer.name in self.style_layers:
                outputs.append(x)

            if layer.name == self.content_layer:
                outputs.append(x)

        self.model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=outputs
        )

        self.model.trainable = False
