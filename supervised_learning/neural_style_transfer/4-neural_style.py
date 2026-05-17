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
                "style_image must be a numpy.ndarray with "
                "shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with "
                "shape (h, w, 3)"
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
        self.generate_features()

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

        x = vgg.input
        outputs_dict = {}

        for layer in vgg.layers[1:]:

            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )

            x = layer(x)

            outputs_dict[layer.name] = x

        outputs = [outputs_dict[name]
                   for name in self.style_layers]

        outputs.append(outputs_dict[self.content_layer])

        self.model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=outputs
        )

        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate the gram matrix of an input layer.

        Args:
            input_layer (tf.Tensor or tf.Variable):
                tensor of shape (1, h, w, c)

        Returns:
            tf.Tensor: gram matrix of shape (1, c, c)
        """

        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )

        _, h, w, c = input_layer.shape

        features = tf.reshape(input_layer, [-1, c])

        gram = tf.matmul(features, features, transpose_a=True)

        gram = gram / tf.cast(h * w, tf.float32)

        gram = tf.expand_dims(gram, axis=0)

        return gram

    def generate_features(self):
        """
        Extract the style and content features used to calculate
        neural style cost.
        """

        style_image = self.style_image * 255.0
        content_image = self.content_image * 255.0

        style_image = tf.keras.applications.vgg19.preprocess_input(
            style_image
        )

        content_image = tf.keras.applications.vgg19.preprocess_input(
            content_image
        )

        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        style_features = style_outputs[:len(self.style_layers)]

        self.gram_style_features = [
            self.gram_matrix(style_feature)
            for style_feature in style_features
        ]

        self.content_feature = content_outputs[
            len(self.style_layers)
        ]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculate the style cost for a single layer.

        Args:
            style_output (tf.Tensor or tf.Variable):
                tensor of shape (1, h, w, c)

            gram_target (tf.Tensor or tf.Variable):
                tensor of shape (1, c, c)

        Returns:
            tf.Tensor: layer style cost
        """

        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        _, _, _, c = style_output.shape

        expected_shape = [1, c, c]

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != expected_shape):
            raise TypeError(
                "gram_target must be a tensor of shape "
                "[1, {}, {}]".format(c, c)
            )

        gram_style = self.gram_matrix(style_output)

        cost = tf.reduce_sum(
            tf.square(gram_style - gram_target)
        )

        cost = cost / tf.cast(c ** 2, tf.float32)

        return cost
