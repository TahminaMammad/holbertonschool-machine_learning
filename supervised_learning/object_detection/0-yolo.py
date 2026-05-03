#!/usr/bin/env python3
"""
Yolo v3 Object Detection Module
"""

import tensorflow.keras as K


class Yolo:
    """
    Yolo class uses the YOLO v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """
        Initialize the Yolo model

        Parameters:
        model_path (str): path to Darknet Keras model
        classes_path (str): path to class names file
        class_t (float): box score threshold
        nms_t (float): IOU threshold for non-max suppression
        anchors (numpy.ndarray): anchor boxes
        """

        # Load model
        self.model = K.models.load_model(model_path)

        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        # Store thresholds and anchors
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
