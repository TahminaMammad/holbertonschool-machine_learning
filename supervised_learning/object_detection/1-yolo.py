#!/usr/bin/env python3
"""
Yolo v3 Object Detection
"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """Initialize Yolo"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes YOLO outputs

        Parameters:
        outputs (list): list of numpy arrays
        image_size (numpy.ndarray): [image_height, image_width]

        Returns:
        boxes, box_confidences, box_class_probs
        """

        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Extract components
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            # 🔥 Correct grid generation (important fix)
            cx = np.arange(grid_w)
            cy = np.arange(grid_h)

            cx = cx.reshape(1, grid_w, 1)
            cy = cy.reshape(grid_h, 1, 1)

            cx = np.tile(cx, (grid_h, 1, anchor_boxes))
            cy = np.tile(cy, (1, grid_w, anchor_boxes))

            # Center coordinates
            bx = (self.sigmoid(tx) + cx) / grid_w
            by = (self.sigmoid(ty) + cy) / grid_h

            # Anchors
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            anchor_w = anchor_w.reshape((1, 1, anchor_boxes))
            anchor_h = anchor_h.reshape((1, 1, anchor_boxes))

            # Width and height
            bw = (anchor_w * np.exp(tw)) / input_w
            bh = (anchor_h * np.exp(th)) / input_h

            # Convert to corner coordinates
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            # Box confidence
            confidence = self.sigmoid(output[..., 4])
            confidence = np.expand_dims(confidence, axis=-1)
            box_confidences.append(confidence)

            # Class probabilities
            class_probs = self.sigmoid(output[..., 5:])
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
