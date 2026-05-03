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
        """Process outputs (same as Task 1)"""

        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]
        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            cx = np.arange(grid_w)
            cy = np.arange(grid_h)

            cx = cx.reshape(1, grid_w, 1)
            cy = cy.reshape(grid_h, 1, 1)

            cx = np.tile(cx, (grid_h, 1, anchor_boxes))
            cy = np.tile(cy, (1, grid_w, anchor_boxes))

            bx = (self.sigmoid(tx) + cx) / grid_w
            by = (self.sigmoid(ty) + cy) / grid_h

            anchor_w = self.anchors[i, :, 0].reshape((1, 1, anchor_boxes))
            anchor_h = self.anchors[i, :, 1].reshape((1, 1, anchor_boxes))

            bw = (anchor_w * np.exp(tw)) / input_w
            bh = (anchor_h * np.exp(th)) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            confidence = self.sigmoid(output[..., 4])
            confidence = np.expand_dims(confidence, axis=-1)
            box_confidences.append(confidence)

            class_probs = self.sigmoid(output[..., 5:])
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on objectness score and class probabilities
        """

        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, c, p in zip(boxes, box_confidences, box_class_probs):
            # Compute box scores
            scores = c * p  # shape: (..., classes)

            # Best class per box
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            # Apply threshold
            mask = class_scores >= self.class_t

            # Filter
            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        # Concatenate all outputs
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
