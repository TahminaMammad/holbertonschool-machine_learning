#!/usr/bin/env python3
"""
Yolo v3 Object Detection module
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Yolo class for object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes the raw outputs from the Darknet model
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        img_h, img_w = image_size
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Extract raw values
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Create grid for offsets
            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cy = np.arange(grid_h).reshape(grid_h, 1, 1)

            # Sigmoid activations
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            # Box center coordinates
            bx = (sigmoid(t_x) + cx) / grid_w
            by = (sigmoid(t_y) + cy) / grid_h

            # Box dimensions
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            bw = (pw * np.exp(t_w)) / input_w
            bh = (ph * np.exp(t_h)) / input_h

            # Rescale to original image size
            # x1, y1, x2, y2 format
            x1 = (bx - bw / 2) * img_w
            y1 = (by - bh / 2) * img_h
            x2 = (bx + bw / 2) * img_w
            y2 = (by + bh / 2) * img_h

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

            # Objectness confidence
            conf = sigmoid(output[..., 4:5])
            box_confidences.append(conf)

            # Class probabilities
            probs = sigmoid(output[..., 5:])
            box_class_probs.append(probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on objectness score and class probabilities
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, conf, prob in zip(boxes, box_confidences, box_class_probs):
            # Calculate full score: confidence * class probability
            # shape will be (grid_h, grid_w, anchor_boxes, classes)
            scores = conf * prob

            # Find best class and its score for each box
            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)

            # Create mask for scores above the threshold
            mask = box_score >= self.class_t

            # Apply mask to collect specific values
            filtered_boxes.append(b[mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        # Concatenate lists from all output layers into single arrays
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
