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
        Processes raw outputs from the model to get boundary boxes
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

            # Convert to [x1, y1, x2, y2]
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

            # Objectness confidence and Class probabilities
            box_confidences.append(sigmoid(output[..., 4:5]))
            box_class_probs.append(sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on class threshold
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, conf, prob in zip(boxes, box_confidences, box_class_probs):
            scores = conf * prob
            b_class = np.argmax(scores, axis=-1)
            b_score = np.max(scores, axis=-1)

            mask = b_score >= self.class_t

            filtered_boxes.append(b[mask])
            box_classes.append(b_class[mask])
            box_scores.append(b_score[mask])

        return (np.concatenate(filtered_boxes, axis=0),
                np.concatenate(box_classes, axis=0),
                np.concatenate(box_scores, axis=0))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-max Suppression to filtered boxes
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Process one class at a time
        for cls in np.unique(box_classes):
            # Get indices for this specific class
            cls_indices = np.where(box_classes == cls)[0]

            # Isolate boxes and scores for the class
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]

            # Sort scores in descending order
            order = cls_scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                if order.size == 1:
                    break

                # Calculate IoU between the current box and the rest
                box_i = cls_boxes[i]
                others = cls_boxes[order[1:]]

                # Coordinates of intersection
                x1 = np.maximum(box_i[0], others[:, 0])
                y1 = np.maximum(box_i[1], others[:, 1])
                x2 = np.minimum(box_i[2], others[:, 2])
                y2 = np.minimum(box_i[3], others[:, 3])

                # Areas
                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                area_others = (others[:, 2] - others[:, 0]) * \
                              (others[:, 3] - others[:, 1])
                union = area_i + area_others - intersection

                iou = intersection / union

                # Keep only boxes with IoU less than threshold
                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            # Add selected boxes for this class to final list
            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), cls))
            predicted_box_scores.append(cls_scores[keep])

        # Concatenate and return final arrays
        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores
