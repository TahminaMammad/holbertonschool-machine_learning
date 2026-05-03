#!/usr/bin/env python3
"""
Yolo v3 Object Detection module
"""
import cv2
import glob
import numpy as np
import os
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

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

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

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cy = np.arange(grid_h).reshape(grid_h, 1, 1)

            bx = (self.sigmoid(t_x) + cx) / grid_w
            by = (self.sigmoid(t_y) + cy) / grid_h

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            bw = (pw * np.exp(t_w)) / input_w
            bh = (ph * np.exp(t_h)) / input_h

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

            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

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

        for cls in np.unique(box_classes):
            cls_indices = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]

            order = cls_scores.argsort()[::-1]
            keep = []

            while order.size > 0:
                i = order[0]
                keep.append(i)

                if order.size == 1:
                    break

                box_i = cls_boxes[i]
                others = cls_boxes[order[1:]]

                x1 = np.maximum(box_i[0], others[:, 0])
                y1 = np.maximum(box_i[1], others[:, 1])
                x2 = np.minimum(box_i[2], others[:, 2])
                y2 = np.minimum(box_i[3], others[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                area_others = (others[:, 2] - others[:, 0]) * \
                              (others[:, 3] - others[:, 1])
                union = area_i + area_others - intersection

                iou = intersection / union
                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), cls))
            predicted_box_scores.append(cls_scores[keep])

        return (np.concatenate(box_predictions, axis=0),
                np.concatenate(predicted_box_classes, axis=0),
                np.concatenate(predicted_box_scores, axis=0))

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a folder
        """
        image_paths = glob.glob(os.path.join(folder_path, '*'))
        images = [cv2.imread(path) for path in image_paths]
        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses a list of images for the Darknet model
        - images: list of images as numpy.ndarrays
        Returns: (pimages, image_shapes)
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            # Save original shape (height, width)
            image_shapes.append(img.shape[:2])

            # Resize image with inter-cubic interpolation
            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)

            # Rescale to [0, 1]
            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
