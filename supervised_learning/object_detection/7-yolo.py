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
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append(img.shape[:2])
            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            pimages.append(resized / 255.0)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with boundary boxes, classes, and scores
        """
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            class_idx = box_classes[i]
            class_name = self.class_names[class_idx]
            score = box_scores[i]

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = "{} {:.2f}".format(class_name, score)
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                        1, cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite(os.path.join('detections', file_name), image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Runs the full YOLO detection pipeline on all images in a folder
        - folder_path: path to the folder holding the images
        Returns: (predictions, image_paths)
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        # Get raw model outputs
        outputs = self.model.predict(pimages)

        predictions = []

        for i in range(len(images)):
            # Isolate the outputs for a single image
            # The model returns a list of arrays (one for each scale)
            image_outputs = [out[i] for out in outputs]

            # Process outputs to get boxes relative to original image size
            boxes, confs, probs = self.process_outputs(image_outputs,
                                                       image_shapes[i])

            # Filter boxes by threshold
            boxes, classes, scores = self.filter_boxes(boxes, confs, probs)

            # Apply NMS
            boxes, classes, scores = self.non_max_suppression(boxes,
                                                              classes,
                                                              scores)

            predictions.append((boxes, classes, scores))

            # Display results (file_name should be base name only)
            file_name = os.path.basename(image_paths[i])
            self.show_boxes(images[i], boxes, classes, scores, file_name)

        return predictions, image_paths
