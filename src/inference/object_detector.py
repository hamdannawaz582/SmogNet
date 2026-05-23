"""Object detection inference module using a fine-tuned KerasCV YOLOv8-XS model."""

from typing import TypedDict

import keras_cv
import numpy as np
import tensorflow as tf


CLASSES = ["person", "car", "rider", "bicycle", "motorcycle", "bus", "caravan", "truck"]


class Predictions(TypedDict):
    """Typed dictionary returned by :meth:`ObjectDetector.infer`.

    Attributes:
        boxes: Float array of shape ``(N, 4)`` with ``xyxy`` coordinates scaled to the 640×640 inference resolution.
        classes: Integer array of shape ``(N,)`` with class indices into :data:`CLASSES`.
        confidence: Float array of shape ``(N,)`` with per-detection scores in ``[0, 1]``.
    """

    boxes: np.ndarray
    classes: np.ndarray
    confidence: np.ndarray


class ObjectDetector:
    """Detects objects in a single image using a fine-tuned YOLOv8-XS model.

    The model was trained on a subset of BDD100K and detects eight categories: ``"person"``, ``"car"``, ``"rider"``,
    ``"bicycle"``, ``"motorcycle"``, ``"bus"``, ``"caravan"``, and ``"truck"``.

    Args:
        weights_path: Path to the ``.keras`` checkpoint file whose weights will be loaded into the model.
    """

    _NUM_CLASSES = len(CLASSES)
    _INPUT_SIZE = (640, 640)

    def __init__(self, weights_path: str) -> None:
        """Builds the YOLOv8-XS detector and loads weights from ``weights_path``.

        Args:
            weights_path: Path to the ``.keras`` file containing saved weights.
        """
        backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_xs_backbone_coco",
            load_weights=False,
        )
        self.model = keras_cv.models.YOLOV8Detector(
            num_classes=self._NUM_CLASSES,
            bounding_box_format="xyxy",
            backbone=backbone,
            fpn_depth=1,
        )
        self.model.load_weights(weights_path)

    def infer(self, image: np.ndarray) -> Predictions:
        """Runs object detection on a single image.

        The image is resized to ``(640, 640)`` and normalised to ``[0, 1]`` before being passed to the model.

        Args:
            image: Input image as a NumPy array of shape ``(H, W, 3)`` with uint8 or float pixel values.

        Returns:
            A :class:`Predictions` dict with keys ``"boxes"``, ``"classes"``, and ``"confidence"``, each a NumPy array
            for the ``N`` detections that survived NMS.
        """
        resized = tf.image.resize(image, self._INPUT_SIZE)
        preprocessed = tf.cast(resized, tf.float32) / 255.0
        batch = tf.expand_dims(preprocessed, axis=0)

        raw = self.model.predict(batch)

        return Predictions(
            boxes=raw["boxes"][0],
            classes=raw["classes"][0].astype(int),
            confidence=raw["confidence"][0],
        )
