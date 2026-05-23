"""Smog-aware object detection pipeline."""

import numpy as np
from PIL import Image, ImageDraw

from .dehazer import DehazeUNet
from .object_detector import CLASSES, ObjectDetector
from .smog_classifier import SmogClassifier

_DETECTOR_INPUT_SIZE = 640

# One visually distinct colour per class (RGB)
_CLASS_COLORS = [
    (220,  20,  60),  # person       – crimson
    ( 30, 144, 255),  # car          – dodger blue
    (255, 140,   0),  # rider        – dark orange
    ( 50, 205,  50),  # bicycle      – lime green
    (148,   0, 211),  # motorcycle   – dark violet
    (255, 215,   0),  # bus          – gold
    (  0, 206, 209),  # caravan      – dark turquoise
    (255,  69,   0),  # truck        – orange-red
]


class Pipeline:
    """Classifies, dehazes if foggy, detects objects, annotates.

    Steps:

    1. :class:`SmogClassifier` decides whether the image is ``"Clear"`` or ``"Foggy"``.
    2. If foggy, :class:`DehazeUNet` removes the haze before detection.
    3. :class:`ObjectDetector` localises objects and returns bounding boxes in 640×640 space.
    4. Detections are drawn on the (possibly dehazed) image and returned at the original input resolution.

    Args:
        classifier_path: Path to the SmogClassifier ``.keras`` model file.
        dehazer_path: Path to the DehazeUNet ``.keras`` weights file.
        detector_path: Path to the ObjectDetector ``.keras`` weights file.
    """

    def __init__(self, classifier_path: str, dehazer_path: str, detector_path: str) -> None:
        """Instantiates all three models.

        Args:
            classifier_path: Path to the SmogClassifier ``.keras`` model file.
            dehazer_path: Path to the DehazeUNet ``.keras`` weights file.
            detector_path: Path to the ObjectDetector ``.keras`` weights file.
        """
        self.classifier = SmogClassifier(classifier_path)
        self.dehazer = DehazeUNet(dehazer_path)
        self.detector = ObjectDetector(detector_path)

    def run_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Runs the full pipeline on a single image.

        Args:
            image: Input image as a ``uint8`` NumPy array of shape
                ``(H, W, 3)``.

        Returns:
            Annotated ``uint8`` NumPy array of shape ``(H, W, 3)`` at the original input resolution with bounding boxes
            and ``class confidence`` labels drawn for each detection.
        """
        orig_h, orig_w = image.shape[:2]

        # Classification
        smog_label = self.classifier.infer(image)

        print(f"Image is {smog_label}")

        display = None

        # Dehazing if needed
        if smog_label == "Foggy":
            print("Dehazing...")
            dehazed = self.dehazer.infer(image)
            display = np.array(
                Image.fromarray(dehazed).resize((orig_w, orig_h), Image.BILINEAR)
            )
            detection_input = dehazed
        else:
            display = image
            detection_input = image

        # Detect objects
        predictions = self.detector.infer(detection_input)

        # Upscale bounding boxes
        scale_x = orig_w / _DETECTOR_INPUT_SIZE
        scale_y = orig_h / _DETECTOR_INPUT_SIZE

        canvas = Image.fromarray(display)
        draw = ImageDraw.Draw(canvas)

        for box, cls_idx, conf in zip(
            predictions["boxes"],
            predictions["classes"],
            predictions["confidence"],
        ):
            x1 = box[0] * scale_x
            y1 = box[1] * scale_y
            x2 = box[2] * scale_x
            y2 = box[3] * scale_y

            color = _CLASS_COLORS[cls_idx % len(_CLASS_COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 2, max(y1 - 14, 0)), f"{CLASSES[cls_idx]} {conf:.2f}", fill=color)

        return np.array(canvas)
