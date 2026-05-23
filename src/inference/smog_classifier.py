"""Smog classification inference module using a MobileNetV2-based tensorflow model."""

import numpy as np
import tensorflow as tf


class SmogClassifier:
    """Classifies images as either Clear or Foggy using a trained tensorflow model.

    The model is expected to be a finetune of MobileNetV2 and/or have used ``tf.keras.applications.mobilenet_v2``
    as the preprocessing function during training.

    Args:
        model_path: Path to the ``.keras`` file containing the trained model.
    """

    CLASSES = ["Clear", "Foggy"]

    def __init__(self, model_path: str) -> None:
        """Loads the tensorflow model from disk.

        Args:
            model_path: Path to the ``.keras`` file to load.
        """
        self.model: tf.keras.Model = tf.keras.models.load_model(model_path, custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input})

    def infer(self, image: np.ndarray) -> str:
        """Runs inference on a single image and returns the predicted class label.

        The image is preprocessed with the MobileNetV2 preprocessing function before being passed to the model.

        Args:
            image: A NumPy array of shape ``(H, W, 3)`` with uint8 or float pixel values.  The image will be resized
            to ``(128, 128)`` internally.

        Returns:
            The predicted class label, either ``"Clear"`` or ``"Foggy"``.
        """
        resized = tf.image.resize(image, (128, 128))
        # preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
        batch = tf.expand_dims(resized, axis=0)
        predictions = self.model(batch, training=False)
        class_index = int(predictions[0] > 0.5)
        return self.CLASSES[class_index]
