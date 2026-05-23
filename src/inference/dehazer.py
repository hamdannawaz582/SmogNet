"""Image dehazing inference module using a custom U-Net tensorflow model."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def _build_model() -> tf.keras.Model:
    """Constructs a mini U-Net architecture.

    The layout is the same as the model defined in ``notebooks/dehazing.ipynb`` so that saved weights can be loaded without
    requiring the original ``.keras`` file format to be compatible across TensorFlow versions.

    Returns:
        An uncompiled  U-Net-ish ``tf.keras.Model``.
    """

    def DecoderBlock(filters, inputs):
        x = layers.Conv2D(filters, 3, padding="same")(inputs)
        x = layers.Activation("relu")(x)
        return x

    def EncoderBlock(filters, inputs):
        x = layers.Conv2D(filters, 3, padding="same")(inputs)
        x = layers.Activation("relu")(x)
        return x

    inputs = layers.Input(shape=(256, 256, 3))
    x = EncoderBlock(64, inputs)
    x = EncoderBlock(64, x)

    sk1 = x
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = EncoderBlock(128, x)
    x = EncoderBlock(128, x)

    sk2 = x
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = EncoderBlock(256, x)
    x = EncoderBlock(256, x)

    sk3 = x
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = EncoderBlock(512, x)
    x = EncoderBlock(512, x)

    sk4 = x
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = DecoderBlock(768, x)
    x = DecoderBlock(768, x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, sk4])

    x = DecoderBlock(512, x)
    x = DecoderBlock(512, x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, sk3])

    x = DecoderBlock(256, x)
    x = DecoderBlock(256, x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, sk2])

    x = DecoderBlock(128, x)
    x = DecoderBlock(128, x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, sk1])

    x = DecoderBlock(64, x)
    x = DecoderBlock(64, x)

    x = layers.Conv2D(3, 1, activation="tanh", padding="same")(x)
    x = layers.Add()([x, inputs])
    outputs = layers.Lambda(lambda x: tf.clip_by_value(x, 0, 1))(x)
    return tf.keras.Model(inputs, outputs)


class DehazeUNet:
    """Removes haze from an image using a trained U-Net model.

    The model expects inputs normalised to ``[0, 1]`` (divide by 255) and produces outputs in the same range, which are
    denormalised back to ``uint8`` before being returned.

    Because of potential TensorFlow version mismatches between the training and inference environments, the architecture
    is rebuilt locally and only the weights are loaded from disk.

    Args:
        weights_path: Path to the ``.keras`` checkpoint file whose weights will be loaded into the rebuilt model.
    """

    def __init__(self, weights_path: str) -> None:
        """Builds the U-Net and loads weights from ``weights_path``.

        Args:
            weights_path: Path to the ``.keras`` file containing saved weights.
        """
        self.model = _build_model()
        self.model.load_weights(weights_path)

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Dehazes a single image.

        The image is resized to ``(256, 256)``, normalised to ``[0, 1]``, passed through the U-Net, and the output is
        denormalised back to ``[0, 255]`` before being returned.

        Args:
            image: Non-normalizd Input image as a NumPy array of shape ``(H, W, 3)`` with uint8 or float pixel values.

        Returns:
            Non-normalized Dehazed image as a ``uint8`` NumPy array of shape ``(256, 256, 3)``.
        """
        resized = tf.image.resize(image, (256, 256))
        preprocessed = tf.cast(resized, tf.float32) / 255.0
        batch = tf.expand_dims(preprocessed, axis=0)
        output = self.model(batch, training=False)
        return (output[0].numpy() * 255).clip(0, 255).astype(np.uint8)
