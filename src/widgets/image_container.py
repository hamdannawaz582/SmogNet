"""Image-display widgets for the SmogNet application."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QLabel, QSizePolicy
import numpy as np

_STYLE = """
QLabel {
    background-color: #1e1e2e;
    border: 2px solid #585b70;
    border-radius: 6px;
    color: #585b70;
    font-size: 13px;
}
"""

_PLACEHOLDER = "Drop image here"


class ImageContainer(QLabel):
    """A QLabel that scales and centres a loaded image.

    Displays placeholder text when no image has been loaded.

    Args:
        parent: Optional parent widget.
    """

    def __init__(self, parent=None) -> None:
        """Initialises an empty container showing placeholder text.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(_PLACEHOLDER, parent)
        self.setStyleSheet(_STYLE)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(100, 100)
        self._source: QPixmap | None = None

    def load_image(self, path: str) -> None:
        """Loads an image from *path* and displays it scaled to fit.

        Args:
            path: Absolute or relative file-system path to the image.
        """
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self._source = pixmap
            self.setText("")
            self._refresh()

    def load_image_from_array(self, array: np.ndarray) -> None:
        """Loads an image from a NumPy array and displays it scaled to fit.

        The array is expected to be a HxWx3 uint8 RGB image.

        Args:
            array: NumPy array with shape ``(H, W, 3)`` and dtype ``uint8``.
        """
        if array is None:
            return

        arr = np.ascontiguousarray(array)
        h, w = arr.shape[:2]
        # Format RGB888, bytes per line = 3 * width
        bytes_per_line = 3 * w
        qimage = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # Make a deep copy so the QImage does not depend on the numpy buffer lifetime
        qimage = qimage.copy()
        pixmap = QPixmap.fromImage(qimage)
        if not pixmap.isNull():
            self._source = pixmap
            self.setText("")
            self._refresh()

    def clear_image(self) -> None:
        """Removes the current image and restores the placeholder text."""
        self._source = None
        super().setPixmap(QPixmap())
        self.setText(_PLACEHOLDER)

    def _refresh(self) -> None:
        """Rescales the stored pixmap to fit the current widget geometry."""
        if self._source is not None:
            scaled = self._source.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        """Rescales the image to match the new widget size.

        Args:
            event: Qt resize event.
        """
        super().resizeEvent(event)
        self._refresh()
