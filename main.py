"""SmogNet application entry point."""

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import time
from PIL import Image, ImageDraw
import numpy as np

from widgets.button import SmogButton
from widgets.image_container import ImageContainer

from src.inference.pipeline import Pipeline
from src.inference.object_detector import CLASSES as DET_CLASSES
from src.inference.pipeline import _CLASS_COLORS, _DETECTOR_INPUT_SIZE

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
)

current_image_path: str | None = None
"""Path of the most recently dropped image, or ``None`` if none has been loaded.

The GUI creates a :class:`Pipeline` instance at startup and uses its
subcomponents (classifier, dehazer, detector) for button actions.
"""


class SmogNetWindow(QMainWindow):
    """Main SmogNet application window.

    Accepts image files via drag-and-drop, places the dropped image in the
    small input container, and exposes buttons to run each inference stage.
    """

    def __init__(self) -> None:
        """Creates and lays out all child widgets."""
        super().__init__()
        self.setWindowTitle("SmogNet")
        self.setMinimumSize(960, 640)
        self.setStyleSheet("background-color: #1e1e2e;")
        self.setAcceptDrops(True)

        # Create pipeline at launch using weights from models/
        classifier_path = "models/mobilenetv2.keras"
        dehazer_path = "models/unet_best_weights.h5"
        detector_path = "models/yolo_best_weights.keras"
        self.pipeline = Pipeline(classifier_path, dehazer_path, detector_path)

        # Couldn't figure out a better place for these
        self.__button_callbacks = {
            "Run Classification": self.on_clicked_classify,
            "Run Dehazing": self.on_clicked_dehaze,
            "Run Object Detection": self.on_clicked_detect,
            "Auto Run": self.on_clicked_auto,
        }

        # Actually put the UI together
        self._build_ui()

    def _build_ui(self) -> None:
        """Assembles the UI layout."""
        central = QWidget()
        central.setStyleSheet("background-color: #1e1e2e;")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(
            "QSplitter::handle { background-color: #585b70; border-radius: 3px; }" # Catppuccin Mocha surface
        )
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        # Left panel ~25 %, right panel ~75 %
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        root.addWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        """Builds the left panel: input image container above the action buttons.

        Returns:
            The assembled left panel widget.
        """
        panel = QWidget()
        panel.setStyleSheet("background-color: #1e1e2e;") # Catppuccin Mocha bg

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(8)

        self.input_container = ImageContainer()
        layout.addWidget(self.input_container, stretch=1)

        # Create buttons and keep references so we can enable/disable them
        self._buttons: dict[str, SmogButton] = {}
        for label in [
            "Run Classification",
            "Run Dehazing",
            "Run Object Detection",
            "Auto Run",
        ]:
            btn = SmogButton(label)
            btn.clicked.connect(self.__button_callbacks[label])
            btn.setEnabled(False)
            self._buttons[label] = btn
            layout.addWidget(btn)


        self.timer_box = QLineEdit("Run something...")
        self.timer_box.setReadOnly(True)
        layout.addWidget(self.timer_box)

        self.output_box = QLineEdit("")
        self.output_box.setReadOnly(True)
        layout.addWidget(self.output_box)

        return panel

    def _build_right_panel(self) -> QWidget:
        """Builds the right panel containing the main output image container.

        Returns:
            The assembled right panel widget.
        """
        panel = QWidget()
        panel.setStyleSheet("background-color: #1e1e2e;") # Catppuccin Mocha bg

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(0)

        self.main_container = ImageContainer()
        layout.addWidget(self.main_container)

        return panel

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accepts drag events containing at least one supported image URL.

        Args:
            event: The incoming drag-enter event.
        """
        if event.mimeData().hasUrls() and any(
            Path(url.toLocalFile()).suffix.lower() in _SUPPORTED_EXTENSIONS
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Loads the first valid dropped image into the input container.

        Updates :data:`current_image_path` with the resolved file path.

        Args:
            event: The incoming drop event.
        """
        global current_image_path

        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in _SUPPORTED_EXTENSIONS:
                current_image_path = path
                self.input_container.load_image(path)
                # enable action buttons after an image has been loaded
                self._set_buttons_enabled(True)
                # reset text boxes
                self.timer_box.setText("Run something...")
                self.output_box.setText("")
                event.acceptProposedAction()
                return

        event.ignore()

    def on_clicked_classify(self) -> None:
        """Runs the classification stage on the currently loaded image."""
        if current_image_path is None or self.pipeline is None:
            return

        self._set_buttons_enabled(False)
        try:
            img = self._read_image(current_image_path)
            start = time.perf_counter()
            label = self.pipeline.classifier.infer(img)
            elapsed = time.perf_counter() - start
            # show results
            self.output_box.setText(label)
            self.timer_box.setText(f"{elapsed:.3f}s")
        finally:
            self._set_buttons_enabled(True)

    def on_clicked_dehaze(self) -> None:
        """Runs the dehazing stage on the currently loaded image."""
        if current_image_path is None or self.pipeline is None:
            return

        self._set_buttons_enabled(False)
        try:
            img = self._read_image(current_image_path)
            orig_h, orig_w = img.shape[:2]
            start = time.perf_counter()
            dehazed = self.pipeline.dehazer.infer(img)
            elapsed = time.perf_counter() - start
            # dehazer returns 256x256, upscaled to original size for display
            try:
                resample = Image.Resampling.BILINEAR
            except Exception:
                # Older Pillow fallback (use getattr to avoid linter errors)
                resample = getattr(Image, "BILINEAR", Image.NEAREST)
            display = np.array(Image.fromarray(dehazed).resize((orig_w, orig_h), resample))
            self.main_container.load_image_from_array(display)
            self.timer_box.setText(f"Time Taken: {elapsed:.3f}s")
        finally:
            self._set_buttons_enabled(True)

    def on_clicked_detect(self) -> None:
        """Runs the object detection stage on the currently loaded image."""
        if current_image_path is None or self.pipeline is None:
            return

        self._set_buttons_enabled(False)
        try:
            img = self._read_image(current_image_path)
            orig_h, orig_w = img.shape[:2]
            start = time.perf_counter()
            predictions = self.pipeline.detector.infer(img)
            elapsed = time.perf_counter() - start

            # Annotate boxes (predictions are in 640x640 coordinates)
            scale_x = orig_w / float(_DETECTOR_INPUT_SIZE)
            scale_y = orig_h / float(_DETECTOR_INPUT_SIZE)
            canvas = Image.fromarray(img.copy())
            draw = ImageDraw.Draw(canvas)
            for box, cls_idx, conf in zip(predictions["boxes"], predictions["classes"], predictions["confidence"]):
                x1 = box[0] * scale_x
                y1 = box[1] * scale_y
                x2 = box[2] * scale_x
                y2 = box[3] * scale_y
                color = _CLASS_COLORS[cls_idx % len(_CLASS_COLORS)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1 + 2, max(y1 - 14, 0)), f"{DET_CLASSES[cls_idx]} {conf:.2f}", fill=color)

            self.main_container.load_image_from_array(np.array(canvas))
            self.timer_box.setText(f"{elapsed:.3f}s")
        finally:
            self._set_buttons_enabled(True)

    def on_clicked_auto(self) -> None:
        """Runs all inference stages on the currently loaded image."""
        if current_image_path is None or self.pipeline is None:
            return

        self._set_buttons_enabled(False)
        try:
            img = self._read_image(current_image_path)
            start = time.perf_counter()
            out = self.pipeline.run_pipeline(img)
            elapsed = time.perf_counter() - start
            self.main_container.load_image_from_array(out)
            self.timer_box.setText(f"{elapsed:.3f}s")
        finally:
            self._set_buttons_enabled(True)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        """Enable or disable all action buttons.

        Args:
            enabled: True to enable buttons, False to disable.
        """
        for btn in getattr(self, "_buttons", {}).values():
            btn.setEnabled(enabled)

    def _read_image(self, path: str) -> np.ndarray:
        """Read an image from *path* and return an RGB NumPy array.

        Args:
            path: Filesystem path to the image.

        Returns:
            NumPy array of shape ``(H, W, 3)`` with dtype ``uint8`` (RGB order).
        """
        pil = Image.open(path).convert("RGB")
        return np.array(pil)


def main() -> None:
    """Launches the SmogNet application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SmogNetWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
