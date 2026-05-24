# SmogNet Documentation

SmogNet is a small desktop CV pipeline for traffic scenarios. The tool uses a three stage inference pipeline and 
exposes both an API (see `src/inference`) and a PySide6 GUI (`main.py`, `src/widgets`).

Pipeline summary

1. Classification: decides whether an image is `Clear` or `Foggy` using a MobileNetV2-based model (`src/inference/smog_classifier.py`). The classifier expects inputs resized to 128x128.
2. Dehazing: If classified as `Foggy`, a U-Net (`src/inference/dehazer.py`) dehazes the image at 256x256.
3. Object detection: a KerasCV YOLOv8-xs detector (`src/inference/object_detector.py`) runs at 640x640 and returns boxes, class indices and confidences.

The pipeline orchestrator `src/inference/pipeline.py` performs the full flow, rescales detector coordinates back to the original image size, and draws annotations using Pillow.

Run locally

## Quick Start

```bash
uv sync
uv pip install -e .
uv run main.py
```

API Example

```python
from inference.pipeline import Pipeline
from PIL import Image
import numpy as np

def main():
    pl = Pipeline(
        classifier_path="models/mobilenetv2.keras",
        dehazer_path="models/unet_best_weights.h5",
        detector_path="models/yolo_best_weights.keras",
    )
    img = Image.open('input.jpg').convert('RGB')

    result = pl.run_pipeline(np.array(img))

    output_image = Image.fromarray(result)
    output_image.save("output.jpg")
```

## API Reference

See the sidebar for full class and method documentation on both the Inference and GUI modules.

## Layout

Where to look when changing the project

- GUI layout: `main.py`
- Widgets: `src/widgets/` (`image_container.py`, `button.py`, `label.py`)
- Orchestrator: `src/inference/pipeline.py`
- Model loading and preprocessing: `src/inference/{smog_classifier.py,dehazer.py,object_detector.py}`

Notes

- If saved `.keras` models fail to load due to TensorFlow version mismatches, prefer rebuilding architectures based on `notebooks/` and
  using `model.load_weights(...)` (as done in `DehazeUNet`).
- Keras-CV and YOLO require compatible `keras-cv` and `tensorflow` versions — mismatches are a common runtime issue.

Models and data

- Model weights: `models/` (e.g. `unet_best_weights.h5`, `yolo_best_weights.keras`). Change these in `main.py` inside the `SmogNetWindow` class.
- Example images: `data/`.

This page is the project's documentation landing page for MkDocs.
