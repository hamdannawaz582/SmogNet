# SmogNet

A desktop computer vision pipeline for traffic scenarios that automatically detects and handles foggy conditions. It uses a three stage inference pipeline with both a programmatic API and an interactive PySide6 GUI.

**Inspired by:** [mena-aq/smog-vision](https://github.com/mena-aq/smog-vision)

## Overview

The pipeline classifies images as clear or foggy, optionally dehazes foggy images, and performs object detection on multiple vehicle types and people.

### Pipeline Stages

1. **Classification** (128x128): MobileNetV2-based model determines if an image is `Clear` or `Foggy`
2. **Dehazing** (256x256): U-Net model removes haze if image is classified as `Foggy`
3. **Object Detection** (640x640): YOLOv8-xs detector localizes 8 object classes (person, car, rider, bicycle, motorcycle, bus, caravan, truck)

The pipeline (`src/inference/pipeline.py`) handles coordinate rescaling and annotation rendering.

## Quick Start

### Setup
Clone the repository and download weights from the [releases page](https://github.com/hamdannawaz582/SmogNet/releases) and put them in `models/`.
Download dependencies with:

```bash
uv sync
uv pip install -e .
```

### Running

```bash
uv run main.py
```

Then drag and drop an image onto the input panel and click any action button.

**Note for macOS (Apple Silicon):** The `pyproject.toml` automatically installs `tensorflow-macos` and `tensorflow-metal` for GPU acceleration via Metal.

### API Example

```python
from src.inference.pipeline import Pipeline
from PIL import Image
import numpy as np

# Initialize pipeline
pipeline = Pipeline(
    classifier_path="models/mobilenetv2.keras",
    dehazer_path="models/unet_best_weights.h5",
    detector_path="models/yolo_best_weights.keras",
)

# Load image
img = Image.open('input.jpg').convert('RGB')
img_array = np.array(img)

# Run full pipeline (classify -> dehaze if foggy -> detect -> annotate)
result = pipeline.run_pipeline(img_array)

# Save annotated output
output_image = Image.fromarray(result)
output_image.save("output.jpg")
```

## Project Structure

```
SmogNet/
├── main.py                              # GUI entry point & PySide6 app
├── src/
│   ├── inference/
│   │   ├── pipeline.py                  # Pipeline orchestrator
│   │   ├── smog_classifier.py           # Classification model
│   │   ├── dehazer.py                   # Dehazing model
│   │   └── object_detector.py           # Object detection model
│   └── widgets/
│       ├── button.py                    # Styled action buttons
│       ├── image_container.py           # Image display & drag-drop
│       └── label.py                     # Text labels
├── notebooks/                           # Training & experimentation
└── docs/                                # MkDocs documentation
```

The GUI expects model weights to be in the `models/` directory. You can download the pre-trained models from the releases page.

## Architecture Highlights

### Smog Classification

- **Model:** MobileNetV2 (fine-tuned), custom classification head trained for 10 epochs, backbone for 20 epochs at small learning rates.
- **Input:** 128x128 RGB images since the task doesn't require particularly high resolution
- **Output:** Binary classification ("Clear" or "Foggy")
- **Dataset:** [Fog-or-Smog Detection Dataset](https://www.kaggle.com/datasets/ahmedislam0/fog-or-smog-detection-dataset) (note: dataset quality is low, a custom or higher-quality dataset is recommended, can probably use RESIDE for this too)

I plan on replacing this with a custom CNN sooner or later to save on parameters.

### Dehazing

- **Model:** Custom U-Net with residual prediction
- **Input:** 256x256 RGB images (normalized to [0, 1])
- **Output:** 256x256 dehazed RGB images
- **Dataset:** [RESIDE-6K](https://www.kaggle.com/datasets/kmljts/reside-6k)

**Design:**
- Predicts residuals instead of regenerating the entire image, which significantly reduces artifacts
- Bottleneck has 768 channels (instead of 1024) to reduce parameters
- Architecture is rebuilt programmatically from `_build_model()` to avoid TensorFlow version incompatibilities. Weights are loaded separately with `model.load_weights()`.

I'm not sure why but I can't get > 25-26 dB PSNR with U-Net on this task, I've tried a few different architectures but they converge around 25 dB and start overfitting after that.
This residual based architecture seems to perform better in terms of artifacts but the PSNR isn't much better. My training method is in `notebooks/dehazing.ipynb`. Please open an issue if you know a better way to go about this.

### Object Detection

- **Model:** KerasCV YOLOv8-xs (fine-tuned)
- **Input:** 640x640 RGB images (normalized to [0, 1])
- **Output:** Bounding boxes, class indices, and confidence scores
- **Classes:** person, car, rider, bicycle, motorcycle, bus, caravan, truck
- **Dataset:** [BDD100K](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k) (a subset of ~10% used due to dataset size, also this dataset is just really amazing)

## GUI Features

### Action Buttons

- **Run Classification:** Infers smog label and displays result in text box to the bottom left
- **Run Dehazing:** Dehazes the image and displays on main image container
- **Run Object Detection:** Detects objects and draws annotated bounding boxes on the main image container
- **Auto Run:** Runs all three stages in sequence

### UI Elements

- Drag-and-drop input panel (left)
- Main output canvas (right of the screen)
- Execution timer and classification label display to the bottom left

## Troubleshooting

### TensorFlow Version Mismatches

If loading a full `.keras` model fails, try using the dehazer approach:
- Rebuild the model architecture programmatically based on the `notebooks/` notebooks
- Load weights separately with `model.load_weights(weights_path)`

See `src/inference/dehazer.py` for an example.

## Datasets

| Task           | Dataset               | Link                                                                                | Notes                                     |
|----------------|-----------------------|-------------------------------------------------------------------------------------|-------------------------------------------|
| Classification | Fog-or-Smog Detection | [Kaggle](https://www.kaggle.com/datasets/ahmedislam0/fog-or-smog-detection-dataset) | Low quality, not recommended              |
| Dehazing       | RESIDE-6K             | [Kaggle](https://www.kaggle.com/datasets/kmljts/reside-6k)                          | High-quality indoor/outdoor hazy images   |
| Detection      | BDD100K               | [Kaggle](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)             | Large-scale traffic dataset, 10%-ish used |

## Future Roadmap

- **AQI Prediction:** Integrate air quality index estimation based on detected haze levels
- **Model Quantization:** Add quantized weights and inference files for TFLite for edge computing
- **Custom Classifier:** Replace MobileNetV2 with a custom architecture tailored to the binary classification task to reduce parameters and latency

## Development

### Project Layout for Developers

- **GUI changes:** `main.py` + `src/widgets/`
- **Inference changes:** `src/inference/`
- **Model loading:** `src/inference/{smog_classifier.py,dehazer.py,object_detector.py}`
- **Pipeline orchestration:** `src/inference/pipeline.py`
- **Training/experimentation:** `notebooks/`

### Model Management

- Update model paths in `main.py` (lines 57–59) if switching weights
- When adding new dependencies, use `uv` and the `pyproject.toml` file will be updated automatically

## Documentation

Documentation is available on GitHub Pages @ [SmogNet Docs](https://hamdannawaz582.github.io/SmogNet)

API documentation is also available in `docs/` and can be served by MkDocs locally with:

```bash
mkdocs serve
```

## License

See `LICENSE` file.

## References

- **Inspiration:** [mena-aq/smog-vision](https://github.com/mena-aq/smog-vision)
- **MobileNetV2:** [MobileNetV2](https://keras.io/api/applications/mobilenet/)
- **U-Net:** [U-Net](https://arxiv.org/abs/1505.04597)
- **YOLOv8:** [YOLOv8](https://www.kaggle.com/models/kerashub/yolov8)
- **Datasets:** [fog-or-smog-detection-dataset](https://www.kaggle.com/datasets/ahmedislam0/fog-or-smog-detection-dataset), [RESIDE-6K](https://www.kaggle.com/datasets/kmljts/reside-6k), [BDD100K](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)
