# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nomeroff Net is an open-source Automatic Number Plate Recognition (ANPR/ALPR) framework based on YOLOv8 for detection and RNN-based OCR for text recognition. It supports multiple countries and license plate formats.

**Version**: 4.0.1
**Python Requirements**: >= 3.9
**Main Tech Stack**: PyTorch, PyTorch Lightning 1.8.6, Ultralytics YOLOv8, OpenCV, FastAPI/Flask/Tornado

## Installation and Setup

### Initial Setup
```bash
# Install system dependencies (Ubuntu/Debian)
apt-get install gcc libglib2.0 libgl1-mesa-glx python3.9-dev git libturbojpeg

# Install Python dependencies
pip3 install -r requirements.txt
```

### For Red Hat-based systems
```bash
yum install libSM python3-devel gcc git libjpeg-turbo-official
pip3 install -r requirements.txt
```

## Common Development Commands

### Running Tests
```bash
# Run all Python inference examples
python tutorials/py/inference/get-started-demo.py
python tutorials/py/inference/get-started-tiny-demo.py
python tutorials/py/inference/number-plate-filling-demo.py

# Run benchmark tests
python tutorials/py/benchmark/accuracy-test.py
python tutorials/py/benchmark/runtime-test.py

# Run specific tool tests
python3 nomeroff_net/tools/test_tools.py

# Test individual modules (run as module with -f flag)
python3 -m nomeroff_net.nnmodels.ocr_model -f nomeroff_net/nnmodels/ocr_model.py
python3 -m nomeroff_net.image_loaders.opencv_loader -f nomeroff_net/image_loaders/opencv_loader.py
```

### Running Jupyter Notebooks
```bash
# Execute and convert notebooks (use for testing)
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html tutorials/ju/inference/get-started-demo.ipynb
```

### Benchmarking
```bash
# Accuracy test with custom parameters
python tutorials/py/benchmark/accuracy-test.py \
    --pipeline_name=number_plate_detection_and_reading \
    --image_loader_name=turbo \
    --images_glob=./data/examples/benchmark_oneline_np_images/* \
    --test_file=./data/examples/accuracy_test_data.json

# Runtime test
python tutorials/py/benchmark/runtime-test.py \
    --pipeline_name=number_plate_detection_and_reading_runtime \
    --image_loader_name=turbo \
    --images_glob=./data/examples/benchmark_oneline_np_images/* \
    --num_run=10 \
    --batch_size=1 \
    --num_workers=1
```

### Docker
```bash
# Build and run with Docker
cd docker/
./build-cpu.sh      # CPU-only version
./build-gpu.sh      # GPU version
./build-tensorrt.sh # TensorRT optimized version

./run-cpu.sh        # Run CPU container
./run-gpu.sh        # Run GPU container
```

### Node.js Moderation App
```bash
cd moderation/
npm install
npm start           # Starts Koa server
npm test            # Run mocha tests
```

## High-Level Architecture

### Pipeline System
The framework uses a **pipeline-based architecture** where high-level pipelines compose low-level pipes:

**Pipelines** ([nomeroff_net/pipelines/](nomeroff_net/pipelines/)):
- `NumberPlateDetectionAndReading` - Full end-to-end detection and OCR
- `NumberPlateDetectionAndReadingRuntime` - Optimized runtime version
- `NumberPlateLocalization` - Only detect license plates
- `NumberPlateClassification` - Classify detected plates by region/type
- `NumberPlateTextReading` - Only OCR on cropped plates
- `NumberPlateFilling` - Fill and normalize plate images

Use pipelines via the factory function:
```python
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading",
                                              image_loader="opencv")
result = number_plate_detection_and_reading(['./data/examples/oneline_images/example1.jpeg'])
(images, images_bboxs, images_points, images_zones, region_ids,
 region_names, count_lines, confidences, texts) = unzip(result)
```

**Pipes** ([nomeroff_net/pipes/](nomeroff_net/pipes/)) - Low-level components:
- `number_plate_localizators/` - YOLO-based detection with bbox and keypoints
  - `Detector` - Main YOLOv8 detector class
  - `yolo_kp_detector.py` - Keypoint detection implementation
- `number_plate_classificators/` - Classify plates by region and orientation
  - `OptionsDetector` - Detects region (country) and count_lines
  - `orientation_detector.py` - Detects plate rotation
- `number_plate_text_readers/` - OCR engines
  - `TextDetector` - Main OCR class with country-specific models
  - `base/ocr.py` - RNN-based OCR implementation
  - `multiple_postprocessing/` - Country-specific text post-processing
- `number_plate_keypoints_detectors/` - Keypoint manipulation tools
- `number_plate_multiline_extractors/` - Handle multi-line plates

### Neural Network Models

**Models** ([nomeroff_net/nnmodels/](nomeroff_net/nnmodels/)) - PyTorch Lightning modules:
- `ocr_model.py` - `NPOcrNet` - ResNet + BiLSTM for text recognition
- `numberplate_options_model.py` - Plate region/type classifier
- `numberplate_orientation_model.py` - Rotation detection
- `numberplate_classification_model.py` - Generic classification base

All models use PyTorch Lightning 1.8.6 for training. They support:
- Automatic checkpointing
- Learning rate scheduling
- Gradient clipping
- Mixed precision training

### Data Modules and Loaders

**Data Modules** ([nomeroff_net/data_modules/](nomeroff_net/data_modules/)):
- `numberplate_ocr_data_module.py` - OCR training data pipeline
- `numberplate_options_data_module.py` - Classification data pipeline
- `numberplate_orientation_data_module.py` - Orientation data pipeline

**Image Loaders** ([nomeroff_net/image_loaders/](nomeroff_net/image_loaders/)):
- `opencv_loader.py` - OpenCV-based (default)
- `pillow_loader.py` - Pillow-based
- `turbo_loader.py` - TurboJPEG-based (fastest)
- `dumpy_loader.py` - NumPy arrays directly

Specify loader in pipeline:
```python
pipeline("number_plate_detection_and_reading", image_loader="turbo")
```

### Training Workflow

Training notebooks are in [tutorials/ju/train/](tutorials/ju/train/):
- `ocr/` - Train OCR models for different countries
- `classification/` - Train region/options classifiers

Models are trained using PyTorch Lightning with data modules. The framework uses:
- **Detection**: Pre-trained YOLOv8 models (via Ultralytics), fine-tuned on license plate datasets
- **OCR**: Custom RNN architecture trained from scratch on country-specific datasets
- **Classification**: ResNet-based classifiers for region/orientation

### TensorRT Support

For production inference, models can be converted to TensorRT:
```bash
python tutorials/py/model_convertors/convert_ocr_to_tensorrt.py
python tutorials/py/model_convertors/convert_numberplate_options_to_tensorrt.py
python tutorials/py/model_convertors/convert_ultralytics_to_tensorrt.py
```

Use TensorRT pipelines with `_trt` suffix:
- `number_plate_detection_and_reading_trt`
- Related pipeline and pipe files have `_trt` suffix

## Key Environment Variables

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0              # GPU selection
TF_FORCE_GPU_ALLOW_GROWTH=true
LRU_CACHE_CAPACITY=1                # Model cache size
PORT=8887                           # REST API port
```

## Important Paths and Files

- `nomeroff_net/__init__.py` - Main package entry point, exports `pipeline`, `Detector`, `TextDetector`, `OptionsDetector`
- `setup.py` - Package configuration, reads version from `nomeroff_net/__init__.py`
- `requirements.txt` - Split into packages and git repos (separated by `# git repos` comment)
- `data/` - Example images and datasets for testing
- `tutorials/py/rest_examples/` - FastAPI, Flask, and Tornado server examples
- `moderation/` - Separate Node.js application for image moderation

## Testing in CI

GitHub Actions workflows:
- `.github/workflows/nn-ci-cpu-testing.yml` - CPU tests on Python 3.9, 3.10, 3.11
- `.github/workflows/benchmark.yaml` - GPU benchmarks on self-hosted runner
- `.github/workflows/nn-ci-js-testing.yml` - Node.js moderation app tests

Tests run inference examples, benchmarks, Jupyter notebooks, and individual module tests.

## Documentation

Build documentation with MkDocs:
```bash
mkdocs serve  # Local development server
mkdocs build  # Generate static site
```

Documentation is hosted on ReadTheDocs (see `.readthedocs.yml`).

## Model Downloads

Models are automatically downloaded from RIA.com servers on first use via the `modelhub-client` package. They're cached locally. Models are stored at various URLs on `nomeroff.net.ua`.

## Country-Specific OCR Models

The framework includes pre-trained OCR models for:
- `eu_ua_2004_2015` - Ukraine (2004/2015 standards)
- `eu_ua_1995` - Ukraine (old design)
- `eu` - European Union
- `ru` - Russia (and occupied territories)
- `kz` - Kazakhstan
- `ge` - Georgia
- `by` - Belarus
- `su` - ex-USSR
- `kg` - Kyrgyzstan
- `am` - Armenia

When adding support for new countries, you need to:
1. Create and train a new OCR model
2. Add post-processing logic in `nomeroff_net/pipes/number_plate_text_readers/multiple_postprocessing/`
3. Update the region classifier
