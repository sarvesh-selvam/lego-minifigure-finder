# lego-minifigure-finder

A binary image classifier that detects whether an image contains a LEGO minifigure. Built as an MLOps project using PyTorch, with support for experiment tracking via MLflow and a pluggable inference bundle for serving.

## Project Structure

```
lego-minifigure-finder/
├── config/
│   └── train.yaml              # Training configuration
├── data/
│   ├── images/                 # Raw images (fig{id}_{shot}.jpg)
│   ├── train.csv               # Train split (filename, label)
│   ├── val.csv                 # Validation split
│   └── test.csv                # Test split
├── notebooks/
│   ├── data_processing.ipynb   # Data exploration and split creation
│   └── model_training.ipynb    # Interactive training experiments
├── scripts/
│   ├── training_pipeline.py    # CLI training entrypoint
│   └── inference_pipeline.py   # CLI inference entrypoint
├── src/
│   ├── classifier/             # Model definitions (ResNet18, SmallCNN)
│   ├── data/                   # Dataset, data loaders, transforms
│   ├── inference/              # Bundle save/load and Predictor
│   ├── model/                  # Training loop and evaluation
│   ├── utils/                  # Config, device, seed helpers
│   └── app/                    # FastAPI serving app (WIP)
├── mlflow/                     # MLflow experiment tracking runs
├── artifacts/                  # Saved model bundles (gitignored)
└── requirements.txt
```

## Setup

```bash
conda create -n lego-mlops python=3.11
conda activate lego-mlops
pip install -r requirements.txt
```

## Data Format

Images live in `data/images/` named `fig{id}_{shot}.jpg`. The CSV splits (`train.csv`, `val.csv`, `test.csv`) each have two columns:

| Column | Description |
|--------|-------------|
| `filename` | Image filename (e.g. `fig001_001.jpg`) |
| `label` | `1` = minifigure, `0` = not minifigure |

## Training

Edit `config/train.yaml` to configure the run, then:

```bash
python -m scripts.training_pipeline --config config/train.yaml
```

Key config options:

| Option | Default | Description |
|--------|---------|-------------|
| `arch` | `resnet18` | Model architecture (`resnet18` or `smallcnn`) |
| `pretrained` | `true` | Use ImageNet weights (ResNet18 only) |
| `epochs` | `10` | Number of training epochs |
| `lr` | `0.0003` | Learning rate (cosine annealed) |
| `batch_size` | `32` | Batch size |
| `run_name` | `test_v1` | Output directory name under `artifacts/` |

The training loop saves the checkpoint with the best validation F1 score. At the end of training, a model bundle is written to `artifacts/{run_name}/bundle/`.

## Inference

```bash
python -m scripts.inference_pipeline \
    --bundle-dir artifacts/test_v1/bundle \
    --image path/to/image.jpg
```

Output:

```
Prediction : minifig
Confidence : 0.9312
Is minifig : True
All probs  : [0.0688, 0.9312]
```

## Model Bundle

The bundle directory contains:

- `model.pt` — model weights and architecture name
- `bundle.json` — metadata (class names, image size, mean/std, threshold)

Load it in Python:

```python
from src.inference.bundle import load_bundle

predictor = load_bundle("artifacts/test_v1/bundle")
from PIL import Image
result = predictor.predict_pil(Image.open("image.jpg").convert("RGB"))
print(result)
```

## Experiment Tracking

MLflow is used for tracking runs. To view the UI:

```bash
mlflow ui --backend-store-uri mlflow/
```

Then open `http://localhost:5000` in your browser.

## Architectures

| Model | Parameters | Notes |
|-------|-----------|-------|
| `resnet18` | ~11M | Pretrained on ImageNet, fine-tuned end-to-end. Best accuracy. |
| `smallcnn` | ~100K | Lightweight 3-layer CNN. Faster to train, lower accuracy. |
