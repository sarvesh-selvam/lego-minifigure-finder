# lego-minifigure-finder

A binary image classifier that detects whether an image contains a LEGO minifigure. Built as an MLOps project using PyTorch, with experiment tracking via MLflow, a REST API via FastAPI, and an interactive UI via Streamlit.

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
│   └── app/
│       ├── app.py              # FastAPI REST API
│       └── streamlit_app.py    # Streamlit interactive UI
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
| `epochs` | `15` | Number of training epochs |
| `lr` | `0.0003` | Learning rate |
| `batch_size` | `32` | Batch size |
| `run_name` | `test_v1` | Output directory name under `artifacts/` |

The training loop saves the checkpoint with the best validation F1 score and logs all metrics to MLflow. At the end of training, a model bundle is written to `artifacts/{run_name}/bundle/`.

## Inference

**CLI:**
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

**Python:**
```python
from src.inference.bundle import load_bundle
from PIL import Image

predictor = load_bundle("artifacts/test_v1/bundle")
result = predictor.predict_pil(Image.open("image.jpg").convert("RGB"))
print(result)
```

## Serving

The project includes two serving options that work together.

### FastAPI (REST API)

```bash
uvicorn src.app.app:app --reload
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check, confirms model is loaded |
| `POST` | `/predict` | Upload an image, returns JSON prediction |

Example:
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/image.jpg"
```

### Streamlit (Interactive UI)

In a second terminal:
```bash
streamlit run src/app/streamlit_app.py
```

Opens at `http://localhost:8501`. Upload an image and click **Run prediction** to see results. The sidebar shows live API status.

> Both servers must be running at the same time for the Streamlit UI to work.

## Experiment Tracking

MLflow tracks every training run automatically. To view the UI:

```bash
mlflow ui --backend-store-uri mlflow/
```

Then open `http://localhost:5000`. Each run logs:
- Hyperparameters (arch, lr, epochs, batch size, etc.)
- Per-epoch train/val loss, accuracy, and F1
- Final test loss, accuracy, and F1
- Model bundle as a downloadable artifact

## Model Bundle

The bundle directory contains:

- `model.pt` — model weights and architecture name
- `bundle.json` — metadata (class names, image size, mean/std, threshold)

## Architectures

| Model | Parameters | Notes |
|-------|-----------|-------|
| `resnet18` | ~11M | Pretrained on ImageNet, fine-tuned end-to-end. Best accuracy. |
| `smallcnn` | ~100K | Lightweight 3-layer CNN. Faster to train, lower accuracy. |
