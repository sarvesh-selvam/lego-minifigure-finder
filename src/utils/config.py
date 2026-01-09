from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data"
    images_subdir: str = "images"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0

    # Model / train
    arch: str = "resnet18"          # "smallcnn" or "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42

    # Outputs
    out_dir: str = "artifacts"
    run_name: str = "run"

    # Optional: S3
    s3_model_uri: Optional[str] = None   # e.g. s3://bucket/path/to/models/run123/
    s3_dataset_uri: Optional[str] = None # e.g. s3://bucket/path/to/datasets/v1/

def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig(**data)
