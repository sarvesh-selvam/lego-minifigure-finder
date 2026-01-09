from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import torch
from PIL import Image

from .data import build_transforms, IMAGENET_MEAN, IMAGENET_STD
from .models import SmallCNN, build_resnet18

# def _device_default() -> torch.device:
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
#         return torch.device("mps")
#     return torch.device("cpu")


def save_bundle(
    bundle_dir: str | Path,
    model: torch.nn.Module,
    arch: str,
    class_names: List[str],
    image_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    threshold: float = 0.5,
):
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "arch": arch,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        bundle_dir / "model.pt",
    )

    meta = {
        "arch": arch,
        "class_names": class_names,
        "image_size": int(image_size),
        "mean": mean,
        "std": std,
        "threshold": float(threshold),
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _build_model_from_arch(arch: str, num_classes: int, pretrained: bool = False) -> torch.nn.Module:
    arch = arch.lower()
    if arch in {"smallcnn", "small_cnn"}:
        return SmallCNN(num_classes=num_classes)
    if arch in {"resnet18", "resnet"}:
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown arch: {arch}")

@dataclass
class Predictor:
    model: torch.nn.Module
    class_names: List[str]
    image_size: int
    threshold: float
    device: torch.device

    def predict_pil(self, img: Image.Image) -> Dict:
        self.model.eval()
        tf = build_transforms(self.image_size)["val"]
        x = tf(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().tolist()

        # for 2-class, treat index 1 as "positive" by convention
        pred_idx = int(torch.argmax(torch.tensor(probs)).item())
        return {
            "pred_idx": pred_idx,
            "pred_label": self.class_names[pred_idx],
            "probs": probs,
            "positive_prob": float(probs[1]) if len(probs) > 1 else float(probs[0]),
            "is_positive": (float(probs[1]) >= self.threshold) if len(probs) > 1 else (float(probs[0]) >= self.threshold),
        }


def load_bundle(bundle_dir: str | Path, device: Optional[str] = None) -> Predictor:
    bundle_dir = Path(bundle_dir)
    meta = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    ckpt = torch.load(bundle_dir / "model.pt", map_location="cpu")

    class_names = meta["class_names"]
    image_size = int(meta["image_size"])
    threshold = float(meta.get("threshold", 0.5))
    arch = meta["arch"]

    model = _build_model_from_arch(arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    dev = torch.device(device) if device else _device_default()
    model.to(dev)

    return Predictor(model=model, class_names=class_names, image_size=image_size, threshold=threshold, device=dev)
