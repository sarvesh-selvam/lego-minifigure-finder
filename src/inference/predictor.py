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