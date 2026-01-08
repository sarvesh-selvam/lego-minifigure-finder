from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Prefer CUDA, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
