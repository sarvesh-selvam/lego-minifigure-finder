import torch.nn as nn
from torchvision import models


def build_resnet18(
    num_classes: int = 2,
    pretrained: bool = True,
    fine_tune_all: bool = True,
):
    """ResNet18 with replaced classifier head for `num_classes`."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace final layer for 2 classes
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    if not fine_tune_all:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    return model