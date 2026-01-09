import argparse
from pathlib import Path
# import random
# import numpy as np
import torch

from sklearn.metrics import classification_report

from src.utils.config import load_config

# from .data import make_loaders
# from .models import SmallCNN, build_resnet18
# from .train_utils import fit, evaluate
# from .inference import save_bundle
# from .s3_utils import download_s3_prefix, upload_dir_to_s3


from src.data.data_loader import make_loaders
from src.classifier.resnet import build_resnet18
from src.classifier.small_cnn import build_small_cnn
from src.model.train import fit
from src.model.evaluate import evaluate

from src.utils.seed import set_seed
from src.utils.device import get_device



# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def pick_device():
#     if torch.cuda.is_available():
#         return "cuda"
#     if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
#         return "mps"
#     return "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--run-name", default=None, help="Override run name")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg.run_name = args.run_name

    set_seed(cfg.seed)
    device = get_device()

    # # Optional: pull dataset from S3 to local disk first
    # if cfg.s3_dataset_uri:
    #     # downloads into cfg.data_dir
    #     download_s3_prefix(cfg.s3_dataset_uri, cfg.data_dir)

    loaders = make_loaders(
        data_dir=cfg.data_dir,
        images_subdir=cfg.images_subdir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
    )

    if cfg.arch.lower() in {"smallcnn", "small_cnn"}:
        model = build_small_cnn(num_classes=cfg.num_classes)
        arch = "smallcnn"
    else:
        model = build_resnet18(num_classes=cfg.num_classes, pretrained=cfg.pretrained)
        arch = "resnet18"

    model, history = fit(model, loaders, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay, device=device)

    crit = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate(model, loaders["test"], crit, device=torch.device(device))
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    run_dir = Path(cfg.out_dir) / cfg.run_name
    bundle_dir = run_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Convention: class index 1 is "minifig"
    class_names = ["not_minifig", "minifig"]
    save_bundle(bundle_dir, model=model, arch=arch, class_names=class_names, image_size=cfg.image_size)

    # # Optional: push bundle to S3
    # if cfg.s3_model_uri:
    #     upload_dir_to_s3(bundle_dir, cfg.s3_model_uri)

if __name__ == "__main__":
    main()
