import argparse
from pathlib import Path
import torch

from sklearn.metrics import classification_report

from src.data.data_loader import make_loaders
from src.classifier.resnet import build_resnet18
from src.classifier.small_cnn import build_small_cnn
from src.model.train import fit
from src.model.evaluate import evaluate
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.config import load_config
from src.inference.bundle import save_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--run-name", default=None, help="Override run name")
    args = ap.parse_args()

    # --- Config ---
    cfg = load_config(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    print(f"Config     : {args.config}")
    print(f"Run name   : {cfg.run_name}")
    print(f"Arch       : {cfg.arch}  |  pretrained={cfg.pretrained}")
    print(f"Epochs     : {cfg.epochs}  |  lr={cfg.lr}  |  batch={cfg.batch_size}")

    set_seed(cfg.seed)
    device = get_device()
    print(f"Device     : {device}")

    # --- Data ---
    print("\nLoading data...")
    loaders = make_loaders(
        data_dir=cfg.data_dir,
        images_subdir=cfg.images_subdir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
    )
    print(f"  train={len(loaders['train'].dataset)} "
          f"val={len(loaders['val'].dataset)} "
          f"test={len(loaders['test'].dataset)} samples")

    # --- Model ---
    print("\nBuilding model...")
    if cfg.arch.lower() in {"smallcnn", "small_cnn"}:
        model = build_small_cnn(num_classes=cfg.num_classes)
        arch = "smallcnn"
    else:
        model = build_resnet18(num_classes=cfg.num_classes, pretrained=cfg.pretrained)
        arch = "resnet18"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {arch}  |  {n_params:,} trainable parameters")

    # --- Training ---
    print(f"\nTraining for {cfg.epochs} epochs...\n")
    model, _ = fit(
        model, loaders,
        epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay, device=device,
    )

    # --- Evaluation ---
    print("\nEvaluating on test set...")
    crit = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate(model, loaders["test"], crit, device=torch.device(device))
    print(f"  loss={test_loss:.4f}  acc={test_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["not_minifig", "minifig"], digits=4))

    # --- Save bundle ---
    run_dir = Path(cfg.out_dir) / cfg.run_name
    bundle_dir = run_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["not_minifig", "minifig"]
    save_bundle(bundle_dir, model=model, arch=arch, class_names=class_names, image_size=cfg.image_size)
    print(f"Bundle saved to {bundle_dir}")


if __name__ == "__main__":
    main()
