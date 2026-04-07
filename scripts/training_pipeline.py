import argparse
from pathlib import Path

import pandas as pd
import torch
import mlflow
import mlflow.pytorch

from sklearn.metrics import classification_report, f1_score

from src.data.data_loader import make_loaders
from src.classifier.resnet import build_resnet18
from src.classifier.small_cnn import build_small_cnn
from src.model.train import fit
from src.model.evaluate import evaluate
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.config import load_config
from src.inference.bundle import save_bundle


def compute_class_weights(train_csv: str, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training CSV."""
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().sort_index().values.astype(float)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--run-name", default=None, help="Override run name")
    ap.add_argument("--alias", default="production", help="MLflow alias to assign (e.g. staging, production)")
    args = ap.parse_args()

    # --- Config ---
    cfg = load_config(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    print(f"Config     : {args.config}")
    print(f"Run name   : {cfg.run_name}")
    print(f"Arch       : {cfg.arch}  |  pretrained={cfg.pretrained}")
    print(f"Epochs     : {cfg.epochs}  |  lr={cfg.lr}  |  batch={cfg.batch_size}")
    print(f"Freeze     : {cfg.freeze_epochs} epoch(s)")

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

    # --- Class weights ---
    train_csv = str(Path(cfg.data_dir) / "train.csv")
    class_weights = compute_class_weights(train_csv, cfg.num_classes)
    print(f"  class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}"
          f"  (not_minifig, minifig)")

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

    # --- MLflow run ---
    mlflow.set_tracking_uri("mlflow")
    mlflow.set_experiment("lego-minifigure-finder")

    with mlflow.start_run(run_name=cfg.run_name):

        mlflow.log_params({
            "arch": arch,
            "pretrained": cfg.pretrained,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "image_size": cfg.image_size,
            "seed": cfg.seed,
            "num_classes": cfg.num_classes,
            "freeze_epochs": cfg.freeze_epochs,
            "class_weight_0": round(class_weights[0].item(), 4),
            "class_weight_1": round(class_weights[1].item(), 4),
            "train_samples": len(loaders["train"].dataset),
            "val_samples": len(loaders["val"].dataset),
            "test_samples": len(loaders["test"].dataset),
            "trainable_params": n_params,
        })

        # --- Training ---
        print(f"\nTraining for {cfg.epochs} epochs...\n")
        model, history = fit(
            model, loaders,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            device=device,
            class_weights=class_weights,
            freeze_epochs=cfg.freeze_epochs,
        )

        # Log per-epoch metrics
        for entry in history:
            mlflow.log_metrics({
                "train_loss": entry["train_loss"],
                "train_acc":  entry["train_acc"],
                "val_loss":   entry["val_loss"],
                "val_acc":    entry["val_acc"],
                "val_f1":     entry["val_f1"],
            }, step=entry["epoch"])

        # --- Evaluation ---
        print("\nEvaluating on test set...")
        crit = torch.nn.CrossEntropyLoss()
        test_loss, test_acc, y_true, y_pred = evaluate(
            model, loaders["test"], crit, device=torch.device(device),
        )
        test_f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        print(f"  loss={test_loss:.4f}  acc={test_acc:.4f}  f1={test_f1:.4f}")
        print(classification_report(y_true, y_pred, target_names=["not_minifig", "minifig"], digits=4))

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_acc":  test_acc,
            "test_f1":   test_f1,
        })

        # --- Save bundle ---
        run_dir = Path(cfg.out_dir) / cfg.run_name
        bundle_dir = run_dir / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        class_names = ["not_minifig", "minifig"]
        save_bundle(bundle_dir, model=model, arch=arch, class_names=class_names, image_size=cfg.image_size)
        print(f"Bundle saved to {bundle_dir}")

        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")

        # Log the model with mlflow.pytorch so it can be registered
        mlflow.pytorch.log_model(model, name="pytorch_model")

        # --- Register model ---
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/pytorch_model"
        registered = mlflow.register_model(model_uri, arch)
        print(f"Model registered as '{arch}' version {registered.version}")

        # --- Conditionally promote alias ---
        client = mlflow.MlflowClient()
        should_promote = True

        try:
            current = client.get_model_version_by_alias(arch, args.alias)
            current_run = client.get_run(current.run_id)
            current_f1 = current_run.data.metrics.get("test_f1", 0.0)
            if test_f1 > current_f1:
                print(f"New test_f1={test_f1:.4f} > current @{args.alias} test_f1={current_f1:.4f} — promoting")
            else:
                print(f"New test_f1={test_f1:.4f} <= current @{args.alias} test_f1={current_f1:.4f} — keeping existing version")
                should_promote = False
        except Exception:
            # No existing alias — first time, always promote
            print(f"No existing @{args.alias} found — promoting version {registered.version}")

        if should_promote:
            client.set_registered_model_alias(
                name=arch,
                alias=args.alias,
                version=registered.version,
            )
            print(f"'{arch}' version {registered.version} promoted to @{args.alias}")

        print(f"MLflow run complete — view with: mlflow ui --backend-store-uri mlflow/")


if __name__ == "__main__":
    main()
