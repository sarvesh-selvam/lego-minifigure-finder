import time
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.model.evaluate import evaluate


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    preds_all, y_all = [], []

    bar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs} [train]", leave=False, unit="batch")
    for imgs, y in bar:
        imgs = imgs.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        preds_all.append(preds.detach().cpu())
        y_all.append(y.detach().cpu())

        bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    y_true = torch.cat(y_all).numpy()
    y_pred = torch.cat(preds_all).numpy()
    epoch_acc = accuracy_score(y_true, y_pred)
    return epoch_loss, epoch_acc


def fit(model, loaders, epochs, lr=1e-3, weight_decay=0.0, device="cpu"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device, epoch, epochs)
        va_loss, va_acc, y_true, y_pred = evaluate(model, loaders["val"], criterion, device)
        va_f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        scheduler.step()
        dt = time.time() - t0

        is_best = va_f1 > best_val_f1
        marker = " *" if is_best else ""
        print(f"[{epoch:02d}/{epochs}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | "
              f"{dt:.1f}s{marker}")

        if is_best:
            best_val_f1 = va_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,
            "sec": dt,
        })

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nBest model restored (val_f1={best_val_f1:.4f})")

    return model, history
