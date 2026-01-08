import os
import time
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score

from models.evaluate import evaluate

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    preds_all, y_all = [], []

    for imgs, y in loader:
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

    epoch_loss = running_loss / len(loader.dataset)
    y_true = torch.cat(y_all).numpy()
    y_pred = torch.cat(preds_all).numpy()
    epoch_acc = accuracy_score(y_true, y_pred)
    return epoch_loss, epoch_acc



def fit(model, loaders, epochs, lr=1e-3, weight_decay=0.0):
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1
    best_state = None
    history = []

    for epoch in range(1, epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        dt = time.time() - t0

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "sec": dt})
        print(f"[{epoch:02d}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} | {dt:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            # # NEW (Phase 4.1) — quick checkpoint
            # os.makedirs(EXPORT_DIR, exist_ok=True)
            # torch.save({
            #     "epoch": epoch,
            #     "arch": model.__class__.__name__,
            #     "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            #     "val_loss": va_loss,
            #     "val_acc": va_acc,
            # }, f"{EXPORT_DIR}/best_ckpt.pt")
        

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history