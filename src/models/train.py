import os
import torch
from sklearn.metrics import accuracy_score

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