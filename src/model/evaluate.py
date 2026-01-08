import torch
from sklearn.metrics import accuracy_score


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_all, y_all = [], []

    for imgs, y in loader:
        imgs = imgs.to(device)
        y = y.to(device)

        logits = model(imgs)
        loss = criterion(logits, y)
        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)
        preds_all.append(preds.detach().cpu())
        y_all.append(y.detach().cpu())

    epoch_loss = running_loss / len(loader.dataset)
    y_true = torch.cat(y_all).numpy()
    y_pred = torch.cat(preds_all).numpy()
    epoch_acc = accuracy_score(y_true, y_pred)
    return epoch_loss, epoch_acc, y_true, y_pred