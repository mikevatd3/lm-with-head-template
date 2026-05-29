import torch
import torch.nn as nn


@torch.no_grad()
def evaluate(model, loader, num_classes=4):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    for embeds, labels in loader:
        logits = model(embeds)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        preds = logits.argmax(-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    n = len(labels)

    # Confusion matrix: cm[i, j] = count of (actual=i, predicted=j)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    indices = labels * num_classes + preds  # flat indices into cm
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    cm = counts.view(num_classes, num_classes)

    # Per-class accuracy = diagonal / row sum
    row_totals = cm.sum(dim=1)
    diag = cm.diag()
    per_class = {
        int(c): (diag[c].item() / row_totals[c].item())
        for c in range(num_classes)
        if row_totals[c] > 0
    }

    return {
        "loss": total_loss / n,
        "accuracy": (diag.sum().item() / n),
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "n": int(n),
    }


def score_against(model, loader):
    """
    A simpler version of the loop above but cannot generate the confusion matrix
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for embeds, labels in loader:
            preds = model(embeds).argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total
