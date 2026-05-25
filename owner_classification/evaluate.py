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
    
    # Overall
    acc = (preds == labels).float().mean().item()
    avg_loss = total_loss / n
    
    # Per-class accuracy
    per_class = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == labels[mask]).float().mean().item()
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "per_class_accuracy": per_class,
        "n": n,
        "preds": preds,
        "labels": labels,
    }

