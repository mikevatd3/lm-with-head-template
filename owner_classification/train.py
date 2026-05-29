from datetime import datetime

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .load_pipeline import load_dataset
from .network import OwnerTypeModel
from .evaluate import evaluate


def train(epochs, batch_size, lr, embedding_model, model_dir):
    print("Starting training run for the parcel owner classifier.")
    ds = load_dataset(model=embedding_model)

    train_set, val_set, test_set = random_split(
        ds,
        [0.80, 0.10, 0.10],
        generator=torch.Generator("cpu").manual_seed(67),
    )

    encoder_dim = train_set[0][0].shape[-1]
    classifier = OwnerTypeModel(encoder_dim, hidden_size=256)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2)

    pbar = tqdm(range(epochs), desc="Training classifier", unit="epoch")
    for _ in pbar:
        classifier.train()
        for embeds, labels in train_loader:
            logits = classifier(embeds)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for embeds, labels in val_loader:
                preds = classifier(embeds).argmax(-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        pbar.set_postfix(val_acc=f"{correct / total:.4f}")

    print(evaluate(classifier, test_loader))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = model_dir / f"model_{stamp}.pth"
    print(f"Saving model to {path}")
    torch.save(classifier.state_dict(), path)
