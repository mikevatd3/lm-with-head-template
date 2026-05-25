from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split

from owner_classification.load_pipeline import load_dataset
from owner_classification.network import OwnerTypeModel
from owner_classification.evaluate import evaluate


LEARNING_RATE = 1e-3
EPOCHS = 20
N_BATCH_SIZE = 256
MODEL_DIR = Path(__file__).parent / "owner_classification" / "trained_models"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def main():
    print("Starting training run for the parcel owner classifier.")
    ds = load_dataset(model=EMBEDDING_MODEL)
    
    train_set, val_set, test_set = random_split(
        ds,
        [0.80, 0.10, 0.10],
        generator=torch.Generator("cpu").manual_seed(67)
    )

    encoder_dim = train_set[0][0].shape[-1]

    model = OwnerTypeModel(encoder_dim, hidden_size=256)


    # Train the head
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=N_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=N_BATCH_SIZE * 2)
    test_loader = DataLoader(test_set, batch_size=N_BATCH_SIZE * 2)

    pbar = tqdm(range(EPOCHS), desc="Trainging classifier", unit="epoch")
    for _ in pbar:
        model.train()
        for embeds, labels in train_loader:
            logits = model(embeds)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for embeds, labels in val_loader:
                preds = model(embeds).argmax(-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        pbar.set_postfix(val_acc=f"{correct / total:.4f}")

    print(evaluate(model, test_loader))

    training_time = datetime.now() 
    stamp = training_time.strftime("%Y%m%d_%H%M%S")
    path = f"model_{stamp}.pth"
    print(f"Saving model to {path}")
    torch.save(model.state_dict(), MODEL_DIR / path)


if __name__ == "__main__":
    main()
