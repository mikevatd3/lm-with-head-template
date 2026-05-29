from pathlib import Path

import click
import torch
from transformers import AutoModel, AutoTokenizer

from owner_classification.load_pipeline.compute_embeddings import compute_embeddings
from owner_classification.network import OwnerTypeModel
from owner_classification.train import train as run_train


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_DIR = Path(__file__).parent / "trained_models"


def _latest_model(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No .pth files found in {model_dir}")
    return candidates[-1]


def _load_model(model_path: Path) -> OwnerTypeModel:
    state = torch.load(model_path, weights_only=True)
    encoder_dim = state["head.0.weight"].shape[1]
    hidden_size = state["head.0.weight"].shape[0]
    n_classes = state["head.3.weight"].shape[0]
    model = OwnerTypeModel(encoder_dim, hidden_size=hidden_size, n_classes=n_classes)
    model.load_state_dict(state)
    return model


@click.group()
def main():
    pass


@main.command()
@click.option("--epochs", default=20, show_default=True, help="Number of training epochs.")
@click.option("--batch-size", default=256, show_default=True, help="Training batch size.")
@click.option("--lr", default=5e-4, show_default=True, help="AdamW learning rate.")
@click.option(
    "--embedding-model",
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help="HuggingFace sentence-transformer model for embeddings.",
)
@click.option(
    "--model-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory to save the trained model. Defaults to trained_models/.",
)
def train(epochs, batch_size, lr, embedding_model, model_dir):
    """Train the owner-type classifier head."""
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_train(epochs, batch_size, lr, embedding_model, model_dir)


@main.command()
@click.argument("names", nargs=-1)
@click.option(
    "--file",
    "input_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="CSV (with 'example' column) or plain text file (one name per line).",
)
@click.option(
    "--model",
    "model_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a .pth model file. Defaults to the most recently saved model.",
)
@click.option(
    "--embedding-model",
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help="HuggingFace sentence-transformer model for embeddings.",
)
def predict(names, input_file, model_path, embedding_model):
    """Predict owner type for one or more parcel owner names."""
    all_names = list(names)

    if input_file is not None:
        if input_file.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(input_file)
            if "example" not in df.columns:
                raise click.UsageError(f"CSV must have an 'example' column; found: {list(df.columns)}")
            all_names.extend(df["example"].dropna().tolist())
        else:
            all_names.extend(
                line.strip() for line in input_file.read_text().splitlines() if line.strip()
            )

    if not all_names:
        raise click.UsageError("Provide at least one name as an argument or via --file.")

    if model_path is None:
        model_path = _latest_model(MODEL_DIR)
        click.echo(f"Using model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    encoder = AutoModel.from_pretrained(embedding_model)

    embeddings = compute_embeddings(all_names, tokenizer, encoder)

    classifier = _load_model(model_path)
    classifier.eval()
    with torch.no_grad():
        preds = classifier(embeddings).argmax(-1).tolist()

    for name, pred in zip(all_names, preds):
        click.echo(f"{name}\t{pred}")


if __name__ == "__main__":
    main()
