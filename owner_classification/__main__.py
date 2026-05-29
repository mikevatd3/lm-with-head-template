from pathlib import Path

import click

from owner_classification.predict import (
    DEFAULT_EMBEDDING_MODEL,
    MODEL_DIR,
    find_latest_model,
    predict as run_predict,
)
from owner_classification.train import train as run_train


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
        model_path = find_latest_model()
        click.echo(f"Using model: {model_path}")

    preds = run_predict(all_names, model_path=model_path, embedding_model=embedding_model)

    for name, pred in zip(all_names, preds):
        click.echo(f"{name}\t{pred}")


if __name__ == "__main__":
    main()
