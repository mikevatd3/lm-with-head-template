from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from owner_classification.load_pipeline.compute_embeddings import compute_embeddings
from owner_classification.network import OwnerTypeModel


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_DIR = Path(__file__).parent / "trained_models"


def find_latest_model(model_dir: Path = MODEL_DIR) -> Path:
    candidates = sorted(model_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No .pth files found in {model_dir}")
    return candidates[-1]


def load_model(model_path: Path) -> OwnerTypeModel:
    state = torch.load(model_path, weights_only=True)
    encoder_dim = state["head.0.weight"].shape[1]
    hidden_size = state["head.0.weight"].shape[0]
    n_classes = state["head.3.weight"].shape[0]
    model = OwnerTypeModel(encoder_dim, hidden_size=hidden_size, n_classes=n_classes)
    model.load_state_dict(state)
    return model


def predict(
    names: list[str],
    model_path: Path | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[int]:
    """Return predicted class indices for a list of owner names."""
    if model_path is None:
        model_path = find_latest_model()

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    encoder = AutoModel.from_pretrained(embedding_model)

    embeddings = compute_embeddings(names, tokenizer, encoder)

    classifier = load_model(model_path)
    classifier.eval()
    with torch.no_grad():
        return classifier(embeddings).argmax(-1).tolist()
