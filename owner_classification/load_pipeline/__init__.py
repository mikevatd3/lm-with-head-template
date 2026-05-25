"""
Nothing lazy about this right now -- it's a read everything into two tensors 
approach. Trying to keep it simple. We cache the embeddings because its the 
slowest part of the training process.
"""
from pathlib import Path
import hashlib
from dotenv import load_dotenv

import torch
from torch.utils.data import TensorDataset
from transformers import AutoModel, AutoTokenizer

from .open_datasets import open_datasets
from .compute_embeddings import compute_embeddings

load_dotenv()
DEFAULT_DEVICE = "cpu"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "embeddings"
MAX_LENGTH = 128


def load_dataset(
    device=DEFAULT_DEVICE,
    model=DEFAULT_MODEL,
    cache_dir=CACHE_DIR,
    max_length=MAX_LENGTH
):
    examples, labels = open_datasets()

    # Check for embeddings already in the cache
    key_input = f"{model}|{max_length}|{'|'.join(examples)}"
    key = hashlib.md5(key_input.encode()).hexdigest()[:16]
    cache_path = cache_dir / f"{key}.pt"

    if cache_path.exists():
        print("Loading cached embeddings.")
        embeddings = torch.load(cache_path)

        return TensorDataset(embeddings, torch.tensor(labels, dtype=torch.long))

    tokenizer = AutoTokenizer.from_pretrained(model)
    encoder = AutoModel.from_pretrained(model)

    embeddings = compute_embeddings(
        examples,
        tokenizer,
        encoder,
        device=device,
        max_length=max_length
    )

    torch.save(embeddings, cache_path)
    return TensorDataset(embeddings, torch.tensor(labels, dtype=torch.long))


