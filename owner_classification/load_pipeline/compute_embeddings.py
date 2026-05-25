import torch
import torch.nn as nn

from tqdm import tqdm


@torch.no_grad()
def compute_embeddings(
    texts: list[str],
    tokenizer,
    encoder: nn.Module,
    device: str = "cpu",
    batch_size: int = 64,
    max_length: int = 128,
)-> torch.Tensor: 
    encoder.eval().to(device)
    vectors = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Embedding names",
        unit="batch"
    ):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = encoder(**enc)

        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        vectors.append(pooled.cpu())

    return torch.cat(vectors, dim=0)



def check_for_cached_embeddings():
    pass
