# TODOs

## Class label names are undefined
The 4 integer labels (0–3) have no string-name mapping anywhere in the code. From inspecting the CSVs:
- 0 → individual (e.g. "LAZENBY, WILLIAM")
- 1 → LLC / corporate (e.g. "HELEN STREET TOWNHOMES, LLC")
- 2 → non-profit / religious (e.g. "AL-HUDA ISLAMIC ASSOCIATION")
- 3 → trust (e.g. "ROBERT BIBB TRUST")

Add a `CLASS_NAMES = {0: "individual", 1: "corporate", 2: "nonprofit", 3: "trust"}` dict (probably in `network.py` or a new `constants.py`) and wire it into `predict` output and `evaluate` reporting.

## `train.py` imports a non-existent module
`owner_classification/train.py` imports `from .dataset import OwnerNames, Collator` — that module doesn't exist. Either delete `train.py` or port the working training loop from `main.py` into it and clean up the stale imports.

## Model save format doesn't include architecture metadata
The `.pth` files only store `state_dict`. The `predict` command infers `encoder_dim`, `hidden_size`, and `n_classes` from weight shapes — which works, but is fragile. Consider saving a metadata dict alongside the weights:
```python
torch.save({"state_dict": model.state_dict(), "encoder_dim": encoder_dim, "hidden_size": 256, "n_classes": 4, "embedding_model": embedding_model}, path)
```
This also lets you record which embedding model the head was trained with, preventing mismatch at inference time.

## `load_dataset` label encoding assumes integer CSVs
`open_datasets()` returns raw strings from the `category` column, and `load_dataset()` calls `torch.tensor(labels, dtype=torch.long)` without any factorization step. This works today because the CSVs store integer labels, but will silently break if any CSV ever uses string labels. Add explicit `pd.factorize()` (or a label encoder) in `open_datasets()` and surface the label-to-index mapping.
