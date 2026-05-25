from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).parent.parent.parent


def open_datasets(
    data_location: Path = BASE_DIR / "data",
    example_col="example",
    label_col="category",
):
    csvs = sorted(data_location.glob("*.csv"))

    if not csvs:
        raise FileNotFoundError("No data to train on!")

    combined = pd.concat([pd.read_csv(csv) for csv in csvs])
    combined = combined.dropna(subset=[example_col, label_col])

    examples = combined["example"].to_list()
    labels = combined["category"].to_list()

    return examples, labels

