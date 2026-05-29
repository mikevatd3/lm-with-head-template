from pathlib import Path
<<<<<<< HEAD
import json
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split

from owner_classification.load_pipeline import load_dataset
from owner_classification.network import OwnerTypeModel
from owner_classification.evaluate import evaluate, score_against

from owner_classification.train import train


LEARNING_RATE = 5e-4
EPOCHS = 20
N_BATCH_SIZE = 256
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

MODEL_DIR = Path(__file__).parent / "owner_classification" / "trained_models"
REPORT_DIR = Path(__file__).parent / "training_reports"


def main():
    train(EPOCHS, N_BATCH_SIZE, LEARNING_RATE, EMBEDDING_MODEL, MODEL_DIR)


if __name__ == "__main__":
    main()
