"""
Currently incomplete -- look at main.py for the training loop. Eventually that 
will be brought over here and we'll build out a cli.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .network import OwnerTypeModel
from .dataset import OwnerNames, Collator


def train(
    model,
    training_data,
    n_epoch=60,
    n_batch_size=64,
    report_every=3,
    learning_rate=0.8,
    criterion=nn.CrossEntropyLoss(),
):
    loader = DataLoader(
        OwnerNames(),
        batch_size=n_batch_size,
        shuffle=True,
        collate_fn=Collator()
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    current_loss = 0
