import torch.nn as nn


class OwnerTypeModel(nn.Module):
    def __init__(
        self,
        encoder_dim,
        hidden_size=256,
        n_classes=4,
    ) -> None:
        super(OwnerTypeModel, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, n_classes),
       )

    def forward(self, embeddings):
        return self.head(embeddings)
