import torch
import torch.nn as nn


class TabTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.ff_dim = ff_dim

        self.feature_embedding = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
            activation="relu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        tokens = x * self.feature_embedding.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = self.transformer(tokens)
        return self.classifier(tokens)
