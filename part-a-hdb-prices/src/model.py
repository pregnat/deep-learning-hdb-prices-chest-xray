import torch
from torch import nn

from config import Config

ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}


class PriceModel(nn.Module):
    """Embeddings + one-hidden-layer MLP.

    With the default Config this is exactly the provided baseline. Optional settings:
    - use_categorical=False drops the embedding tables (A2a ablation).
    - activation="sigmoid" swaps the hidden ReLU for a Sigmoid (A2b ablation).
    - wide_lambda=λ adds a linear branch: output = MLP(x) + λ * Linear(x) (A3).
    """

    def __init__(
        self,
        cardinalities: list[int],
        target_mean: float,
        target_std: float,
        config: Config,
    ) -> None:
        super().__init__()
        self.use_categorical = config.use_categorical
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, width)
            for cardinality, width in zip(cardinalities, config.embedding_dims)
            if config.use_categorical
        )

        embedding_width = sum(config.embedding_dims) if config.use_categorical else 0
        input_width = embedding_width + len(config.continuous_features)
        self.mlp = nn.Sequential(
            nn.Linear(input_width, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            ACTIVATIONS[config.activation](),
            nn.Linear(config.hidden_width, 1),
        )

        self.wide_lambda = config.wide_lambda
        if self.wide_lambda is not None:
            self.linear = nn.Linear(input_width, 1)

        self.target_mean = target_mean
        self.target_std = target_std

    def forward(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        features = torch.cat(embedded + [continuous], dim=1)
        output = self.mlp(features).squeeze(1)
        if self.wide_lambda is not None:
            output = output + self.wide_lambda * self.linear(features).squeeze(1)
        return output

    def predict_price(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        return self(categorical, continuous) * self.target_std + self.target_mean
