"""CNN model definitions based on torch.nn.Module."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D CNN classifier for aggregated MFCC feature vectors."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_rate: float = 0.3,
    ) -> None:
        super(CNN1D, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Unflatten(1, (1, input_dim)),
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )

        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, input_dim)).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
