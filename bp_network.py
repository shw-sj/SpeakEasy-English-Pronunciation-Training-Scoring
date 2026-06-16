"""BP network model definitions based on torch.nn.Module."""

from __future__ import annotations

import torch
import torch.nn as nn


class BPNetwork(nn.Module):
    """BP/MLP classifier with shared base layers and task-specific head."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        task: str = "letters",
        dropout_rate: float = 0.2,
    ) -> None:
        super(BPNetwork, self).__init__()
        self.task = task

        self.base = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        if task == "letters":
            self.task_specific = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, output_size),
            )
        elif task == "words":
            self.task_specific = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, output_size),
            )
        elif task == "both":
            self.task_specific = nn.Sequential(
                nn.Linear(256, 384),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(384, 192),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(192, output_size),
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        x = self.task_specific(x)
        return x

