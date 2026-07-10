"""Standard PyTorch LSTM classifier for pronunciation sequences."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMNetwork(nn.Module):
    """Classify variable-length MFCC sequences with ``torch.nn.LSTM``."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = True,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim

        lstm_dropout = dropout_rate if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.embedding = nn.Sequential(
            nn.Linear(output_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_embedding: bool = False,
    ):
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(x)

        directions = 2 if self.bidirectional else 1
        hidden = hidden.view(
            self.num_layers, directions, x.shape[0], self.hidden_size
        )
        last_layer = hidden[-1]
        if self.bidirectional:
            sequence_summary = torch.cat(
                (last_layer[0], last_layer[1]), dim=1
            )
        else:
            sequence_summary = last_layer[0]

        embedding = self.embedding(sequence_summary)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits
