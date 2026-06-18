"""Bidirectional LSTM model for frame-level pronunciation sequences."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMNetwork(nn.Module):
    """Classify variable-length MFCC sequences with a BiLSTM and attention."""

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
        self.attention = nn.Linear(output_size, 1)
        self.embedding = nn.Sequential(
            nn.Linear(output_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, parameter in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
                # Positive forget-gate bias helps retain early context.
                gate_size = parameter.shape[0] // 4
                parameter.data[gate_size:2 * gate_size].fill_(1.0)
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        nn.init.xavier_uniform_(self.embedding[0].weight)
        nn.init.zeros_(self.embedding[0].bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

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
            packed_output, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=x.shape[1]
            )
            valid = (
                torch.arange(x.shape[1], device=x.device)[None, :]
                < lengths[:, None]
            )
        else:
            output, _ = self.lstm(x)
            valid = torch.ones(
                x.shape[:2], dtype=torch.bool, device=x.device
            )

        attention_logits = self.attention(output).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~valid, -1e4)
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)
        embedding = self.embedding(pooled)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits
