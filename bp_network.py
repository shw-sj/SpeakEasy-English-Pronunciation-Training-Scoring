"""BP network model definitions based on torch.nn.Module.

Enhanced version with BatchNorm and optional residual connections.
BPNetwork operates on aggregated (78-dim or 156-dim) feature vectors
for letter-level pronunciation recognition.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════
#  ResidualBlock
# ═══════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Residual block with two Linear-BN-ReLU layers and a skip connection.

    Dropout is applied inside the residual branch (before the addition)
    rather than after, following the standard practice used in ResNet
    variants: the skip path preserves the original signal while the
    residual branch learns a regularised transformation.  This is
    analogous to how transformer residual streams use dropout in the
    sub-layer outputs before the residual add.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = (in_features == out_features)

        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(out_features, out_features))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)

        if not self.use_residual:
            self.projection = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features) if use_batchnorm else nn.Identity(),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.projection(x)


# ═══════════════════════════════════════════════════════════════════════
#  BPNetwork  (aggregated-feature MLP)
# ═══════════════════════════════════════════════════════════════════════

class BPNetwork(nn.Module):
    """BP/MLP classifier for letter pronunciation recognition.

    Enhanced with BatchNorm and optional residual connections.
    Operates on **aggregated** feature vectors (78-dim standard, 156-dim rich).

    Parameters
    ----------
    input_size : int
        Input feature dimension (78 standard, 156 rich).
    output_size : int
        Number of output classes.
    dropout_rate : float
        Dropout probability applied after each hidden layer.
    use_batchnorm : bool
        Whether to include BatchNorm1d layers (default True).
    use_residual : bool
        Whether to use residual connections in the shared base (default True).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout_rate: float = 0.3,
        use_batchnorm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super(BPNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual

        # ── Input feature dropout: light regularisation on raw features ──
        self.input_dropout = nn.Dropout(0.1)

        # ── Shared base: input → 256 → 128 ──
        if use_residual:
            self.base = nn.Sequential(
                ResidualBlock(input_size, 256, dropout_rate, use_batchnorm),
                ResidualBlock(256, 128, dropout_rate, use_batchnorm),
            )
            base_out_dim = 128
        else:
            base_layers: list[nn.Module] = [
                nn.Linear(input_size, 256),
            ]
            if use_batchnorm:
                base_layers.append(nn.BatchNorm1d(256))
            base_layers.extend([nn.ReLU(), nn.Dropout(dropout_rate)])
            base_layers.append(nn.Linear(256, 128))
            if use_batchnorm:
                base_layers.append(nn.BatchNorm1d(128))
            base_layers.extend([nn.ReLU(), nn.Dropout(dropout_rate)])
            self.base = nn.Sequential(*base_layers)
            base_out_dim = 128

        # ── Classification head: 128 → 256 → 128 → output ──
        head_layers: list[nn.Module] = [
            nn.Linear(base_out_dim, 256),
        ]
        if use_batchnorm:
            head_layers.append(nn.BatchNorm1d(256))
        head_layers.extend([nn.ReLU(), nn.Dropout(dropout_rate)])
        head_layers.append(nn.Linear(256, 128))
        if use_batchnorm:
            head_layers.append(nn.BatchNorm1d(128))
        head_layers.extend([nn.ReLU(), nn.Dropout(dropout_rate)])
        head_layers.append(nn.Linear(128, output_size))
        self.task_specific = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """Forward pass.

        Parameters
        ----------
        x : (B, input_size)
        return_embedding : bool
            If True, also returns the 128-dim penultimate embedding
            from the second-to-last layer, usable for pronunciation
            quality assessment via embedding similarity.

        Returns
        -------
        logits : (B, output_size)
        embedding : (B, 128), only if ``return_embedding=True``
        """
        x = self.input_dropout(x)
        x = self.base(x)  # (B, 128) shared representation
        # Walk task_specific until we reach the final Linear layer
        # task_specific is always built as:
        #   Linear(128→256) → [BN] → ReLU → Dropout → Linear(256→128) → [BN] → ReLU → Dropout → Linear(128→output)
        # We intercept after the second-to-last Linear (the one that outputs 128).
        h = x
        embedding = h  # fallback: use base output if task_specific is empty
        for i, layer in enumerate(self.task_specific):
            h = layer(h)
            # Intercept after the 128-dim Linear layer (second-to-last Linear)
            if isinstance(layer, nn.Linear) and layer.out_features == 128:
                embedding = h
        logits = h
        if return_embedding:
            return logits, embedding
        return logits

# ═══════════════════════════════════════════════════════════════════════
#  Convenience alias
# ═══════════════════════════════════════════════════════════════════════

MLPClassifier = BPNetwork
