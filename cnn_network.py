"""CNN model definitions based on torch.nn.Module.

Enhanced with BatchNorm, residual convolution blocks, SE attention,
and AdaptiveAvgPool for input-length-independent feature extraction.
Also provides CNN1DLegacy for backward-compatible checkpoint loading.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════
#  Building blocks
# ═══════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D signals.

    Learns per-channel scaling factors via a bottleneck MLP,
    letting the model emphasise informative feature maps.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → scale: (B, C, 1)
        scale = self.se(x).unsqueeze(-1)
        return x * scale


class ResidualConvBlock(nn.Module):
    """1D residual convolution block: Conv-BN-ReLU-Conv-BN + shortcut.

    When ``in_channels != out_channels``, the shortcut is a 1×1
    convolution so the addition dimensions match.  When ``stride`` > 1
    the shortcut uses the same stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        branch_layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        ]
        if dropout_rate > 0:
            branch_layers.append(nn.Dropout1d(dropout_rate))
        self.branch = nn.Sequential(*branch_layers)

        # Build shortcut
        need_projection = (in_channels != out_channels) or (stride != 1)
        if need_projection:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.branch(x) + self.shortcut(x))


# ═══════════════════════════════════════════════════════════════════════
#  CNN1D  (enhanced)
# ═══════════════════════════════════════════════════════════════════════

class CNN1D(nn.Module):
    """1D CNN classifier for aggregated MFCC feature vectors.

    Enhanced architecture (v2):
    - 4-stage residual convolution with expanding channels (32→64→128→256)
    - BatchNorm after every convolution for stable training
    - SE channel attention on the final feature map
    - AdaptiveAvgPool1d for input-length-independent output
    - 2-layer MLP classifier head with dropout

    Parameters
    ----------
    input_dim : int
        Input feature dimension (78 standard, 156 rich).
    num_classes : int
        Number of output classes.
    dropout_rate : float
        Dropout probability in classifier head and conv blocks.
    use_se : bool
        Whether to include SE attention (default True).
    use_residual : bool
        Whether to use residual convolution blocks (default True).
    channels : tuple[int, ...]
        Channel progression; defaults to (32, 64, 128, 256) for
        the standard config.  Pass a shorter tuple for a shallower
        network.
    freq_groups : int or None
        If provided, reshapes input (B, input_dim) → (B, input_dim//freq_groups, freq_groups)
        so that 1D convolution slides across *freq_groups* frequency-coefficient bins,
        which have meaningful acoustic ordering.  For 78-dim standard features,
        freq_groups=13 groups by mel bin (6 stat-channels × 13 bins).
        If None (default), uses the legacy reshape (B, 1, input_dim).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_rate: float = 0.3,
        use_se: bool = True,
        use_residual: bool = True,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        freq_groups: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_se = use_se
        self.use_residual = use_residual
        self.channels = channels
        self.freq_groups = freq_groups

        # ── Build feature extractor ──
        layers: list[nn.Module] = []

        # Reshape: organise features for meaningful convolution
        if freq_groups is not None and input_dim % freq_groups == 0:
            in_channels_first = input_dim // freq_groups
            layers.append(nn.Unflatten(1, (in_channels_first, freq_groups)))
        else:
            in_channels_first = 1
            layers.append(nn.Unflatten(1, (1, input_dim)))

        # Inter-block dropout — light regularisation
        inter_dropout = dropout_rate * 0.5

        # Track spatial dimension for dynamic stride (avoid collapsing to 1 too early)
        spatial_dim = freq_groups if freq_groups is not None else input_dim

        in_ch = in_channels_first
        for i, out_ch in enumerate(channels):
            # Dynamic stride: only downsample if spatial dim stays >= 2 afterward
            # (BatchNorm1d requires at least 2 spatial samples when batch=1)
            last_block = (i == len(channels) - 1)
            can_stride = (spatial_dim >= 4)  # stride=2 → spatial >= 2
            stride = 2 if (not last_block and can_stride) else 1

            if use_residual:
                layers.append(
                    ResidualConvBlock(in_ch, out_ch, kernel_size=3,
                                      stride=stride, dropout_rate=inter_dropout)
                )
            else:
                # Plain conv block (non-residual fallback)
                layers.append(
                    nn.Conv1d(in_ch, out_ch, kernel_size=3,
                              stride=stride, padding=1, bias=False)
                )
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                if inter_dropout > 0:
                    layers.append(nn.Dropout1d(inter_dropout))

            in_ch = out_ch
            # Update spatial dim for next block
            if stride == 2:
                spatial_dim = spatial_dim // 2

        # SE attention on the final feature map
        if use_se:
            layers.append(SEBlock(in_ch))

        # Adaptive pooling → fixed-size output independent of input_dim
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())

        self.features = nn.Sequential(*layers)

        # ── Determine flattened dimension (batch=2 avoids BN crash on single spatial) ──
        self.features.eval()
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(2, input_dim)).shape[1]
        self.features.train()

        # ── Classifier head ──
        # Use LayerNorm when freq_groups is set (handles batch=1 edge case
        # when spatial dim collapses to 1).  Keep BatchNorm1d for legacy
        # mode so old checkpoints remain loadable.
        if freq_groups is not None:
            norm_layer: type[nn.Module] = nn.LayerNorm
        else:
            norm_layer = nn.BatchNorm1d

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialisation for convolutions, Xavier for linears."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Legacy model  (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════

class CNN1DLegacy(nn.Module):
    """Original v1 CNN (kept for loading old checkpoints).

    This is the pre-optimisation architecture — two plain Conv1d layers
    followed by a two-layer MLP head, without BatchNorm or residuals.
    New training should use :class:`CNN1D` instead.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
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


# ═══════════════════════════════════════════════════════════════════════
#  Convenience aliases
# ═══════════════════════════════════════════════════════════════════════

CNNClassifier = CNN1D
"""Alias for the primary CNN classifier."""
