"""Shared training utilities for all models.

Provides:
- FocalLoss: addresses class imbalance by up-weighting hard examples
- CosineWarmRestarts: cosine LR schedule with periodic restarts
- SWA helpers: Stochastic Weight Averaging for better generalization
- Mixup: data-level regularization for vectors and sequences
- Gradient noise: flat-minima regularization
- LR warmup: linear ramp-up at training start
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR


# ═══════════════════════════════════════════════════════════════════════
#  Focal Loss
# ═══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss: CE(pt) * (1 - pt)^gamma.

    Down-weights easy examples so training focuses on hard ones.
    Useful for addressing class imbalance in pronunciation recognition.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  0 = standard CE, 2 = strong focus on hards.
    label_smoothing : float
        Softens one-hot targets (0 = hard targets).
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits : (B, C) raw logits.
        targets : (B,) class indices, or (B, C) soft targets from mixup.
        """
        if targets.dim() == 2:
            # Soft targets (from mixup) — use KL-divergence style loss
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            # Per-sample focal weight based on max prob
            with torch.no_grad():
                pt = (targets * probs).sum(dim=-1).clamp(min=1e-7)
                focal_weight = (1 - pt) ** self.gamma
            loss = -(targets * log_probs).sum(dim=-1)
            loss = focal_weight * loss
            return loss.mean()

        # Hard targets
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.label_smoothing > 0:
            # Apply label smoothing in focal context
            n_classes = logits.size(-1)
            smooth_loss = -log_probs.mean(dim=-1)
            nll = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth_loss

        with torch.no_grad():
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7)
            focal_weight = (1 - pt) ** self.gamma

        return (focal_weight * nll).mean()


# ═══════════════════════════════════════════════════════════════════════
#  LR Schedules
# ═══════════════════════════════════════════════════════════════════════

class CosineWarmRestarts:
    """Cosine annealing with warm restarts (SGDR).

    Each restart cycle is T_0 * T_mult^(cycle) epochs long.
    Within each cycle, LR follows a cosine curve from base_lr to min_lr.

    Parameters
    ----------
    base_lr : float
        Peak learning rate at the start of each cycle.
    min_lr : float
        Minimum learning rate at the end of each cycle.
    T_0 : int
        Length of the first cycle in epochs.
    T_mult : int
        Cycle length multiplier (T_mult=2 → each cycle is twice as long).
    warmup_epochs : int
        Number of linear warmup epochs at the very beginning.
    """

    def __init__(
        self,
        base_lr: float,
        min_lr: float,
        T_0: int = 30,
        T_mult: int = 2,
        warmup_epochs: int = 5,
    ):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.warmup_epochs = warmup_epochs

    def get_lr(self, epoch: int) -> float:
        """Return learning rate for the given epoch (1-indexed)."""
        if epoch <= self.warmup_epochs:
            # Linear warmup
            return self.min_lr + (self.base_lr - self.min_lr) * epoch / self.warmup_epochs

        # Shift epoch to account for warmup
        t = epoch - self.warmup_epochs

        # Find which cycle we're in
        cycle_len = self.T_0
        cycle_start = 0
        while t >= cycle_start + cycle_len:
            cycle_start += cycle_len
            cycle_len *= self.T_mult

        t_in_cycle = t - cycle_start
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * t_in_cycle / cycle_len)
        )


def cosine_lr_simple(epoch: int, base_lr: float, min_lr: float,
                     total_epochs: int, warmup_epochs: int = 0) -> float:
    """Simple cosine decay (no restarts).  Used for letter training (unchanged)."""
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return min_lr + (base_lr - min_lr) * epoch / warmup_epochs
    effective = epoch - warmup_epochs
    effective_total = total_epochs - warmup_epochs
    return min_lr + 0.5 * (base_lr - min_lr) * (
        1.0 + math.cos(math.pi * effective / max(effective_total, 1))
    )


# ═══════════════════════════════════════════════════════════════════════
#  Mixup
# ═══════════════════════════════════════════════════════════════════════

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixup to a batch of vector features.

    Returns (mixed_x, mixed_y_soft) where mixed_y is a one-hot mixture.
    """
    if alpha <= 0 or x.size(0) < 2:
        return x, y

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)

    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]

    y_onehot = torch.zeros(x.size(0), num_classes, device=x.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    y_idx_onehot = torch.zeros(x.size(0), num_classes, device=x.device)
    y_idx_onehot.scatter_(1, y[index].unsqueeze(1), 1.0)
    mixed_y = lam * y_onehot + (1 - lam) * y_idx_onehot

    return mixed_x, mixed_y


# ═══════════════════════════════════════════════════════════════════════
#  Gradient Noise
# ═══════════════════════════════════════════════════════════════════════

def add_gradient_noise(model: nn.Module, std: float) -> None:
    """Add small Gaussian noise to gradients (flat-minima regularizer)."""
    if std <= 0:
        return
    for p in model.parameters():
        if p.grad is not None:
            p.grad.add_(torch.randn_like(p.grad) * std)


# ═══════════════════════════════════════════════════════════════════════
#  Stochastic Weight Averaging (SWA)
# ═══════════════════════════════════════════════════════════════════════

class SWAManager:
    """Manages SWA lifecycle: wraps model, handles BN update, saves final weights.

    Usage::

        swa = SWAManager(model, optimizer, swa_lr=1e-4, device=device)

        for epoch in range(1, epochs + 1):
            ...
            if epoch >= swa_start:
                swa.step()

        # After training:
        swa.update_bn(train_loader)
        swa.save(save_path, **extra_meta)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        swa_lr: float = 1e-4,
        device: torch.device | None = None,
    ):
        self.device = device or next(model.parameters()).device
        self.model = model  # keep reference for update_parameters()
        self.averaged = AveragedModel(model).to(self.device)
        self.scheduler = SWALR(
            optimizer, swa_lr=swa_lr,
            anneal_epochs=5, anneal_strategy="cos",
        )
        self._active = False

    def step(self) -> None:
        """Call once per epoch during SWA phase."""
        if not self._active:
            self._active = True
        self.averaged.update_parameters(self.model)
        self.scheduler.step()

    def update_bn(self, train_loader: torch.utils.data.DataLoader) -> None:
        """Update BatchNorm statistics in the averaged model."""
        if not self._active:
            return
        from torch.optim.swa_utils import update_bn as _update_bn
        _update_bn(train_loader, self.averaged, device=self.device)

    def get_model(self) -> nn.Module:
        """Return the SWA-averaged model (or original if SWA never activated)."""
        if self._active:
            return self.averaged
        return self.model

    def save(self, save_path: Path, **extra_meta) -> None:
        """Save the SWA-averaged model checkpoint."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict = {"state_dict": self.averaged.state_dict(), **extra_meta}
        torch.save(checkpoint, save_path)

    @property
    def is_active(self) -> bool:
        return self._active


def swa_update_bn(
    swa_model: AveragedModel,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device | None = None,
) -> None:
    """Update BatchNorm running stats in the SWA model using training data."""
    from torch.optim.swa_utils import update_bn as _update_bn
    _update_bn(train_loader, swa_model, device=device)


def save_swa_checkpoint(
    swa_model: AveragedModel,
    save_path: Path,
    **extra_meta,
) -> None:
    """Save the SWA-averaged model state_dict along with metadata."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = swa_model.state_dict()
    checkpoint: dict = {"state_dict": state, **extra_meta}
    torch.save(checkpoint, save_path)


# ═══════════════════════════════════════════════════════════════════════
#  Training loss helper (handles both hard and soft targets)
# ═══════════════════════════════════════════════════════════════════════
#  Feature-level augmentation
# ═══════════════════════════════════════════════════════════════════════

def freq_mask_features(
    x: torch.Tensor,
    freq_groups: int = 13,
    mask_prob: float = 0.15,
    mask_ratio: float = 0.3,
) -> torch.Tensor:
    """Randomly mask frequency bins in feature vectors during training.

    For features organised as (B, input_dim) where input_dim % freq_groups == 0,
    groups dimensions into freq_groups bins and randomly masks entire bins.
    This simulates frequency-selective noise / microphone roll-off and
    forces the model to use multiple frequency regions for classification.

    Parameters
    ----------
    x : (B, D) float tensor.
    freq_groups : int
        Number of frequency bins (13 for mel-scale features).
    mask_prob : float
        Probability of masking any given frequency bin.
    mask_ratio : float
        Maximum fraction of bins to mask in a single sample.
    """
    if not (x.dim() == 2 and x.size(1) % freq_groups == 0):
        return x  # can't group, skip
    B, D = x.shape
    ch_per_bin = D // freq_groups
    x_reshaped = x.view(B, freq_groups, ch_per_bin)

    max_mask = max(1, int(freq_groups * mask_ratio))
    with torch.no_grad():
        mask = torch.rand(B, freq_groups, device=x.device) < mask_prob
        # Limit max number of masked bins per sample
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=True)[0]
            if len(idx) > max_mask:
                keep = idx[torch.randperm(len(idx))[:max_mask]]
                mask[b].fill_(False)
                mask[b, keep] = True
        mask = mask.unsqueeze(-1)  # (B, freq_groups, 1)
    x_reshaped = x_reshaped.masked_fill(mask, 0.0)
    return x_reshaped.view(B, D)


# ═══════════════════════════════════════════════════════════════════════

def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute loss, handling both hard labels (1D) and soft mixup labels (2D).

    When the criterion is :class:`FocalLoss` (which natively supports 2D
    soft targets with focal weighting), delegates directly so that focal
    weighting and label smoothing remain active during mixup.  For plain
    ``CrossEntropyLoss`` (which only accepts 1D class indices), soft
    targets are handled via a manual KL-divergence‑style computation.
    """
    if targets.dim() == 2:
        if isinstance(criterion, FocalLoss):
            return criterion(logits, targets)
        # Plain CE cannot ingest soft targets — compute manually
        return -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    return criterion(logits, targets)
