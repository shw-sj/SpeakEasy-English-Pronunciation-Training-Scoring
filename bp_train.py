"""Train BPNetwork (MLP) on aggregated MFCC features for letter recognition.

Improvements over baseline:
- FocalLoss: focuses training on hard-to-classify letters (B/D, M/N, etc.)
- OneCycleLR: single-cycle cosine anneal for faster, better convergence
- SWA (Stochastic Weight Averaging): flat-minima ensemble for better generalisation
- SiLU activation: smoother gradients than ReLU
- Input dropout + frequency masking for better robustness
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, TensorDataset

from bp_network import BPNetwork

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from audio_feature import feature_dim_rich
from config import FEATURES_DIR
from train_utils import (
    FocalLoss,
    add_gradient_noise,
    compute_loss,
    freq_mask_features,
    mixup_batch as mixup_batch_new,
)

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False


INPUT_DIM = feature_dim_rich()  # 156-dim rich features
MAX_LR = 5e-3
WEIGHT_DECAY = 5e-3
DROPOUT_RATE = 0.4
CLIP_GRAD = 3.0
EPOCHS = 150
EARLY_STOP_PATIENCE = 30
BATCH_SIZE = 64

# ── Regularisation knobs ──
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.3
GRAD_NOISE_STD = 0.001

# ── Focal Loss ──
FOCAL_GAMMA = 2.0

# ── OneCycleLR ──
ONECYCLE_PCT_START = 0.3
ONECYCLE_FINAL_DIV = 1e4

# ── SWA ──
SWA_START = 100  # epoch to start averaging
SWA_LR = 1e-4


def normalize(train_x: np.ndarray, val_x: np.ndarray):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-8
    return (train_x - mean) / std, (val_x - mean) / std, mean, std


def load_dataset():
    train_path = FEATURES_DIR / "letters_train_features.npy"
    train_csv = FEATURES_DIR / "letters_train_manifest.csv"
    val_path = FEATURES_DIR / "letters_val_features.npy"
    val_csv = FEATURES_DIR / "letters_val_manifest.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Missing prepared letter features. Run src/prepare_data.py first."
        )
    x_train_raw = np.load(train_path).astype(np.float32)
    x_val_raw = np.load(val_path).astype(np.float32)
    y_train_raw = pd.read_csv(train_csv)["label"].values
    y_val_raw = pd.read_csv(val_csv)["label"].values

    label_encoder = LabelEncoder().fit(np.concatenate([y_train_raw, y_val_raw]))
    y_train = label_encoder.transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    x_train, x_val, mean, std = normalize(x_train_raw, x_val_raw)
    return x_train, x_val, y_train, y_val, label_encoder, mean, std


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ── Training ─────────────────────────────────────────────────────────

def train_bp_network() -> None:
    x_train, x_val, y_train, y_val, label_encoder, norm_mean, norm_std = load_dataset()
    input_dim = x_train.shape[1]
    output_dim = len(label_encoder.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BPNetwork(
        input_size=input_dim,
        output_size=output_dim,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    # FocalLoss: auto-focuses on confusing letter pairs
    criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR,
                                  weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(x_train, y_train, BATCH_SIZE, shuffle=True)
    val_x = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(y_val, dtype=torch.long, device=device)

    # ── OneCycleLR: warmup 30% → peak → cosine anneal to ~0 ──
    total_steps = EPOCHS * len(train_loader)
    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=total_steps,
        pct_start=ONECYCLE_PCT_START, anneal_strategy="cos",
        final_div_factor=ONECYCLE_FINAL_DIV,
    )

    # ── SWA for better generalisation ──
    swa_model = AveragedModel(model).to(device)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR,
                          anneal_epochs=5, anneal_strategy="cos")

    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    model_save_path = save_dir / "bp_letters_best_acc.pth"
    swa_save_path = save_dir / "bp_letters_swa.pth"
    plot_save_path = save_dir / "bp_letters_curve.png"

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    patience = 0

    print("=" * 72)
    print(
        f"[BP] train={x_train.shape} | val={x_val.shape} | "
        f"classes={output_dim} | input_dim={input_dim} | device={device}"
    )
    print(f"[BP] FocalLoss γ={FOCAL_GAMMA} | label_smoothing={LABEL_SMOOTHING} | "
          f"mixup_alpha={MIXUP_ALPHA}")
    print(f"[BP] dropout={DROPOUT_RATE} | weight_decay={WEIGHT_DECAY}")
    print(f"[BP] OneCycleLR max_lr={MAX_LR} pct_start={ONECYCLE_PCT_START} "
          f"final_div={ONECYCLE_FINAL_DIV}")
    print(f"[BP] SWA start={SWA_START} swa_lr={SWA_LR}")
    print("=" * 72)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            batch_x, batch_y_mixed = mixup_batch_new(
                batch_x, batch_y, MIXUP_ALPHA, output_dim)

            # Frequency-bin masking: forces model to use diverse frequency regions
            batch_x = freq_mask_features(batch_x, freq_groups=13)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)

            loss = compute_loss(logits, batch_y_mixed, criterion)

            loss.backward()

            add_gradient_noise(model, GRAD_NOISE_STD)

            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            lr_schedule.step()  # per-batch LR update
            total_loss += float(loss.item()) * len(batch_x)
            total_seen += len(batch_x)

        train_loss = total_loss / max(total_seen, 1)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(val_criterion(val_logits, val_y).item())
            val_pred = val_logits.argmax(dim=1)
            val_acc = float((val_pred == val_y).float().mean().item())
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # ── SWA update ──
        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        if epoch == 1 or epoch % 10 == 0:
            swa_tag = " [SWA]" if epoch >= SWA_START else ""
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[BP] Epoch {epoch:03d}/{EPOCHS} | lr={current_lr:.6f} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.2%}{swa_tag}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_acc = val_acc
            patience = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "dropout_rate": DROPOUT_RATE,
                    "label_encoder_classes": label_encoder.classes_,
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                },
                model_save_path,
            )
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"[BP] Early stop: val_loss did not improve for {EARLY_STOP_PATIENCE} epochs")
                break

    # ── Finalise SWA: update BN stats, save ──
    if epoch >= SWA_START:
        print("[BP] Updating SWA BatchNorm statistics …")
        update_bn(train_loader, swa_model, device=device)
        # Strip 'module.' prefix so GUI can load without AveragedModel wrapper
        swa_state = {k[len("module."):] if k.startswith("module.") else k: v
                     for k, v in swa_model.state_dict().items()
                     if k != "n_averaged"}
        torch.save(
            {
                "state_dict": swa_state,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "dropout_rate": DROPOUT_RATE,
                "label_encoder_classes": label_encoder.classes_,
                "norm_mean": norm_mean,
                "norm_std": norm_std,
            },
            swa_save_path,
        )
        print(f"[BP] SWA model saved: {swa_save_path}")

    print(f"[BP] Best model saved: {model_save_path}")
    print(f"[BP] Best epoch={best_epoch} | best_val_loss={best_val_loss:.4f} | best_val_acc={best_val_acc:.2%}")

    # ── Evaluate SWA on validation set ──
    if epoch >= SWA_START:
        swa_model.eval()
        with torch.no_grad():
            swa_logits = swa_model(val_x)
            swa_pred = swa_logits.argmax(dim=1)
            swa_val_acc = float((swa_pred == val_y).float().mean().item())
        print(f"[BP] SWA validation accuracy: {swa_val_acc:.2%}")

    # ── Evaluate on test set if available ──
    test_path = FEATURES_DIR / "letters_test_features.npy"
    test_csv = FEATURES_DIR / "letters_test_manifest.csv"
    if test_path.exists() and test_csv.exists():
        x_test_raw = np.load(test_path).astype(np.float32)
        test_df = pd.read_csv(test_csv)
        y_test_raw = test_df["label"].values
        y_test = label_encoder.transform(y_test_raw)
        x_test = (x_test_raw - norm_mean) / (norm_std + 1e-8)
        test_x_t = torch.as_tensor(x_test, dtype=torch.float32, device=device)
        test_y_t = torch.as_tensor(y_test, dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            test_preds = model(test_x_t).argmax(dim=1)
            test_acc = float((test_preds == test_y_t).float().mean().item())
        print(f"[BP] Test accuracy: {test_acc:.2%}")

        if epoch >= SWA_START:
            swa_model.eval()
            with torch.no_grad():
                swa_test_preds = swa_model(test_x_t).argmax(dim=1)
                swa_test_acc = float((swa_test_preds == test_y_t).float().mean().item())
            print(f"[BP] SWA Test accuracy: {swa_test_acc:.2%}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train loss", color="blue")
    plt.plot(val_losses, label="val loss", color="red")
    if SWA_START <= EPOCHS:
        plt.axvline(x=SWA_START - 1, color="orange", linestyle="--", alpha=0.6, label="SWA start")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="val accuracy", color="green")
    if SWA_START <= EPOCHS:
        plt.axvline(x=SWA_START - 1, color="orange", linestyle="--", alpha=0.6, label="SWA start")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    print(f"[BP] Curve saved: {plot_save_path}")
    plt.show()


def load_bp_model(model_path: str | Path):
    data = torch.load(model_path, map_location="cpu", weights_only=False)
    model = BPNetwork(
        input_size=int(data["input_dim"]),
        output_size=int(data["output_dim"]),
        dropout_rate=float(data.get("dropout_rate", 0.0)),
    )
    model.load_state_dict(data["state_dict"])
    model.eval()
    label_encoder = LabelEncoder()
    label_encoder.classes_ = data["label_encoder_classes"]
    return model, label_encoder, data.get("norm_mean"), data.get("norm_std")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BP neural network for letter recognition")
    args = parser.parse_args()
    train_bp_network()
