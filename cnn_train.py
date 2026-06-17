"""Train CNN1D on aggregated MFCC features for letter recognition.

Improvements over baseline:
- FocalLoss: focuses training on hard-to-classify letters
- CosineWarmRestarts + warmup: better LR schedule
- SWA (Stochastic Weight Averaging): flat-minima ensemble for better generalisation
- Larger channels (32→64→128→256) with freq_groups=13 for meaningful conv over mel bins
"""

from __future__ import annotations

import argparse
import sys
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

from cnn_network import CNN1D

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import FEATURES_DIR
from train_utils import (
    CosineWarmRestarts,
    FocalLoss,
    add_gradient_noise,
    compute_loss,
    mixup_batch as mixup_batch_new,
)

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _normalize(X_train_raw, X_val_raw):
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0) + 1e-8
    return (X_train_raw - mean) / std, (X_val_raw - mean) / std, mean, std


# ── Data loading ─────────────────────────────────────────────────────

def load_pipeline_data():
    train_path = FEATURES_DIR / "letters_train_features.npy"
    train_csv = FEATURES_DIR / "letters_train_manifest.csv"
    val_path = FEATURES_DIR / "letters_val_features.npy"
    val_csv = FEATURES_DIR / "letters_val_manifest.csv"

    if not train_path.exists():
        print(f"[ERROR] Feature file not found: {train_path}")
        print(f"  Run: python src/prepare_data.py")
        return None

    X_train_raw = np.load(train_path).astype(np.float32)
    y_train_raw = pd.read_csv(train_csv)["label"].values
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)

    if not val_path.exists() or not val_csv.exists():
        print(f"[ERROR] Validation files not found: {val_path}, {val_csv}")
        print(f"  Run: python src/prepare_data.py")
        return None

    X_val_raw = np.load(val_path).astype(np.float32)
    y_val_raw = pd.read_csv(val_csv)["label"].values
    y_val = label_encoder.transform(y_val_raw)

    X_train, X_val, mean, std = _normalize(X_train_raw, X_val_raw)
    num_classes = len(label_encoder.classes_)
    return X_train, X_val, y_train, y_val, num_classes, label_encoder, mean, std


def _make_loader(X, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=shuffle,
    )


# ── Training ─────────────────────────────────────────────────────────

def train_cnn_network():
    # ── Hyperparameters ──
    learning_rate = 1e-3
    lr_min = 1e-6
    epochs = 120
    early_stop_patience = 25
    batch_size = 64
    weight_decay = 4e-3
    dropout_rate = 0.35
    clip_grad = 3.0
    label_smoothing = 0.1
    mixup_alpha = 0.3
    grad_noise_std = 0.0

    # ── Focal Loss ──
    focal_gamma = 2.0

    # ── LR schedule ──
    warmup_epochs = 5
    cosine_t0 = 30
    cosine_tmult = 2

    # ── SWA ──
    swa_start = 80
    swa_lr = 1e-4

    # ── CNN architecture ──
    # freq_groups=13 groups the 156-dim vector into (12, 13):
    #   12 = 3 channels (static/Δ/ΔΔ) × 4 stats (mean/std/min/max)
    #   13 = mel-frequency bins → convolution slides along the frequency axis
    channels = (32, 64, 128, 256)
    freq_groups = 13

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_pipeline_data()
    if data is None:
        return
    X_train, X_val, y_train, y_val, num_classes, label_encoder, norm_mean, norm_std = data
    input_dim = X_train.shape[1]

    print("=" * 72)
    print(
        f"[CNN] train={X_train.shape} | val={X_val.shape} | "
        f"classes={num_classes} | input_dim={input_dim} | device={device}"
    )
    print(f"[CNN] FocalLoss γ={focal_gamma} | label_smoothing={label_smoothing} | "
          f"mixup_alpha={mixup_alpha}")
    print(f"[CNN] dropout={dropout_rate} | weight_decay={weight_decay}")
    print(f"[CNN] channels={channels} | freq_groups={freq_groups}")
    print(f"[CNN] CosineWarmRestarts T0={cosine_t0} Tmult={cosine_tmult} "
          f"warmup={warmup_epochs}")
    print(f"[CNN] SWA start={swa_start} swa_lr={swa_lr}")
    print("=" * 72)

    # ── Model ──
    model = CNN1D(input_dim=input_dim, num_classes=num_classes,
                  dropout_rate=dropout_rate,
                  channels=channels, freq_groups=freq_groups).to(device)

    # FocalLoss: auto-focuses on confusing letter pairs
    criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_X = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(y_val, dtype=torch.long, device=device)

    # ── LR schedule ──
    lr_schedule = CosineWarmRestarts(
        base_lr=learning_rate, min_lr=lr_min,
        T_0=cosine_t0, T_mult=cosine_tmult,
        warmup_epochs=warmup_epochs,
    )

    # ── SWA ──
    swa_model = AveragedModel(model).to(device)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr,
                          anneal_epochs=5, anneal_strategy="cos")

    train_loss, val_loss, val_acc = [], [], []
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    early_stop_counter = 0
    save_path = Path("results") / "best_cnn_letters_best_acc.pth"
    swa_save_path = Path("results") / "best_cnn_letters_swa.pth"

    for epoch in range(1, epochs + 1):
        current_lr = lr_schedule.get_lr(epoch)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        model.train()
        total_loss = 0.0
        total_seen = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            X_batch, y_target = mixup_batch_new(X_batch, y_batch, mixup_alpha, num_classes)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)

            loss = compute_loss(logits, y_target, criterion)

            loss.backward()

            add_gradient_noise(model, grad_noise_std)

            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            total_loss += float(loss.item()) * len(X_batch)
            total_seen += len(X_batch)

        avg_train_loss = total_loss / max(total_seen, 1)
        train_loss.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            vl = float(val_criterion(val_logits, val_y).item())
            pred = val_logits.argmax(dim=1)
            va = float((pred == val_y).float().mean().item())
        val_loss.append(vl)
        val_acc.append(va)

        # ── SWA update ──
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        if epoch == 1 or epoch % 10 == 0:
            swa_tag = " [SWA]" if epoch >= swa_start else ""
            print(
                f"[CNN] Epoch {epoch:03d}/{epochs} | lr={current_lr:.6f} | "
                f"train_loss={avg_train_loss:.4f} | val_loss={vl:.4f} | "
                f"val_acc={va:.2%}{swa_tag}"
            )

        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch = epoch
            best_val_acc = va
            early_stop_counter = 0
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "dropout_rate": dropout_rate,
                "channels": model.channels,
                "freq_groups": model.freq_groups,
                "label_encoder_classes": label_encoder.classes_,
                "norm_mean": norm_mean,
                "norm_std": norm_std,
            }, save_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"[CNN] Early stop at epoch {epoch}")
                break

    # ── Finalise SWA: update BN stats, save ──
    if epoch >= swa_start:
        print("[CNN] Updating SWA BatchNorm statistics …")
        update_bn(train_loader, swa_model, device=device)
        swa_state = swa_model.state_dict()
        torch.save({
            "state_dict": swa_state,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate,
            "channels": model.channels,
            "freq_groups": model.freq_groups,
            "label_encoder_classes": label_encoder.classes_,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
        }, swa_save_path)
        print(f"[CNN] SWA model saved: {swa_save_path}")

    print(f"[CNN] Best model saved: {save_path}")
    print(f"[CNN] Best epoch={best_epoch} | best_val_loss={best_val_loss:.4f} | best_val_acc={best_val_acc:.2%}")

    # ── Evaluate SWA on validation set ──
    if epoch >= swa_start:
        swa_model.eval()
        with torch.no_grad():
            swa_logits = swa_model(val_X)
            swa_pred = swa_logits.argmax(dim=1)
            swa_val_acc = float((swa_pred == val_y).float().mean().item())
        print(f"[CNN] SWA validation accuracy: {swa_val_acc:.2%}")

    # ── Test set evaluation ──
    test_path = FEATURES_DIR / "letters_test_features.npy"
    test_csv = FEATURES_DIR / "letters_test_manifest.csv"
    if test_path.exists() and test_csv.exists():
        x_test_raw = np.load(test_path).astype(np.float32)
        test_df = pd.read_csv(test_csv)
        y_test = label_encoder.transform(test_df["label"].values)
        x_test = (x_test_raw - norm_mean) / (norm_std + 1e-8)
        test_x_t = torch.as_tensor(x_test, dtype=torch.float32, device=device)
        test_y_t = torch.as_tensor(y_test, dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            test_preds = model(test_x_t).argmax(dim=1)
            test_acc = float((test_preds == test_y_t).float().mean().item())
        print(f"[CNN] Test accuracy: {test_acc:.2%}")

        if epoch >= swa_start:
            swa_model.eval()
            with torch.no_grad():
                swa_test_preds = swa_model(test_x_t).argmax(dim=1)
                swa_test_acc = float((swa_test_preds == test_y_t).float().mean().item())
            print(f"[CNN] SWA Test accuracy: {swa_test_acc:.2%}")

    Path("results").mkdir(exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="train loss", color="blue")
    plt.plot(val_loss, label="val loss", color="red")
    if swa_start <= epochs:
        plt.axvline(x=swa_start - 1, color="orange", linestyle="--", alpha=0.6, label="SWA start")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label="val accuracy", color="green")
    if swa_start <= epochs:
        plt.axvline(x=swa_start - 1, color="orange", linestyle="--", alpha=0.6, label="SWA start")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    curve_path = Path("results") / "cnn_letters_curve.png"
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    print(f"[CNN] Curve saved: {curve_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch 1D-CNN model for letter recognition")
    args = parser.parse_args()

    train_path = FEATURES_DIR / "letters_train_features.npy"
    if not train_path.exists():
        print(f"[INFO] Feature file not found: {train_path}")
        print(f"  Run: python src/prepare_data.py")
        sys.exit(1)

    train_cnn_network()
