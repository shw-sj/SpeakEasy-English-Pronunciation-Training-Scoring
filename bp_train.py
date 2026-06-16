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
from torch.utils.data import DataLoader, TensorDataset

from bp_network import BPNetwork

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from audio_feature import feature_dim
from config import FEATURES_DIR

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False


INPUT_DIM = feature_dim()
LR = 1e-3
LR_DECAY = 0.95
WEIGHT_DECAY = 1e-3
DROPOUT_RATE = 0.3
CLIP_GRAD = 5.0
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
BATCH_SIZE = 64


def normalize(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-8
    return (train_x - mean) / std, (val_x - mean) / std, mean, std


def load_dataset(dataset: str):
    if dataset == "both":
        train_features, val_features = [], []
        train_labels, val_labels = [], []
        for prefix, name in [("letter", "letters"), ("word", "words")]:
            train_path = FEATURES_DIR / f"{name}_train_features.npy"
            train_csv = FEATURES_DIR / f"{name}_train_manifest.csv"
            val_path = FEATURES_DIR / f"{name}_val_features.npy"
            val_csv = FEATURES_DIR / f"{name}_val_manifest.csv"
            if not train_path.exists() or not val_path.exists():
                continue
            train_features.append(np.load(train_path).astype(np.float32))
            val_features.append(np.load(val_path).astype(np.float32))
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            train_labels.extend(f"{prefix}:{x}" for x in train_df["label"].values)
            val_labels.extend(f"{prefix}:{x}" for x in val_df["label"].values)

        if not train_features:
            raise FileNotFoundError("No prepared letter/word features found. Run src/prepare_data.py first.")

        x_train_raw = np.concatenate(train_features, axis=0)
        x_val_raw = np.concatenate(val_features, axis=0)
        y_train_raw = np.array(train_labels)
        y_val_raw = np.array(val_labels)
    else:
        train_path = FEATURES_DIR / f"{dataset}_train_features.npy"
        train_csv = FEATURES_DIR / f"{dataset}_train_manifest.csv"
        val_path = FEATURES_DIR / f"{dataset}_val_features.npy"
        val_csv = FEATURES_DIR / f"{dataset}_val_manifest.csv"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Missing prepared features for {dataset}. Run src/prepare_data.py --dataset {dataset}")
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


def train_bp_network(dataset: str = "letters") -> None:
    x_train, x_val, y_train, y_val, label_encoder, norm_mean, norm_std = load_dataset(dataset)
    input_dim = x_train.shape[1]
    output_dim = len(label_encoder.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BPNetwork(
        input_size=input_dim,
        output_size=output_dim,
        dropout_rate=DROPOUT_RATE,
        task=dataset,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(x_train, y_train, BATCH_SIZE, shuffle=True)
    val_x = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(y_val, dtype=torch.long, device=device)

    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    model_save_path = save_dir / f"bp_{dataset}_model.pth"
    plot_save_path = save_dir / f"bp_{dataset}_curve.png"

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    patience = 0

    print("=" * 72)
    print(
        f"[BP] dataset={dataset} | train={x_train.shape} | val={x_val.shape} | "
        f"classes={output_dim} | input_dim={input_dim} | device={device}"
    )
    print("=" * 72)

    for epoch in range(1, EPOCHS + 1):
        current_lr = LR * (LR_DECAY ** (epoch - 1))
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            total_seen += len(batch_x)

        train_loss = total_loss / max(total_seen, 1)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).item())
            val_pred = val_logits.argmax(dim=1)
            val_acc = float((val_pred == val_y).float().mean().item())
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[BP] Epoch {epoch:03d}/{EPOCHS} | lr={current_lr:.6f} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2%}"
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
                    "task": dataset,
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

    print(f"[BP] Best model saved: {model_save_path}")
    print(f"[BP] Best epoch={best_epoch} | best_val_loss={best_val_loss:.4f} | best_val_acc={best_val_acc:.2%}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train loss", color="blue")
    plt.plot(val_losses, label="val loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="val accuracy", color="green")
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
        task=str(data.get("task", "letters")),
    )
    model.load_state_dict(data["state_dict"])
    model.eval()
    label_encoder = LabelEncoder()
    label_encoder.classes_ = data["label_encoder_classes"]
    return model, label_encoder, data.get("norm_mean"), data.get("norm_std")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BP neural network")
    parser.add_argument("--dataset", choices=["letters", "words", "both"], default="letters")
    args = parser.parse_args()
    train_bp_network(args.dataset)
