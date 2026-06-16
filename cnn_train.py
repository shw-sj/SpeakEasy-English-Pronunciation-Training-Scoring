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
from torch.utils.data import DataLoader, TensorDataset

from cnn_network import CNN1D

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import FEATURES_DIR

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _normalize(X_train_raw: np.ndarray, X_val_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0) + 1e-8
    return (X_train_raw - mean) / std, (X_val_raw - mean) / std, mean, std


def load_combined_data():
    all_X_train, all_y_train_raw = [], []
    all_X_val, all_y_val_raw = [], []

    for prefix, ds in [("letter", "letters"), ("word", "words")]:
        train_path = FEATURES_DIR / f"{ds}_train_features.npy"
        train_csv = FEATURES_DIR / f"{ds}_train_manifest.csv"
        val_path = FEATURES_DIR / f"{ds}_val_features.npy"
        val_csv = FEATURES_DIR / f"{ds}_val_manifest.csv"

        if not train_path.exists():
            continue
        X_tr = np.load(train_path).astype(np.float32)
        df_tr = pd.read_csv(train_csv)
        all_X_train.append(X_tr)
        all_y_train_raw.extend(f"{prefix}:{lbl}" for lbl in df_tr["label"].values)

        if val_path.exists() and val_csv.exists():
            X_v = np.load(val_path).astype(np.float32)
            df_v = pd.read_csv(val_csv)
            all_X_val.append(X_v)
            all_y_val_raw.extend(f"{prefix}:{lbl}" for lbl in df_v["label"].values)

    if not all_X_train or not all_X_val:
        return None

    X_train_raw = np.concatenate(all_X_train, axis=0)
    X_val_raw = np.concatenate(all_X_val, axis=0)
    y_train_raw = np.array(all_y_train_raw)
    y_val_raw = np.array(all_y_val_raw)

    label_encoder = LabelEncoder().fit(np.concatenate([y_train_raw, y_val_raw]))
    y_train = label_encoder.transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    X_train, X_val, mean, std = _normalize(X_train_raw, X_val_raw)

    num_classes = len(label_encoder.classes_)
    return X_train, X_val, y_train, y_val, num_classes, label_encoder, mean, std


def load_pipeline_data(dataset: str = "letters"):
    train_path = FEATURES_DIR / f"{dataset}_train_features.npy"
    train_csv = FEATURES_DIR / f"{dataset}_train_manifest.csv"
    val_path = FEATURES_DIR / f"{dataset}_val_features.npy"
    val_csv = FEATURES_DIR / f"{dataset}_val_manifest.csv"

    if not train_path.exists():
        print(f"[ERROR] Feature file not found: {train_path}")
        print(f"  Run: python src/prepare_data.py --dataset {dataset}")
        return None

    X_train_raw = np.load(train_path).astype(np.float32)
    y_train_raw = pd.read_csv(train_csv)["label"].values
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)

    if not val_path.exists() or not val_csv.exists():
        print(f"[ERROR] Validation files not found: {val_path}, {val_csv}")
        print(f"  Run: python src/prepare_data.py --dataset {dataset}")
        return None

    X_val_raw = np.load(val_path).astype(np.float32)
    y_val_raw = pd.read_csv(val_csv)["label"].values
    y_val = label_encoder.transform(y_val_raw)

    X_train, X_val, mean, std = _normalize(X_train_raw, X_val_raw)
    num_classes = len(label_encoder.classes_)
    return X_train, X_val, y_train, y_val, num_classes, label_encoder, mean, std


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def train_cnn_network(dataset: str = "letters"):
    learning_rate = 1e-3
    lr_decay = 0.98
    epochs = 100
    early_stop_patience = 15
    batch_size = 32
    weight_decay = 1e-4
    dropout_rate = 0.3
    clip_grad = 5.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_combined_data() if dataset == "both" else load_pipeline_data(dataset)
    if data is None:
        return
    X_train, X_val, y_train, y_val, num_classes, label_encoder, norm_mean, norm_std = data
    input_dim = X_train.shape[1]

    print("=" * 72)
    print(
        f"[CNN] dataset={dataset} | train={X_train.shape} | val={X_val.shape} | "
        f"classes={num_classes} | input_dim={input_dim} | device={device}"
    )
    print("=" * 72)

    model = CNN1D(input_dim=input_dim, num_classes=num_classes, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_X = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(y_val, dtype=torch.long, device=device)

    train_loss, val_loss, val_acc = [], [], []
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    early_stop_counter = 0
    save_path = Path("results") / f"best_cnn_{dataset}_model.pth"

    for epoch in range(1, epochs + 1):
        current_lr = learning_rate * (lr_decay ** (epoch - 1))
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        model.train()
        total_loss = 0.0
        total_seen = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            total_loss += float(loss.item()) * len(X_batch)
            total_seen += len(X_batch)

        avg_train_loss = total_loss / max(total_seen, 1)
        train_loss.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            vl = float(criterion(val_logits, val_y).item())
            pred = val_logits.argmax(dim=1)
            va = float((pred == val_y).float().mean().item())
        val_loss.append(vl)
        val_acc.append(va)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[CNN] Epoch {epoch:03d}/{epochs} | lr={current_lr:.6f} | "
                f"train_loss={avg_train_loss:.4f} | val_loss={vl:.4f} | val_acc={va:.2%}"
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
                "label_encoder_classes": label_encoder.classes_,
                "norm_mean": norm_mean,
                "norm_std": norm_std,
            }, save_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"[CNN] Early stop: val_loss did not improve for {early_stop_patience} epochs")
                break

    print(f"[CNN] Best model saved: {save_path}")
    print(f"[CNN] Best epoch={best_epoch} | best_val_loss={best_val_loss:.4f} | best_val_acc={best_val_acc:.2%}")

    Path("results").mkdir(exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="train loss", color="blue")
    plt.plot(val_loss, label="val loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label="val accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    curve_path = Path("results") / f"cnn_{dataset}_curve.png"
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    print(f"[CNN] Curve saved: {curve_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch 1D-CNN model")
    parser.add_argument("--dataset", choices=["letters", "words", "both"], default="letters")
    args = parser.parse_args()

    if args.dataset != "both":
        train_path = FEATURES_DIR / f"{args.dataset}_train_features.npy"
        if not train_path.exists():
            print(f"[INFO] Feature file not found: {train_path}")
            print(f"  Run: python src/prepare_data.py --dataset {args.dataset}")
            sys.exit(1)

    train_cnn_network(args.dataset)
