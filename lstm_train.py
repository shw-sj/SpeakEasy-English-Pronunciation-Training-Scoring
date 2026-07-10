"""Train a bidirectional LSTM on frame-level MFCC sequences."""

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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from lstm_network import LSTMNetwork

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import FEATURES_DIR, RANDOM_SEED

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], labels: np.ndarray) -> None:
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return (
            torch.as_tensor(self.sequences[index], dtype=torch.float32),
            int(self.labels[index]),
        )


def collate_sequences(batch):
    sequences, labels = zip(*batch)
    lengths = torch.as_tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, torch.as_tensor(labels, dtype=torch.long)


def load_split(split: str) -> tuple[list[np.ndarray], np.ndarray]:
    npz_path = FEATURES_DIR / f"letters_{split}_sequences.npz"
    csv_path = FEATURES_DIR / f"letters_{split}_seq_manifest.csv"
    if not npz_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            f"缺少 LSTM 序列数据：{npz_path} 或 {csv_path}\n"
            "请先运行：python src/prepare_data.py --export-sequences"
        )
    with np.load(npz_path) as archive:
        keys = sorted(archive.files)
        sequences = [archive[key].astype(np.float32) for key in keys]
    labels = pd.read_csv(csv_path)["label"].astype(str).to_numpy()
    if len(sequences) != len(labels):
        raise ValueError(f"{split} 序列数与标签数不一致")
    return sequences, labels


def compute_normalization(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_frames = np.concatenate(sequences, axis=0)
    mean = all_frames.mean(axis=0).astype(np.float32)
    std = (all_frames.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std


def normalize_sequences(
    sequences: list[np.ndarray], mean: np.ndarray, std: np.ndarray
) -> list[np.ndarray]:
    return [((seq - mean) / std).astype(np.float32) for seq in sequences]


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for sequences, lengths, labels in loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(sequences, lengths)
            total_loss += float(criterion(logits, labels).item()) * len(labels)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += len(labels)
    return total_loss / total_count, total_correct / total_count


def train(args) -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    train_sequences, train_labels_raw = load_split("train")
    val_sequences, val_labels_raw = load_split("val")
    label_encoder = LabelEncoder().fit(train_labels_raw)
    train_labels = label_encoder.transform(train_labels_raw)
    val_labels = label_encoder.transform(val_labels_raw)

    norm_mean, norm_std = compute_normalization(train_sequences)
    train_sequences = normalize_sequences(train_sequences, norm_mean, norm_std)
    val_sequences = normalize_sequences(val_sequences, norm_mean, norm_std)

    input_dim = train_sequences[0].shape[1]
    num_classes = len(label_encoder.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNetwork(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
        bidirectional=not args.unidirectional,
        embedding_dim=args.embedding_dim,
    ).to(device)

    train_loader = DataLoader(
        SequenceDataset(train_sequences, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        SequenceDataset(val_sequences, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    patience = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []

    print(
        f"[LSTM] train={len(train_sequences)} val={len(val_sequences)} "
        f"input_dim={input_dim} classes={num_classes} device={device}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for sequences, lengths, labels in train_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(sequences, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)

        train_loss = running_loss / seen
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(
            f"[LSTM] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.2%}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout_rate": args.dropout,
                    "bidirectional": not args.unidirectional,
                    "embedding_dim": args.embedding_dim,
                    "label_encoder_classes": label_encoder.classes_,
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                },
                output_path,
            )
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[LSTM] early stop at epoch {epoch}")
                break

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path("results") / "lstm_letters_curve.png", dpi=160)
    plt.close()
    print(f"[LSTM] 模型已保存：{output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SpeakEasy LSTM")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="results/best_lstm_letters_best_acc.pth",
        help="Checkpoint output path (use a new name to avoid overwriting the deployed model)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
