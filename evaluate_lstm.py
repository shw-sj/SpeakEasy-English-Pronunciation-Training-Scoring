"""Evaluate a trained BiLSTM checkpoint on the held-out test split.

Loads a checkpoint + pre-exported ``letters_test_sequences.npz``, runs
inference, and reports accuracy / macro-F1 / per-class metrics + figures.

Usage (run on server)::

    python evaluate_lstm.py                          # default checkpoint
    python evaluate_lstm.py --checkpoint results/best_lstm_letters_best_acc.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import FEATURES_DIR
from lstm_network import LSTMNetwork
from metrics import ClassificationMetrics, plot_confusion_matrix, plot_per_class_f1

DEFAULT_CHECKPOINT = Path("results") / "best_lstm_letters_best_acc.pth"


def load_model(device, checkpoint_path: Path):
    ck = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model = LSTMNetwork(
        input_dim=int(ck["input_dim"]),
        num_classes=int(ck["num_classes"]),
        hidden_size=int(ck.get("hidden_size", 128)),
        num_layers=int(ck.get("num_layers", 2)),
        dropout_rate=float(ck.get("dropout_rate", 0.3)),
        bidirectional=bool(ck.get("bidirectional", True)),
        embedding_dim=int(ck.get("embedding_dim", 128)),
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    labels = [str(x) for x in ck["label_encoder_classes"]]
    return model, labels, ck["norm_mean"], ck["norm_std"]


def load_test_split(split: str = "test") -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Load pre-exported sequences from npz (fast path)."""
    npz_path = FEATURES_DIR / f"letters_{split}_sequences.npz"
    csv_path = FEATURES_DIR / f"letters_{split}_seq_manifest.csv"
    if not npz_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {npz_path} or {csv_path}\n"
            "Run first: python prepare_lstm_data.py"
        )
    with np.load(npz_path) as archive:
        keys = sorted(archive.files)
        sequences = [archive[key].astype(np.float32) for key in keys]
    df = pd.read_csv(csv_path)
    labels = df["label"].astype(str).to_numpy()
    speakers = df["speaker"].astype(str).to_list()
    if len(sequences) != len(labels):
        raise ValueError(f"Sequence count mismatch: {len(sequences)} vs {len(labels)}")
    return sequences, labels, speakers


@torch.no_grad()
def predict_batch(model, sequences, mean, std, device) -> list[int]:
    results = []
    for seq in sequences:
        x = (seq.astype(np.float32) - mean) / (std + 1e-8)
        xt = torch.as_tensor(x[None, :, :], dtype=torch.float32, device=device)
        lengths = torch.as_tensor([len(x)], dtype=torch.long, device=device)
        logits = model(xt, lengths)
        results.append(int(logits.argmax(dim=1).item()))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LSTM checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}  checkpoint={args.checkpoint}  split={args.split}")

    if not args.checkpoint.exists():
        print(f"[eval] ERROR: checkpoint not found: {args.checkpoint}")
        return 1

    model, labels, mean, std = load_model(device, args.checkpoint)
    sequences, y_labels, speakers = load_test_split(args.split)

    print(f"[eval] loaded {len(sequences)} sequences, {len(set(speakers))} speakers")
    print(f"[eval] speakers: {sorted(set(speakers))}")

    y_pred = predict_batch(model, sequences, mean, std, device)

    label_to_idx = {c: i for i, c in enumerate(labels)}
    y_true = np.array([label_to_idx[lbl] for lbl in y_labels])
    y_pred = np.array(y_pred)

    cm = ClassificationMetrics(num_classes=len(labels), class_names=labels)
    res = cm.compute_all(y_true, y_pred)

    print(f"\n=== Test Results ({args.split}, {len(y_true)} samples) ===")
    print(f"  accuracy    = {res['accuracy']:.4f}")
    print(f"  macro-F1    = {res['macro_avg']['f1']:.4f}")
    print(f"  micro-F1    = {res['micro_avg']['f1']:.4f}")
    cm.print_report(y_true, y_pred)

    out_path = Path("results") / "eval_metrics.json"
    out = {
        "n_evaluated": int(len(y_true)),
        "accuracy": res["accuracy"],
        "macro_avg": res["macro_avg"],
        "micro_avg": res["micro_avg"],
        "per_class": res["per_class"],
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] metrics saved -> {out_path}")

    # Only generate figures when we have a meaningful spread (not 100%)
    if 0.0 < res["accuracy"] < 1.0:
        plot_confusion_matrix(
            res["confusion_matrix"], labels,
            title=f"LSTM Letter Recognition — Confusion Matrix ({args.split})",
            save_path="results/confusion_matrix_test.png",
        )
        plot_per_class_f1(res["per_class"], save_path="results/per_class_f1_test.png")
        print("[eval] figures saved -> results/confusion_matrix_test.png + per_class_f1_test.png")
    else:
        print("[eval] (skipping figures: accuracy at boundary — diagonal/flat chart would not be informative)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
