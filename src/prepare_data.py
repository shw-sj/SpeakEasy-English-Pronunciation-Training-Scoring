"""Dataset preparation: preprocessing, augmentation, feature extraction, and splitting."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from audio_feature import extract_features, extract_mfcc_sequence, feature_dim
from audio_preprocess import load_audio, preprocess_audio, save_audio
from config import (
    AUGMENT_FACTOR,
    FEATURES_DIR,
    LETTERS_DIR,
    MANIFESTS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    SAMPLE_RATE,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    WORDS_DIR,
)
from data_augmentation import augment_all
from public_datasets import import_fsdd, import_speech_commands


def scan_audio_files(root: Path) -> list[dict]:
    """
    Scan directory tree for WAV files.
    Expected layout: root/<speaker>/<label>/<file>.wav
    """
    records = []
    for wav in sorted(root.rglob("*.wav")):
        parts = wav.relative_to(root).parts
        if len(parts) < 3:
            continue
        speaker, label = parts[0], parts[1]
        records.append({
            "audio_path": str(wav),
            "label": label.upper() if len(label) == 1 else label.lower(),
            "speaker": speaker,
        })
    return records


def speaker_independent_split(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> dict[str, list[dict]]:
    """Split records by speaker so no speaker appears in multiple splits.

    Falls back to stratified sample-level split when there is only 1 speaker,
    because GroupShuffleSplit requires ≥2 groups to produce non-empty splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    speakers = sorted({r["speaker"] for r in records})

    # ── Fallback: single speaker → split by samples (stratified) ──
    if len(speakers) < 2:
        print(f"  [INFO] Only {len(speakers)} speaker(s), using sample-level "
              f"stratified split instead of speaker-independent split.")
        labels = [r["label"] for r in records]

        # 60% train, 40% rest
        train, rest, _, rest_labels = train_test_split(
            records, labels,
            test_size=1 - train_ratio,
            random_state=seed,
            stratify=labels,
        )
        # From the 40% rest: half val, half test (i.e. 20% / 20%)
        val_fraction = val_ratio / (val_ratio + test_ratio)  # 0.2 / 0.4 = 0.5
        val, test = train_test_split(
            rest,
            test_size=1 - val_fraction,
            random_state=seed,
            stratify=rest_labels,
        )
        return {"train": train, "val": val, "test": test}

    # ── Multi-speaker: GroupShuffleSplit ──
    groups = np.array([r["speaker"] for r in records])

    gss1 = GroupShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed)
    train_idx, rest_idx = next(gss1.split(records, groups=groups))

    rest_records = [records[i] for i in rest_idx]
    rest_groups = groups[rest_idx]
    val_fraction = val_ratio / (val_ratio + test_ratio)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=1 - val_fraction, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest_records, groups=rest_groups))

    return {
        "train": [records[i] for i in train_idx],
        "val": [rest_records[i] for i in val_idx],
        "test": [rest_records[i] for i in test_idx],
    }


def process_record(
    record: dict,
    processed_dir: Path,
    augment: bool = False,
    augment_factor: int = AUGMENT_FACTOR,
) -> list[dict]:
    """Preprocess one audio file, optionally augment, and extract features."""
    audio, sr = load_audio(record["audio_path"])
    audio = preprocess_audio(audio, sr)

    rel = Path(record["audio_path"]).stem
    speaker = record["speaker"]
    label = record["label"]
    out_dir = processed_dir / speaker / label
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = out_dir / f"{rel}.wav"
    save_audio(str(base_path), audio, sr)

    results = [{
        **record,
        "processed_path": str(base_path),
        "features": extract_features(audio, sr, preprocess=False),
        "augmentation": "original",
    }]

    if augment:
        for aug_audio, aug_name in augment_all(audio, sr, augment_factor):
            aug_path = out_dir / f"{rel}_{aug_name}.wav"
            save_audio(str(aug_path), aug_audio, sr)
            results.append({
                **record,
                "processed_path": str(aug_path),
                "features": extract_features(aug_audio, sr, preprocess=False),
                "augmentation": aug_name,
            })

    return results


def prepare_dataset(
    dataset_type: str = "letters",
    augment: bool = True,
    augment_factor: int = AUGMENT_FACTOR,
) -> dict[str, list[dict]]:
    """Full pipeline for one dataset type ('letters' or 'words')."""
    root = LETTERS_DIR if dataset_type == "letters" else WORDS_DIR
    if not root.exists():
        Path(root).mkdir(parents=True, exist_ok=True)

    records = scan_audio_files(root)
    if not records:
        print(f"[WARN] No audio files found in {root}")
        return {}

    print(f"Found {len(records)} files in {dataset_type} dataset")
    splits = speaker_independent_split(records)

    processed_dir = PROCESSED_DIR / dataset_type
    all_processed: dict[str, list[dict]] = {}

    for split_name, split_records in splits.items():
        print(f"  Processing {split_name}: {len(split_records)} files …")
        processed = []
        for rec in split_records:
            processed.extend(
                process_record(rec, processed_dir, augment, augment_factor)
            )
        all_processed[split_name] = processed

    return all_processed


def export_features(
    splits: dict[str, list[dict]],
    output_dir: Path,
    dataset_type: str,
    export_sequences: bool = False,
) -> None:
    """Export features as .npy arrays and a manifest .csv.

    When ``export_sequences`` is True, also exports per-frame MFCC
    sequences as .npz archives suitable for sequential model training.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, records in splits.items():
        if not records:
            continue

        features = np.stack([r["features"] for r in records])
        labels = [r["label"] for r in records]
        paths = [r.get("processed_path", r["audio_path"]) for r in records]

        npy_path = output_dir / f"{dataset_type}_{split_name}_features.npy"
        np.save(npy_path, features)

        csv_path = output_dir / f"{dataset_type}_{split_name}_manifest.csv"
        df = pd.DataFrame({
            "audio_path": paths,
            "label": labels,
            "speaker": [r["speaker"] for r in records],
            "augmentation": [r.get("augmentation", "original") for r in records],
        })
        df.to_csv(csv_path, index=False)

        print(f"  {split_name}: {features.shape} → {npy_path.name}, {csv_path.name}")

    if export_sequences:
        print("\n--- Exporting frame-level MFCC sequences ---")
        export_frame_sequences(splits, output_dir, dataset_type)


def export_frame_sequences(
    splits: dict[str, list[dict]],
    output_dir: Path,
    dataset_type: str,
) -> None:
    """
    Export per-frame MFCC sequences as .npz archives for sequential model training.

    Unlike ``export_features`` which saves aggregated (78,) vectors, this
    preserves the full temporal structure ``(T, 39)`` per utterance so
    sequential models can capture genuine temporal dynamics.  A companion ``_seq_meta.npy``
    stores the frame count of every sample, enabling fast 95‑th‑percentile
    computation without loading the full sequence array.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, records in splits.items():
        if not records:
            continue

        sequences = []          # ragged arrays, each (T_i, 39)
        frame_counts = []       # T_i per sample
        labels = []
        paths = []

        for r in records:
            proc_path = r.get("processed_path", r["audio_path"])
            try:
                audio, sr = load_audio(proc_path)
                # audio was already preprocessed → skip redundant VAD/norm
                seq = extract_mfcc_sequence(
                    audio, sr, preprocess=False, max_frames=None,
                )  # (T, 39) variable length
            except Exception as e:
                print(f"  [WARN] skipping {proc_path}: {e}")
                continue

            sequences.append(seq)
            frame_counts.append(seq.shape[0])
            labels.append(r["label"])
            paths.append(proc_path)

        if not sequences:
            print(f"  [WARN] No valid sequences for {split_name}")
            continue

        # Save as compressed archive — one key per sample
        npz_path = output_dir / f"{dataset_type}_{split_name}_sequences.npz"
        arrays = {f"seq_{i:06d}": seq for i, seq in enumerate(sequences)}
        np.savez_compressed(npz_path, **arrays)

        # Save frame-count metadata for fast lookup
        meta_path = output_dir / f"{dataset_type}_{split_name}_seq_meta.npy"
        np.save(meta_path, np.array(frame_counts, dtype=np.int32))

        # Save companion manifest (same columns as the aggregated one)
        csv_path = output_dir / f"{dataset_type}_{split_name}_seq_manifest.csv"
        pd.DataFrame({
            "audio_path": paths,
            "label": labels,
            "speaker": [r["speaker"] for r in records],
            "augmentation": [r.get("augmentation", "original") for r in records],
            "num_frames": frame_counts,
        }).to_csv(csv_path, index=False)

        total_frames = sum(frame_counts)
        avg_frames = total_frames / len(sequences)
        print(f"  {split_name}: {len(sequences)} seqs, "
              f"avg {avg_frames:.0f} frames → {npz_path.name}, {meta_path.name}")


def save_split_manifest(
    splits: dict[str, list[dict]],
    dataset_type: str,
) -> None:
    """Save split metadata as JSON."""
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split_name, records in splits.items():
        manifest[split_name] = [
            {k: v for k, v in r.items() if k != "features"}
            for r in records
        ]
    path = MANIFESTS_DIR / f"{dataset_type}_splits.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Split manifest → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SpeakEasy datasets")
    parser.add_argument(
        "--dataset", choices=["letters", "words", "both"], default="both",
    )
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--augment-factor", type=int, default=AUGMENT_FACTOR)
    parser.add_argument("--export-sequences", action="store_true",
                        help="Also export per-frame MFCC sequences (.npz) for sequential model training")
    parser.add_argument(
        "--import-public", choices=["fsdd", "speech_commands", "all"], default="all",
        help="Import public datasets before processing",
    )
    args = parser.parse_args()

    if args.import_public:
        if args.import_public in ("fsdd", "all"):
            import_fsdd()
        if args.import_public in ("speech_commands", "all"):
            import_speech_commands()

    datasets = (
        ["letters", "words"] if args.dataset == "both" else [args.dataset]
    )

    print(f"Feature vector dimension: {feature_dim()}")

    for ds in datasets:
        print(f"\n{'='*50}\nPreparing {ds} dataset\n{'='*50}")
        splits = prepare_dataset(
            dataset_type=ds,
            augment=not args.no_augment,
            augment_factor=args.augment_factor,
        )
        if not splits:
            continue
        export_features(splits, FEATURES_DIR, ds,
                        export_sequences=args.export_sequences)
        save_split_manifest(splits, ds)

    print("\nDone.")


if __name__ == "__main__":
    main()
