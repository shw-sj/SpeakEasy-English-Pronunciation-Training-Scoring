"""Build a leave-one-human-out split and export frame-level MFCC sequences
for LSTM training, reusing the already-processed WAV files (no re-augmentation).

Rationale
---------
The original split placed all three human speakers in the training set and used
only gTTS voices for validation/test. Because the held-out gTTS "speakers" are
merely TLD variants of the training gTTS voices, the test accuracy hit ~100% but
said nothing about generalization to real, unseen humans.

This script rebuilds the split so that the test set is a *human speaker the model
never saw*, giving a meaningful generalization number:

    test  = SWJ  (human, original recordings only)   -> the headline number
    val   = one gTTS speaker (original only)          -> early-stopping signal
    train = the two remaining humans + 9 gTTS voices  -> all augmentations

It reads the authoritative per-utterance records (speaker/label/augmentation)
from ``letters_splits.json`` and only remaps the stale absolute ``processed_path``
onto the current machine, then reuses ``export_frame_sequences`` to write the
``letters_{split}_sequences.npz`` archives that ``lstm_train.py`` consumes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from config import PROCESSED_DIR, MANIFESTS_DIR, FEATURES_DIR
from prepare_data import export_frame_sequences

# ── Leave-one-human-out configuration ───────────────────────────────────────
TEST_HUMAN = "SWJ"                 # unseen human -> test set (original only)
VAL_SPEAKER = "tts_gtts_01_com"    # held-out voice -> validation (original only)
# Everything else (LJQ, Lcp + remaining gTTS) goes to train with augmentations.


def remap_path(processed_path: str) -> Path:
    """Rebuild a stale absolute processed_path under the current PROCESSED_DIR."""
    parts = [p for p in re.split(r"[\\/]", processed_path) if p]
    idx = max(i for i, p in enumerate(parts) if p == "processed")
    rel = parts[idx + 1:]  # letters/<speaker>/<label>/<file>.wav
    return PROCESSED_DIR.joinpath(*rel)


def build_splits() -> dict[str, list[dict]]:
    manifest = json.loads(
        (MANIFESTS_DIR / "letters_splits.json").read_text(encoding="utf-8")
    )
    # Flatten the old splits back into one pool of records.
    pool = manifest["train"] + manifest["val"] + manifest["test"]

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in pool:
        rec = dict(r)
        rec["processed_path"] = str(remap_path(r["processed_path"]))
        spk = rec["speaker"]
        is_original = rec.get("augmentation") == "original"

        if spk == TEST_HUMAN:
            if is_original:
                splits["test"].append(rec)          # unseen human, clean only
        elif spk == VAL_SPEAKER:
            if is_original:
                splits["val"].append(rec)           # early-stop signal, clean only
        else:
            splits["train"].append(rec)             # everyone else, all augmentations

    return splits


def main() -> int:
    splits = build_splits()
    for name in ("train", "val", "test"):
        recs = splits[name]
        speakers = sorted({r["speaker"] for r in recs})
        print(f"[split] {name}: {len(recs)} records  speakers={speakers}")

    missing = [
        r["processed_path"]
        for recs in splits.values()
        for r in recs
        if not Path(r["processed_path"]).exists()
    ]
    if missing:
        print(f"[WARN] {len(missing)} processed files not found on disk, e.g. {missing[0]}")

    print("\n--- Exporting frame-level MFCC sequences ---")
    export_frame_sequences(splits, FEATURES_DIR)
    print(f"\n[done] sequences written under {FEATURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
