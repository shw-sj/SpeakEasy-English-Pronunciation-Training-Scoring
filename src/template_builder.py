"""Build standard pronunciation templates using TTS and extract MFCC features."""

from __future__ import annotations
import librosa
import sys
from pathlib import Path
import re

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import pickle

import numpy as np

from audio_feature import extract_features
from audio_preprocess import load_audio, save_audio
from config import (
    DEFAULT_WORDS,
    LETTERS_DIR,
    LETTERS,
    SAMPLE_RATE,
    TEMPLATES_DIR,
)

from gtts import gTTS

GTTS_VOICE_TLDS = [
    "com",
    "co.uk",
    "com.au",
    "ca",
    "co.in",
    "ie",
    "co.za",
    "com.ng",
    "com.sg",
    "co.nz",
]


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_").lower()

def generate_tts_gtts(
    text: str,
    output_path: Path,
    lang: str = "en",
    tld: str = "com",
) -> bool:
    """Generate speech using gTTS (requires internet)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts = gTTS(text=text, lang=lang, tld=tld)
    mp3_path = output_path.with_suffix(".mp3")
    tts.save(str(mp3_path))

    audio, sr = librosa.load(str(mp3_path), sr=SAMPLE_RATE, mono=True)
    save_audio(str(output_path), audio, SAMPLE_RATE)
    mp3_path.unlink(missing_ok=True)
    return True


def generate_letter_dataset_tts(
    output_dir: Path = LETTERS_DIR,
    voice_count: int = 5,
    repeats: int = 10,
    overwrite: bool = False,
) -> list[dict]:
    """
    Generate A-Z letter recordings with multiple gTTS voice variants.

    Output layout matches the training pipeline:
    data/raw/letters/<speaker>/<label>/<file>.wav
    """
    if voice_count > len(GTTS_VOICE_TLDS):
        raise RuntimeError(f"Requested {voice_count} voices, but only {len(GTTS_VOICE_TLDS)} gTTS variants are configured.")

    selected_voices = GTTS_VOICE_TLDS[:voice_count]
    records = []

    output_dir.mkdir(parents=True, exist_ok=True)
    for voice_index, tld in enumerate(selected_voices, start=1):
        speaker = f"tts_gtts_{voice_index:02d}_{_safe_name(tld)}"
        for label in LETTERS:
            label_dir = output_dir / speaker / label
            label_dir.mkdir(parents=True, exist_ok=True)

            for repeat in range(1, repeats + 1):
                wav_path = label_dir / f"{label}_{speaker}_{repeat:02d}.wav"
                if wav_path.exists() and not overwrite:
                    records.append({
                        "audio_path": str(wav_path),
                        "label": label,
                        "speaker": speaker,
                        "source": "gtts",
                    })
                    continue

                generate_tts_gtts(label, wav_path, tld=tld)
                records.append({
                    "audio_path": str(wav_path),
                    "label": label,
                    "speaker": speaker,
                    "source": "gtts",
                })

        print(f"Generated {speaker}: {len(LETTERS) * repeats} files")

    print(f"Generated {len(records)} letter TTS recordings -> {output_dir}")
    return records


def generate_template_audio(
    label: str,
    output_dir: Path,
) -> Path | None:
    """Generate a standard pronunciation WAV for *label*."""
    output_path = output_dir / f"{label.lower()}.wav"
    if output_path.exists():
        return output_path

    ok = generate_tts_gtts(label, output_path)

    if not ok:
        print(f"  [WARN] Failed to generate TTS for '{label}'")
        return None
    return output_path


def build_templates(
    labels: list[str] | None = None,
    output_dir: Path = TEMPLATES_DIR,
) -> dict[str, np.ndarray]:
    """
    Generate standard pronunciation audio and extract MFCC template features.
    Returns {label: feature_vector}.
    """
    labels = labels or (LETTERS + DEFAULT_WORDS)
    audio_dir = output_dir / "audio"
    templates: dict[str, np.ndarray] = {}

    print(f"Building templates for {len(labels)} labels …")
    for label in labels:
        path = generate_template_audio(label, audio_dir)
        if path is None:
            continue
        audio, sr = load_audio(str(path))
        feat = extract_features(audio, sr)
        templates[label] = feat
        print(f"  {label}: dim={feat.shape[0]}")

    return templates


def save_templates(
    templates: dict[str, np.ndarray],
    output_dir: Path = TEMPLATES_DIR,
    fmt: str = "both",
) -> None:
    """Save templates as JSON and/or pickle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable = {k: v.tolist() for k, v in templates.items()}

    if fmt in ("json", "both"):
        json_path = output_dir / "templates.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"Templates (JSON) → {json_path}")

    if fmt in ("pkl", "both"):
        pkl_path = output_dir / "templates.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(templates, f)
        print(f"Templates (PKL)  → {pkl_path}")


def load_templates(path: Path) -> dict[str, np.ndarray]:
    """Load templates from JSON or pickle file."""
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {k: np.array(v, dtype=np.float32) for k, v in data.items()}
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pronunciation templates")

    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="Specific labels (default: all letters + words)",
    )
    parser.add_argument(
        "--format", choices=["json", "pkl", "both"], default="both",
    )
    parser.add_argument(
        "--generate-letter-dataset", action="store_true",
        help="Generate A-Z raw letter dataset with gTTS voice variants",
    )
    parser.add_argument(
        "--tts-voices", type=int, default=10,
        help=f"Number of gTTS voice variants to use, max {len(GTTS_VOICE_TLDS)}",
    )
    parser.add_argument("--tts-repeats", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.generate_letter_dataset:
        generate_letter_dataset_tts(
            voice_count=args.tts_voices,
            repeats=args.tts_repeats,
            overwrite=args.overwrite,
        )
        return

    templates = build_templates(args.labels)
    save_templates(templates, fmt=args.format)


if __name__ == "__main__":
    main()
