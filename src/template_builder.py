"""Build standard pronunciation templates using TTS and extract MFCC features."""

from __future__ import annotations

import sys
from pathlib import Path

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
    LETTERS,
    SAMPLE_RATE,
    TEMPLATES_DIR,
)

from gtts import gTTS

def generate_tts_gtts(text: str, output_path: Path, lang: str = "en") -> bool:
    """Generate speech using gTTS (requires internet)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts = gTTS(text=text, lang=lang)
    mp3_path = output_path.with_suffix(".mp3")
    tts.save(str(mp3_path))

    try:
        import librosa
        audio, sr = librosa.load(str(mp3_path), sr=SAMPLE_RATE, mono=True)
        save_audio(str(output_path), audio, SAMPLE_RATE)
        mp3_path.unlink(missing_ok=True)
        return True
    except Exception:
        return False


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
    args = parser.parse_args()

    templates = build_templates(args.labels)
    save_templates(templates, fmt=args.format)


if __name__ == "__main__":
    main()
