"""Audio recording tool for collecting letter and word pronunciation samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import (
    CHANNELS,
    DEFAULT_WORDS,
    LETTERS,
    LETTERS_DIR,
    SAMPLE_RATE,
    WORDS_DIR,
)


def record_audio(
    duration: float = 3.0,
    sr: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> np.ndarray:
    """Record *duration* seconds of audio from the default microphone."""
    print(f"Recording {duration:.1f}s (speak now)")
    audio = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=channels,
        dtype="float32",
    )
    sd.wait()
    if channels > 1:
        audio = audio.mean(axis=1)
    return audio.flatten()


def save_recording(
    audio: np.ndarray,
    path: Path,
    sr: int = SAMPLE_RATE,
) -> None:
    """Save a recording as 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    print(f"  Saved -> {path}")


def _next_index(directory: Path, prefix: str) -> int:
    existing = list(directory.glob(f"{prefix}_*.wav"))
    if not existing:
        return 1

    indices = []
    for file_path in existing:
        try:
            indices.append(int(file_path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return max(indices, default=0) + 1


def collect_letters(
    speaker: str,
    repeats: int = 10,
    duration: float = 2.0,
    output_dir: Path = LETTERS_DIR,
) -> None:
    """Interactive letter collection: record each letter *repeats* times."""
    speaker_dir = output_dir / speaker
    print(f"\n=== Letter collection for speaker '{speaker}' ===")
    print(f"Each letter will be recorded {repeats} times.\n")

    for letter in LETTERS:
        for rep in range(1, repeats + 1):
            input(f"  [{letter}] repetition {rep}/{repeats} - press Enter to record ")
            audio = record_audio(duration)
            idx = _next_index(speaker_dir / letter, letter)
            filename = f"{letter}_{idx:04d}.wav"
            path = speaker_dir / letter / filename
            save_recording(audio, path)


def collect_words(
    speaker: str,
    words: list[str] | None = None,
    repeats: int = 5,
    duration: float = 3.0,
    output_dir: Path = WORDS_DIR,
) -> None:
    """Interactive word collection: record each word *repeats* times."""
    words = words or DEFAULT_WORDS
    speaker_dir = output_dir / speaker
    print(f"\n=== Word collection for speaker '{speaker}' ===")
    print(f"{len(words)} words x {repeats} repetitions\n")

    for word in words:
        for rep in range(1, repeats + 1):
            input(f"  [{word}] repetition {rep}/{repeats} - press Enter to record ")
            audio = record_audio(duration)
            idx = _next_index(speaker_dir / word, word)
            filename = f"{word}_{idx:04d}.wav"
            path = speaker_dir / word / filename
            save_recording(audio, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="SpeakEasy audio collector")
    parser.add_argument("--speaker", required=True, help="Speaker identifier")
    parser.add_argument(
        "--mode", choices=["letters", "words", "both"], default="both",
    )
    parser.add_argument("--letter-repeats", type=int, default=10)
    parser.add_argument("--word-repeats", type=int, default=5)
    parser.add_argument(
        "--duration",
        type=float,
        default=1,
        help="Max recording duration (seconds)",
    )
    args = parser.parse_args()

    if args.mode in ("letters", "both"):
        collect_letters(
            args.speaker, args.letter_repeats, args.duration,
        )
    if args.mode in ("words", "both"):
        collect_words(
            args.speaker, repeats=args.word_repeats, duration=args.duration,
        )


if __name__ == "__main__":
    main()
