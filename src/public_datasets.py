"""Download and convert public speech datasets to SpeakEasy format."""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from config import LETTERS_DIR, SAMPLE_RATE, WORDS_DIR
from audio_preprocess import load_audio, save_audio

FSDD_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"Downloading {url} …")
    urllib.request.urlretrieve(url, dest)
    return dest


def import_fsdd(output_dir: Path = WORDS_DIR, speaker: str = "fsdd") -> list[dict]:
    """
    Import Free Spoken Digit Dataset (digits 0-9 spoken as words).
    Converts to SpeakEasy layout: output_dir/<speaker>/<label>/<file>.wav
    """
    cache = output_dir.parent / "_cache"
    zip_path = _download(FSDD_URL, cache / "fsdd.zip")

    extract_dir = cache / "fsdd"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache)

    src_root = next(cache.glob("free-spoken-digit-dataset-*"), None)
    if src_root is None:
        print("[WARN] FSDD extraction failed")
        return []

    digit_names = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    }

    records = []
    recordings_dir = src_root / "recordings"
    if not recordings_dir.exists():
        print(f"[WARN] FSDD recordings dir not found: {recordings_dir}")
        return []

    for wav in sorted(recordings_dir.glob("*.wav")):
        # FSDD naming: {digit}_{speaker}_{index}.wav
        parts = wav.stem.split("_")
        if not parts:
            continue
        digit = parts[0]
        label = digit_names.get(digit, digit)
        fsdd_speaker = parts[1] if len(parts) > 1 else speaker

        dest = output_dir / fsdd_speaker / label / wav.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        audio, sr = load_audio(str(wav))
        save_audio(str(dest), audio, SAMPLE_RATE)

        records.append({
            "audio_path": str(dest),
            "label": label,
            "speaker": fsdd_speaker,
            "source": "fsdd",
        })

    print(f"Imported {len(records)} FSDD recordings")
    return records


def import_speech_commands(
    output_dir: Path = WORDS_DIR,
    target_words: list[str] | None = None,
    max_per_word: int = 50,
) -> list[dict]:
    """
    Import selected words from Google Speech Commands v0.02.
    Requires manual download from:
    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    Place the extracted folder at data/_cache/speech_commands/
    """
    cache = output_dir.parent / "_cache" / "speech_commands"
    if not cache.exists():
        print(
            "[INFO] Speech Commands not found. Download and extract to:\n"
            f"  {cache}\n"
            "  URL: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        )
        return []

    target_words = target_words or [
        "yes", "no", "up", "down", "left", "right", "on", "off",
        "stop", "go", "cat", "dog", "bird", "happy", "wow",
    ]
    target_set = set(target_words)

    records = []
    for word_dir in sorted(cache.iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith("_"):
            continue
        if word_dir.name not in target_set:
            continue

        for i, wav in enumerate(sorted(word_dir.glob("*.wav"))):
            if i >= max_per_word:
                break
            # Filename encodes speaker hash: speaker_hash_n.wav
            speaker = wav.stem.rsplit("_", 2)[0]
            dest = output_dir / speaker / word_dir.name / wav.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(wav, dest)
            records.append({
                "audio_path": str(dest),
                "label": word_dir.name,
                "speaker": speaker,
                "source": "speech_commands",
            })

    print(f"Imported {len(records)} Speech Commands recordings")
    return records
