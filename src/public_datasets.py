"""Download and convert public speech datasets to SpeakEasy format."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from torchaudio.datasets import SPEECHCOMMANDS
from config import SAMPLE_RATE, WORDS_DIR
from audio_preprocess import load_audio, save_audio

FSDD_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest

    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp_dest)
    tmp_dest.replace(dest)
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
        if not dest.exists():
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
    Download and import selected words from Google Speech Commands v0.02 via torchaudio.

    Torchaudio caches the dataset under data/_cache/ and handles download/extraction.
    """

    target_words = target_words or [
        "yes", "no", "up", "down", "left", "right", "on", "off",
        "stop", "go", "cat", "dog", "bird", "happy", "wow",
    ]
    target_set = set(target_words)
    imported_per_word = {word: 0 for word in target_words}

    cache_root = output_dir.parent / "_cache"
    speech_commands_dir = cache_root / "SpeechCommands" / "speech_commands_v0.02"
    if speech_commands_dir.exists() and not any(speech_commands_dir.glob("*/*.wav")):
        shutil.rmtree(speech_commands_dir)

    dataset = SPEECHCOMMANDS(root=str(cache_root), download=True)

    records = []
    for i in range(len(dataset)):
        relpath, sr, label, speaker, utterance_number = dataset.get_metadata(i)
        if label not in target_set:
            continue
        if imported_per_word[label] >= max_per_word:
            continue

        src = Path(dataset._archive) / relpath
        dest = output_dir / speaker / label / f"{speaker}_{utterance_number}.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)

        records.append({
            "audio_path": str(dest),
            "label": label,
            "speaker": speaker,
            "source": "speech_commands",
        })
        imported_per_word[label] += 1

        if all(count >= max_per_word for count in imported_per_word.values()):
            break

    print(f"Imported {len(records)} Speech Commands recordings")
    return records
