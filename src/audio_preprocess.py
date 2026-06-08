"""Audio preprocessing pipeline: resample, normalize, VAD, pre-emphasis, framing."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from config import (
    FRAME_LENGTH_MS,
    FRAME_SHIFT_MS,
    PRE_EMPHASIS_COEF,
    SAMPLE_RATE,
    VAD_ENERGY_THRESHOLD,
    VAD_FRAME_MS,
    VAD_HOP_MS,
    VAD_ZCR_THRESHOLD,
)


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load a WAV file and resample to *target_sr* if needed."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = resample_to(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def resample_to(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling."""
    if orig_sr == target_sr:
        return audio
    gcd = np.gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // gcd, orig_sr // gcd).astype(np.float32)


def normalize_amplitude(audio: np.ndarray) -> np.ndarray:
    """Scale audio to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return (audio / peak).astype(np.float32)


def _frame_signal(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    n_frames = max(1, 1 + (len(signal) - frame_len) // hop)
    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frames[i] = signal[start : start + frame_len]
    return frames


def _zero_crossing_rate(frame: np.ndarray) -> float:
    signs = np.sign(frame)
    signs[signs == 0] = 1
    return np.sum(np.abs(np.diff(signs))) / (2 * len(frame))


def trim_silence_vad(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    energy_threshold: float = VAD_ENERGY_THRESHOLD,
    zcr_threshold: float = VAD_ZCR_THRESHOLD,
    frame_ms: int = VAD_FRAME_MS,
    hop_ms: int = VAD_HOP_MS,
) -> np.ndarray:
    """Remove leading/trailing silence using energy + ZCR dual-threshold VAD."""
    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(audio) < frame_len:
        return audio

    frames = _frame_signal(audio, frame_len, hop)
    voiced = []
    for frame in frames:
        energy = np.mean(frame ** 2)
        zcr = _zero_crossing_rate(frame)
        voiced.append(energy > energy_threshold or zcr > zcr_threshold)

    voiced = np.array(voiced)
    if not voiced.any():
        return audio

    first = int(np.argmax(voiced))
    last = len(voiced) - int(np.argmax(voiced[::-1])) - 1
    start_sample = first * hop
    end_sample = min(len(audio), (last + 1) * hop + frame_len)
    return audio[start_sample:end_sample]


def pre_emphasis(audio: np.ndarray, coef: float = PRE_EMPHASIS_COEF) -> np.ndarray:
    """Apply pre-emphasis filter: y[n] = x[n] - coef * x[n-1]."""
    return np.append(audio[0], audio[1:] - coef * audio[:-1]).astype(np.float32)


def hamming_window(length: int) -> np.ndarray:
    return np.hamming(length).astype(np.float32)


def frame_signal(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: int = FRAME_LENGTH_MS,
    hop_ms: int = FRAME_SHIFT_MS,
) -> np.ndarray:
    """Split signal into overlapping frames and apply Hamming window."""
    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    frames = _frame_signal(audio, frame_len, hop)
    window = hamming_window(frame_len)
    return frames * window


def preprocess_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    apply_vad: bool = True,
) -> np.ndarray:
    """Full preprocessing: normalize → VAD → pre-emphasis."""
    audio = normalize_amplitude(audio)
    if apply_vad:
        audio = trim_silence_vad(audio, sr)
    audio = pre_emphasis(audio)
    return audio


def preprocess_file(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load and preprocess a single audio file."""
    audio, sr = load_audio(path, target_sr)
    return preprocess_audio(audio, sr)


def save_audio(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save audio as 16-bit PCM WAV."""
    sf.write(path, audio, sr, subtype="PCM_16")
