"""Audio data augmentation: noise, time stretch, pitch shift, volume scaling."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import random
from typing import Callable

import librosa
import numpy as np

from config import (
    AUGMENT_FACTOR,
    AUGMENT_PITCH_SEMITONES,
    AUGMENT_SNR_DB,
    AUGMENT_SPEED_RANGE,
    AUGMENT_VOLUME_DB,
    SAMPLE_RATE,
)


def add_gaussian_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise at the specified SNR (dB)."""
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def _colored_noise(n_samples: int, exponent: float) -> np.ndarray:
    """Generate colored noise with 1/f^exponent power spectrum.

    exponent=0 → white, exponent=1 → pink, exponent=2 → brown/red.
    Uses the Voss-McCartney algorithm for pink noise and a simple
    cumulative-sum method for arbitrary exponents.
    """
    white = np.random.randn(n_samples).astype(np.float32)
    if exponent == 0:
        return white
    # Frequency-domain method: generate spectrum, apply 1/f^exponent, IFFT
    f = np.fft.rfftfreq(n_samples)
    f[0] = f[1] * 0.1  # avoid DC division by zero
    spectrum = np.fft.rfft(white)
    shaped = spectrum / (f ** (exponent / 2))
    noise = np.fft.irfft(shaped, n=n_samples)
    noise = noise / (noise.std() + 1e-8)
    return noise.astype(np.float32)


def add_colored_noise(audio: np.ndarray, snr_db: float,
                       color: str = "pink") -> np.ndarray:
    """Add pink/brown noise (more realistic than white noise for room environments)."""
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    exponent = {"pink": 1.0, "brown": 2.0}.get(color, 1.0)
    noise = _colored_noise(len(audio), exponent) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def add_room_reverb(audio: np.ndarray, rt60: float = 0.3,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate small-room reverberation via exponential-decay impulse response.

    rt60: reverberation time in seconds (0.1–0.5 for small rooms).
    """
    delay_samples = int(rt60 * sr)
    if delay_samples < 10:
        return audio

    # Build exponential decay IR
    t = np.arange(delay_samples, dtype=np.float32) / sr
    decay = 10 ** (-3 * t / rt60)  # -60 dB at rt60
    ir = np.random.randn(delay_samples).astype(np.float32) * decay

    # Normalize IR energy
    ir = ir / (np.sqrt(np.sum(ir ** 2)) + 1e-8) * 0.3

    # Convolve
    wet = np.convolve(audio, ir, mode="full")[:len(audio)]
    wet = wet.astype(np.float32)

    # Mix dry/wet
    out = audio * 0.7 + wet
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def time_stretch(audio: np.ndarray, rate: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Time-stretch without changing pitch (*rate* < 1 slows down)."""
    return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


def pitch_shift(
    audio: np.ndarray,
    n_steps: float,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Shift pitch by *n_steps* semitones."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps).astype(np.float32)


def volume_scale(audio: np.ndarray, db: float) -> np.ndarray:
    """Scale volume by *db* decibels."""
    factor = 10 ** (db / 20)
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


AUGMENTATIONS: list[tuple[str, Callable[..., np.ndarray]]] = [
    ("noise_10db", lambda a, sr: add_gaussian_noise(a, 10)),
    ("noise_15db", lambda a, sr: add_gaussian_noise(a, 15)),
    ("noise_20db", lambda a, sr: add_gaussian_noise(a, 20)),
    ("noise_30db", lambda a, sr: add_gaussian_noise(a, 30)),
    ("pink_noise_15db", lambda a, sr: add_colored_noise(a, 15, "pink")),
    ("pink_noise_25db", lambda a, sr: add_colored_noise(a, 25, "pink")),
    ("brown_noise_15db", lambda a, sr: add_colored_noise(a, 15, "brown")),
    ("room_reverb_0.2s", lambda a, sr: add_room_reverb(a, 0.2, sr)),
    ("room_reverb_0.4s", lambda a, sr: add_room_reverb(a, 0.4, sr)),
    ("speed_0.85", lambda a, sr: time_stretch(a, 0.85, sr)),
    ("speed_0.9", lambda a, sr: time_stretch(a, 0.9, sr)),
    ("speed_1.1", lambda a, sr: time_stretch(a, 1.1, sr)),
    ("speed_1.15", lambda a, sr: time_stretch(a, 1.15, sr)),
    ("pitch_up_2", lambda a, sr: pitch_shift(a, 2, sr)),
    ("pitch_down_2", lambda a, sr: pitch_shift(a, -2, sr)),
    ("pitch_up_4", lambda a, sr: pitch_shift(a, 4, sr)),
    ("pitch_down_4", lambda a, sr: pitch_shift(a, -4, sr)),
    ("volume_up_3db", lambda a, sr: volume_scale(a, 3)),
    ("volume_down_3db", lambda a, sr: volume_scale(a, -3)),
    ("volume_up_6db", lambda a, sr: volume_scale(a, 6)),
    ("volume_down_6db", lambda a, sr: volume_scale(a, -6)),
]


def augment_random(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    seed: int | None = None,
) -> tuple[np.ndarray, str]:
    """Apply one randomly chosen augmentation."""
    if seed is not None:
        random.seed(seed)
    name, fn = random.choice(AUGMENTATIONS)
    return fn(audio, sr), name


def augment_all(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    factor: int = AUGMENT_FACTOR,
) -> list[tuple[np.ndarray, str]]:
    """
    Generate *factor* augmented copies (including random combinations).
    Cycles through the augmentation pool to reach the target count.
    """
    results: list[tuple[np.ndarray, str]] = []
    pool = list(AUGMENTATIONS)
    random.shuffle(pool)
    for i in range(factor):
        name, fn = pool[i % len(pool)]
        results.append((fn(audio, sr), name))
    return results


def augment_with_params(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    snr_db: float | None = None,
    speed: float | None = None,
    pitch_semitones: float | None = None,
    volume_db: float | None = None,
) -> np.ndarray:
    """Apply specific augmentation parameters (useful for controlled experiments)."""
    out = audio.copy()
    if snr_db is not None:
        out = add_gaussian_noise(out, snr_db)
    if speed is not None:
        out = time_stretch(out, speed, sr)
    if pitch_semitones is not None:
        out = pitch_shift(out, pitch_semitones, sr)
    if volume_db is not None:
        out = volume_scale(out, volume_db)
    return out
