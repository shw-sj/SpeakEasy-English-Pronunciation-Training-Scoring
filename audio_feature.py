"""MFCC feature extraction: Mel filterbank, DCT, delta features, aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.fft import rfft
from scipy.fftpack import dct

from audio_preprocess import frame_signal, preprocess_audio
from config import (
    FMAX,
    FMIN,
    FRAME_LENGTH_MS,
    FRAME_SHIFT_MS,
    N_FFT,
    N_MEL_FILTERS,
    N_MFCC,
    SAMPLE_RATE,
)


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    n_filters: int = N_MEL_FILTERS,
    n_fft: int = N_FFT,
    sr: int = SAMPLE_RATE,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Create triangular Mel filterbank matrix (n_filters × n_fft//2+1)."""
    n_freqs = n_fft // 2 + 1
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_filters, n_freqs), dtype=np.float32)
    for i in range(n_filters):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(left, center):
            if center != left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                fb[i, j] = (right - j) / (right - center)
    return fb


def power_spectrum(frames: np.ndarray, n_fft: int = N_FFT) -> np.ndarray:
    """Compute power spectrum for each frame via FFT."""
    spectrum = np.abs(rfft(frames, n=n_fft, axis=1)) ** 2
    return spectrum.astype(np.float32)


def compute_mfcc_frames(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_mel: int = N_MEL_FILTERS,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """Extract per-frame MFCC coefficients (n_frames × n_mfcc)."""
    frames = frame_signal(audio, sr, FRAME_LENGTH_MS, FRAME_SHIFT_MS)
    power = power_spectrum(frames, n_fft)
    mel_fb = build_mel_filterbank(n_mel, n_fft, sr)
    mel_energy = np.maximum(power @ mel_fb.T, 1e-10)
    log_mel = np.log(mel_energy)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return mfcc.astype(np.float32)


def compute_delta(features: np.ndarray, width: int = 2) -> np.ndarray:
    """Compute delta (1st derivative) features along the time axis."""
    n_frames, n_coeff = features.shape
    delta = np.zeros_like(features)
    denom = 2 * sum(i ** 2 for i in range(1, width + 1))
    for t in range(n_frames):
        for i in range(1, width + 1):
            prev_idx = max(0, t - i)
            next_idx = min(n_frames - 1, t + i)
            delta[t] += i * (features[next_idx] - features[prev_idx])
    return (delta / denom).astype(np.float32)


def aggregate_features(mfcc: np.ndarray) -> np.ndarray:
    """
    Aggregate frame-level MFCC + Δ + ΔΔ into a fixed-length vector.
    Returns mean and std for each coefficient set → n_mfcc × 3 × 2 dims.
    """
    delta = compute_delta(mfcc)
    delta2 = compute_delta(delta)

    parts = []
    for feat in (mfcc, delta, delta2):
        parts.append(feat.mean(axis=0))
        parts.append(feat.std(axis=0))
    return np.concatenate(parts).astype(np.float32)


def aggregate_features_rich(mfcc: np.ndarray) -> np.ndarray:
    """
    Aggregate frame-level MFCC + Δ + ΔΔ into a richer fixed-length vector.

    Computes 4 statistics per coefficient: mean, std, min, max.
    Total dimension: n_mfcc × 3 (static/Δ/ΔΔ) × 4 (stats) = 156 dims.

    The additional min/max statistics capture extreme values that mean+std
    alone miss — useful for distinguishing letters with different articulatory dynamics.
    """
    delta = compute_delta(mfcc)
    delta2 = compute_delta(delta)

    parts = []
    for feat in (mfcc, delta, delta2):
        parts.append(feat.mean(axis=0))
        parts.append(feat.std(axis=0))
        parts.append(feat.min(axis=0))
        parts.append(feat.max(axis=0))
    return np.concatenate(parts).astype(np.float32)


def extract_mfcc_sequence(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    max_frames: int | None = None,
    preprocess: bool = True,
) -> np.ndarray:
    """
    Extract frame-level MFCC + Δ + ΔΔ as a time sequence.

    Unlike ``extract_features`` which aggregates across all frames into a
    single 78‑dim vector, this returns the full per‑frame feature matrix
    so that sequential models can capture genuine temporal structure.

    Parameters
    ----------
    audio : ndarray
        Raw audio samples (float32).
    sr : int
        Sample rate (default 16000).
    n_mfcc : int
        Number of MFCC coefficients per frame (default 13).
    max_frames : int or None
        If provided, longer sequences are truncated and shorter ones are
        zero-padded so the output always has exactly ``max_frames`` rows.
    preprocess : bool
        Whether to apply VAD, normalisation, and pre-emphasis before
        feature extraction.

    Returns
    -------
    features : ndarray (float32)
        Shape ``(n_frames, n_mfcc * 3)`` if ``max_frames`` is None,
        or ``(max_frames, n_mfcc * 3)`` otherwise.
        The 3 channels per frame are: static MFCC, Δ, ΔΔ.
    """
    if preprocess:
        audio = preprocess_audio(audio, sr)

    mfcc = compute_mfcc_frames(audio, sr, n_mfcc)          # (T, 13)
    delta = compute_delta(mfcc)                             # (T, 13)
    delta2 = compute_delta(delta)                           # (T, 13)

    features = np.concatenate([mfcc, delta, delta2], axis=1)  # (T, 39)

    if max_frames is not None:
        n_frames = features.shape[0]
        if n_frames > max_frames:
            features = features[:max_frames, :]
        elif n_frames < max_frames:
            pad = np.zeros(
                (max_frames - n_frames, features.shape[1]),
                dtype=np.float32,
            )
            features = np.concatenate([features, pad], axis=0)

    return features.astype(np.float32)


def extract_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    preprocess: bool = True,
    rich: bool = True,
) -> np.ndarray:
    """End-to-end feature extraction from raw audio.

    Parameters
    ----------
    rich : bool
        If True, use richer aggregation (4 stats → 156 dims).
        Default True (156-dim rich features).
    """
    if preprocess:
        audio = preprocess_audio(audio, sr)
    mfcc = compute_mfcc_frames(audio, sr, n_mfcc)
    if rich:
        return aggregate_features_rich(mfcc)
    return aggregate_features(mfcc)


def extract_features_from_file(
    path: str,
    n_mfcc: int = N_MFCC,
    rich: bool = False,
) -> np.ndarray:
    """Extract aggregated MFCC features from an audio file.

    Parameters
    ----------
    rich : bool
        If True, use richer aggregation (4 stats → 156 dims).
    """
    from audio_preprocess import load_audio

    audio, sr = load_audio(path)
    return extract_features(audio, sr, n_mfcc, rich=rich)


def feature_dim(n_mfcc: int = N_MFCC) -> int:
    """Return the standard aggregated feature vector dimension (78)."""
    return n_mfcc * 3 * 2


def feature_dim_rich(n_mfcc: int = N_MFCC) -> int:
    """Return the rich aggregated feature vector dimension (156)."""
    return n_mfcc * 3 * 4


# Map feature mode names to their dim functions and aggregators
FEATURE_MODES = {
    "standard": (feature_dim, aggregate_features),
    "rich": (feature_dim_rich, aggregate_features_rich),
}
