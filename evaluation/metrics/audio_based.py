"""Audio-input metrics -- pure callables over raw waveforms.

Currently holds the **magnitude axis** (spectral-magnitude distances) and the
**timbre axis** (MFCC distances); the remaining audio axes (loudness, pitch,
perceptual) land here in later build-order slices, each tagged by its
``MetricSpecification.axis`` in the registry.

Pure functions over two **raw mono waveforms** -- the target and the re-rendered
prediction -- sharing the uniform audio call convention::

    compute(target, prediction, *, sample_rate) -> float

All magnitude metrics are lower-is-better and compare the audio **as rendered**
(no level normalization; see D-METRIC-NORM). They never touch a live synthesizer.

``lsd`` and ``spectral_convergence`` follow the anchor definitions in
``paper_repos/preset-gen-vae/utils/audio.py`` (``SimilarityEvaluator``): MAE on
``log10(|STFT|)`` with an ``eps`` floor, and the Frobenius spectral convergence.
``mel_mae`` / ``mel_mse`` compare log-mel spectrograms; ``mss`` is the DDSP-style
multi-scale spectral loss (Engel et al., 2020). The STFT framing for the single-
resolution metrics matches the anchor (``n_fft=1024``, ``hop=256``).

``librosa`` is a hard dependency of the core panel and is imported eagerly.
"""
from __future__ import annotations

import numpy as np
import librosa

# Single-resolution STFT framing -- matches the preset-gen-vae anchor.
N_FFT = 1024
HOP_LENGTH = 256
# Mel filterbank resolution for the log-mel and MFCC metrics.
MEL_BINS = 128
# Number of MFCC coefficients for the timbre metrics.
N_MFCC = 13
# FFT sizes for the multi-scale spectral loss (DDSP), 75% overlap per scale.
MSS_FFT_SIZES = (2048, 1024, 512, 256, 128, 64)


def _magnitude_stft(signal: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """Magnitude STFT ``|STFT(signal)|`` as a ``(frequency, time)`` array."""
    return np.abs(librosa.stft(np.asarray(signal, dtype=np.float32), n_fft=n_fft, hop_length=hop_length))


def _log_mel(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Log-mel spectrogram in dB (``librosa.power_to_db``)."""
    mel_power = librosa.feature.melspectrogram(
        y=np.asarray(signal, dtype=np.float32),
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=MEL_BINS,
    )
    return librosa.power_to_db(mel_power)


def _mfcc(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """MFCCs as a ``(coefficient, time)`` array, sharing the module STFT/mel framing."""
    return librosa.feature.mfcc(
        y=np.asarray(signal, dtype=np.float32),
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=MEL_BINS,
    )


def lsd(target: np.ndarray, prediction: np.ndarray, *, sample_rate: int) -> float:
    """Log-spectral distance: MAE on ``log10(|STFT|)`` (lower is better).

    Anchored to ``SimilarityEvaluator.get_mae_log_stft``: the un-normalized
    magnitude STFTs are floored at ``eps = 1e-4`` (-80 dB) before the log.
    """
    eps = 1e-4
    log_target = np.log10(np.maximum(_magnitude_stft(target), eps))
    log_prediction = np.log10(np.maximum(_magnitude_stft(prediction), eps))
    return float(np.mean(np.abs(log_target - log_prediction)))


def spectral_convergence(target: np.ndarray, prediction: np.ndarray, *, sample_rate: int) -> float:
    """Spectral convergence ``||S_t - S_p||_F / ||S_t||_F`` (lower is better).

    Anchored to ``SimilarityEvaluator.get_spectral_convergence``. Normalizing by
    the *target* magnitude makes this metric order-dependent. Returns ``0.0`` when
    the target is silent (zero denominator).
    """
    magnitude_target = _magnitude_stft(target)
    magnitude_prediction = _magnitude_stft(prediction)
    denominator = float(np.linalg.norm(magnitude_target))
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(magnitude_target - magnitude_prediction) / denominator)


def mel_mae(target: np.ndarray, prediction: np.ndarray, *, sample_rate: int) -> float:
    """Mean absolute error on log-mel spectrograms in dB (lower is better)."""
    return float(np.mean(np.abs(_log_mel(target, sample_rate) - _log_mel(prediction, sample_rate))))


def mel_mse(target: np.ndarray, prediction: np.ndarray, *, sample_rate: int) -> float:
    """Mean squared error on log-mel spectrograms in dB (lower is better)."""
    return float(np.mean((_log_mel(target, sample_rate) - _log_mel(prediction, sample_rate)) ** 2))


def mss(target: np.ndarray, prediction: np.ndarray, *, sample_rate: int) -> float:
    """Multi-scale spectral loss (DDSP; lower is better).

    Sums, over several FFT resolutions, the mean absolute error on the linear
    magnitude spectrogram plus the mean absolute error on its log (Engel et al.,
    2020). A small ``eps`` stabilizes the log on silent bins.
    """
    eps = 1e-7
    total = 0.0
    for n_fft in MSS_FFT_SIZES:
        hop_length = n_fft // 4
        magnitude_target = _magnitude_stft(target, n_fft=n_fft, hop_length=hop_length)
        magnitude_prediction = _magnitude_stft(prediction, n_fft=n_fft, hop_length=hop_length)
        linear_error = float(np.mean(np.abs(magnitude_target - magnitude_prediction)))
        log_error = float(
            np.mean(np.abs(np.log(magnitude_target + eps) - np.log(magnitude_prediction + eps)))
        )
        total += linear_error + log_error
    return total


def mfcc_mae(target: np.ndarray, prediction: np.ndarray, *, sample_rate: int) -> float:
    """Mean absolute error on MFCCs -- the timbre axis (lower is better).

    Anchored to ``SimilarityEvaluator.get_mae_mfcc``: MAE on librosa MFCCs.
    Invariant to a gain common to both inputs (it shifts only the 0th cepstral
    coefficient, equally, so the difference cancels).
    """
    return float(np.mean(np.abs(_mfcc(target, sample_rate) - _mfcc(prediction, sample_rate))))


def mfcc_mse(target: np.ndarray, prediction: np.ndarray, *, sample_rate: int) -> float:
    """Mean squared error on MFCCs -- the timbre axis (lower is better)."""
    return float(np.mean((_mfcc(target, sample_rate) - _mfcc(prediction, sample_rate)) ** 2))
