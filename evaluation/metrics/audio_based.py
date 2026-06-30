"""Audio-input metrics -- pure callables over raw waveforms.

Currently holds the **magnitude axis** (spectral-magnitude distances), the
**timbre axis** (MFCC distances), the **loudness axis** (level and loudness-
contour distances), and the **pitch axis** (F0-contour distance), each tagged by
its ``MetricSpecification.axis`` in the registry. The remaining audio axis --
perceptual (embedding) metrics -- is out of scope (possible future work; see
``D-METRIC-PERCEPTUAL`` in ``docs/DECISIONS.md``).

Pure functions over two **raw mono waveforms** -- the target and the re-rendered
prediction -- sharing the uniform audio call convention::

    compute(target, prediction, *, sample_rate) -> float

Every metric here is lower-is-better and compares the audio **as rendered** (no
level normalization; see D-METRIC-NORM). They never touch a live synthesizer.
The loudness metrics are deliberately *level-sensitive*: loudness-matching the
prediction to the target first would cancel exactly what they measure, which is
one reason the panel compares raw audio throughout.

``lsd`` and ``spectral_convergence`` follow the anchor definitions in
``paper_repos/preset-gen-vae/utils/audio.py`` (``SimilarityEvaluator``): MAE on
``log10(|STFT|)`` with an ``eps`` floor, and the Frobenius spectral convergence.
``mel_mae`` / ``mel_mse`` compare log-mel spectrograms; ``mss`` is the DDSP-style
multi-scale spectral loss (Engel et al., 2020). The STFT framing for the single-
resolution metrics matches the anchor (``n_fft=1024``, ``hop=256``).

``loudness_envelope_l1`` follows DDSP's A-weighted loudness contour;
``integrated_loudness_error`` uses the same ITU-R BS.1770 meter (``pyloudnorm``)
as the dataset builder's near-silence gate (D-SILENCE). ``f0_rmse`` compares
``librosa.pyin`` F0 contours. The loudness-contour, pitch-range, and FFT-size
choices below are reasonable defaults, not anchored to a reference -- they are
pickable later (as for the mel / MSS parameters).

``librosa`` and ``pyloudnorm`` are hard dependencies of the core panel and are
imported eagerly.
"""
from __future__ import annotations

import numpy as np
import librosa
import pyloudnorm

# Single-resolution STFT framing -- matches the preset-gen-vae anchor.
N_FFT = 1024
HOP_LENGTH = 256
# Mel filterbank resolution for the log-mel and MFCC metrics.
MEL_BINS = 128
# Number of MFCC coefficients for the timbre metrics.
N_MFCC = 13
# FFT sizes for the multi-scale spectral loss (DDSP), 75% overlap per scale.
MSS_FFT_SIZES = (2048, 1024, 512, 256, 128, 64)
# Finite floor (LUFS) for the -inf integrated loudness of silence.
LOUDNESS_FLOOR_LUFS = -70.0
# pyin F0 search range (~C2..C7); outside it is treated as unvoiced.
F0_MIN_HZ = 65.0
F0_MAX_HZ = 2093.0


def _magnitude_stft(signal: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """Magnitude STFT ``|STFT(signal)|`` as a ``(frequency, time)`` array."""
    return np.abs(librosa.stft(np.asarray(signal, dtype=np.float32), n_fft=n_fft, hop_length=hop_length))


def _log_mel(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Log-mel spectrogram in dB (``librosa.power_to_db``).

    ``top_db=None`` disables librosa's default 80 dB clip, which is taken
    *relative to each signal's own maximum*. Without it the target and prediction
    would be floored at different absolute dB levels, so the MAE/MSE would depend
    on each signal's peak rather than purely on their difference.
    """
    mel_power = librosa.feature.melspectrogram(
        y=np.asarray(signal, dtype=np.float32),
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=MEL_BINS,
    )
    return librosa.power_to_db(mel_power, top_db=None)


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
    the *target* magnitude makes this metric order-dependent. Returns ``nan`` when
    the target is silent (zero denominator): the ratio is undefined, and returning
    ``0.0`` would falsely score any prediction -- including a loud one -- as a
    perfect match.
    """
    magnitude_target = _magnitude_stft(target)
    magnitude_prediction = _magnitude_stft(prediction)
    denominator = float(np.linalg.norm(magnitude_target))
    if denominator == 0.0:
        return float("nan")
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


def _loudness_envelope(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Per-frame A-weighted loudness contour in dB (DDSP-style).

    Each STFT frame's power spectrum is A-weighted (perceptual frequency
    response), averaged over frequency, and expressed in dB. A small floor keeps
    the log finite on silent frames.
    """
    power = _magnitude_stft(signal) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)
    with np.errstate(divide="ignore"):
        # A-weighting at the 0 Hz DC bin is -inf dB (full attenuation, weight 0).
        power_weighting = 10.0 ** (librosa.A_weighting(frequencies) / 10.0)
    weighted_power = power * power_weighting[:, np.newaxis]
    return 10.0 * np.log10(np.mean(weighted_power, axis=0) + 1e-10)


def _integrated_loudness(signal: np.ndarray, sample_rate: int) -> float:
    """Gated integrated loudness (LUFS, ITU-R BS.1770) floored at silence.

    Uses the same ``pyloudnorm`` meter as the dataset builder's near-silence
    gate (D-SILENCE). Silence integrates to ``-inf``; it is clamped to
    :data:`LOUDNESS_FLOOR_LUFS` so downstream differences stay finite.
    """
    meter = pyloudnorm.Meter(sample_rate)
    loudness = float(meter.integrated_loudness(np.asarray(signal, dtype=np.float64)))
    return max(loudness, LOUDNESS_FLOOR_LUFS)


def loudness_envelope_l1(target: np.ndarray, prediction: np.ndarray, *, sample_rate: int) -> float:
    """L1 distance between A-weighted loudness contours in dB (lower is better).

    A *level* metric: a common gain shifts both contours by the same number of
    dB, so it does not cancel (unlike the magnitude/timbre metrics).
    """
    return float(np.mean(np.abs(
        _loudness_envelope(target, sample_rate) - _loudness_envelope(prediction, sample_rate)
    )))


def integrated_loudness_error(target: np.ndarray, prediction: np.ndarray, *, sample_rate: int) -> float:
    """Absolute difference in integrated loudness (LUFS; lower is better).

    Captures whether the prediction matches the target's overall perceived level.
    """
    return abs(_integrated_loudness(target, sample_rate) - _integrated_loudness(prediction, sample_rate))


def f0_rmse(target: np.ndarray, prediction: np.ndarray, *, sample_rate: int) -> float:
    """RMSE (Hz) between pyin F0 contours over commonly-voiced frames.

    F0 is estimated with ``librosa.pyin`` over :data:`F0_MIN_HZ`..:data:`F0_MAX_HZ`;
    only frames voiced in *both* signals contribute (pyin marks the rest ``nan``).
    Lower is better; returns ``float('nan')`` when no frame is voiced in both
    (e.g. against silence), since the contours never overlap. Independent of a
    common gain because pitch does not depend on amplitude.
    """
    f0_target, _, _ = librosa.pyin(
        np.asarray(target, dtype=np.float32), fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, sr=sample_rate
    )
    f0_prediction, _, _ = librosa.pyin(
        np.asarray(prediction, dtype=np.float32), fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, sr=sample_rate
    )
    commonly_voiced = ~np.isnan(f0_target) & ~np.isnan(f0_prediction)
    if not np.any(commonly_voiced):
        return float("nan")
    errors = f0_target[commonly_voiced] - f0_prediction[commonly_voiced]
    return float(np.sqrt(np.mean(errors ** 2)))
