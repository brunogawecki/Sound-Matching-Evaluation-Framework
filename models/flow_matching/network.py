"""The complete flow-matching network: mel front-end + AST encoder + vector field.

Composes the pieces the paper wires together in Hydra configs into one plain
``nn.Module`` the framework can rebuild offline from hparams (no VST, no Lightning).
Featurization lives inside the network (D-REPR): raw mono audio -> mel power
spectrogram (the reference's ``make_spectrogram``: 128 mels, ~25 ms window, 10 ms hop,
Hamming window, per-sample max-referenced dB) -> corpus-statistics standardization ->
:class:`AudioSpectrogramTransformer` conditioning -> vector field.

Two contract adaptations from the Surge reference, both driven by the corpus render
contract (D-SELFDESC / D-METRIC-SR): mono 22.05 kHz input instead of stereo 44.1 kHz
(window/hop are specified in milliseconds and recomputed in samples), and the mel-dB
standardization uses *scalar* corpus mean/std (measured at fit time, the presetgen
D-MELNORM pattern) instead of the reference's per-bin dataset statistics file.

``sample`` returns vectors in the flow's own space: the ML-side vector affine-rescaled
to ``[-1, 1]`` (the reference's ``rescale_params``). Callers map back with
``(x + 1) / 2`` before ``ml_vector_to_synth_dict``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.flow_matching.encoder import AudioSpectrogramTransformer
from models.flow_matching.flow_matching import rk4_sample
from models.flow_matching.vector_field import (
    ConditionalResidualMLPField,
    EquivariantTransformerField,
)

_DB_AMIN = 1e-10  # librosa.power_to_db amin
_DB_TOP = 80.0  # librosa.power_to_db top_db


def _build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    """A ``[n_mels, 1 + n_fft // 2]`` mel filterbank at librosa defaults (Slaney norm)."""
    import librosa  # heavy import kept off module load

    filterbank = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    return torch.from_numpy(np.asarray(filterbank, dtype=np.float32))


def _mel_analysis_lengths(
    sample_rate: int, window_duration_ms: float, hop_duration_ms: float
) -> Tuple[int, int]:
    """Window (= ``n_fft``) and hop lengths in samples, the reference's ms-based spec."""
    n_fft = int(window_duration_ms / 1000.0 * sample_rate)
    hop_length = int(hop_duration_ms / 1000.0 * sample_rate)
    return n_fft, hop_length


def _compute_mel_db_spectrogram(
    audio: torch.Tensor,
    window: torch.Tensor,
    mel_filterbank: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """Audio ``[batch, num_samples]`` -> mel-dB ``[batch, n_mels, frames]``.

    The reference's ``librosa.feature.melspectrogram`` + ``power_to_db(ref=np.max)``
    in torch: power STFT -> mel -> dB relative to each sample's own maximum, floored
    ``top_db`` below it.
    """
    # torch.stft needs float32/64; under bf16-mixed autocast the input arrives bf16.
    audio = audio.float()
    complex_stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    power = complex_stft.abs().square()  # [batch, freq, frames]
    mel_power = torch.matmul(mel_filterbank, power)  # [batch, n_mels, frames]
    reference = mel_power.amax(dim=(-2, -1), keepdim=True)
    decibels = 10.0 * torch.log10(torch.clamp(mel_power, min=_DB_AMIN))
    decibels = decibels - 10.0 * torch.log10(torch.clamp(reference, min=_DB_AMIN))
    return torch.clamp(decibels, min=-_DB_TOP)


def measure_corpus_mel_db_statistics(
    dataset: RenderedCorpusDataset,
    sample_rate: int,
    n_mels: int,
    window_duration_ms: float,
    hop_duration_ms: float,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """Scalar mean/std of the mel-dB values over the corpus (the input standardization).

    One pass over ``dataset`` accumulating sum and sum of squares. The scalar stand-in
    for the reference's per-bin ``stats.npz`` (see module docstring).
    """
    from torch.utils.data import DataLoader

    n_fft, hop_length = _mel_analysis_lengths(sample_rate, window_duration_ms, hop_duration_ms)
    window = torch.hamming_window(n_fft, periodic=True)
    mel_filterbank = _build_mel_filterbank(sample_rate, n_fft, n_mels)
    total, total_squared, count = 0.0, 0.0, 0
    for audio, _ in DataLoader(dataset, batch_size=batch_size):
        decibels = _compute_mel_db_spectrogram(audio, window, mel_filterbank, n_fft, hop_length)
        total += float(decibels.sum().double())
        total_squared += float(decibels.double().square().sum())
        count += decibels.numel()
    mean = total / count
    variance = max(total_squared / count - mean**2, 0.0)
    return mean, float(np.sqrt(variance))


class FlowMatchingNetwork(nn.Module):
    """Audio in, conditioning/velocity/samples out -- the CNF families' network.

    ``encode`` runs the front-end + AST once per audio clip; ``velocity`` evaluates the
    vector field (the training target); ``sample`` integrates the ODE from noise with
    CFG-guided RK4. ``forward`` is ``sample`` with fresh unseeded noise, so the module
    still maps ``audio -> ML-side-shaped output`` like every other family's network.
    """

    def __init__(
        self,
        ml_dimension: int,
        num_audio_samples: int,
        sample_rate: int,
        n_mels: int = 128,
        window_duration_ms: float = 25.0,
        hop_duration_ms: float = 10.0,
        mel_mean_db: float = 0.0,
        mel_std_db: float = 1.0,
        encoder_d_model: int = 512,
        encoder_num_heads: int = 8,
        encoder_num_layers: int = 8,
        num_conditioning_outputs: int = 9,
        patch_size: int = 16,
        patch_stride: int = 10,
        vector_field_architecture: str = "mlp",
        field_d_model: int = 768,
        field_num_layers: int = 9,
        field_num_heads: int = 8,
        num_parameter_tokens: int = 128,
        projection_penalty: float = 0.01,
        time_encoding_dimension: int = 256,
        rectified_sigma_min: float = 0.0,
        sample_steps: int = 200,
        sample_cfg_strength: float = 2.0,
    ) -> None:
        super().__init__()
        self.ml_dimension = ml_dimension
        self.rectified_sigma_min = rectified_sigma_min
        self.sample_steps = sample_steps
        self.sample_cfg_strength = sample_cfg_strength
        self._mel_mean_db = mel_mean_db
        self._mel_std_db = mel_std_db

        self._n_fft, self._hop_length = _mel_analysis_lengths(
            sample_rate, window_duration_ms, hop_duration_ms
        )
        num_frames = 1 + num_audio_samples // self._hop_length
        self.register_buffer(
            "window", torch.hamming_window(self._n_fft, periodic=True), persistent=False
        )
        self.register_buffer(
            "mel_filterbank",
            _build_mel_filterbank(sample_rate, self._n_fft, n_mels),
            persistent=False,
        )

        self.encoder = AudioSpectrogramTransformer(
            d_model=encoder_d_model,
            num_heads=encoder_num_heads,
            num_layers=encoder_num_layers,
            num_conditioning_outputs=num_conditioning_outputs,
            patch_size=patch_size,
            patch_stride=patch_stride,
            input_channels=1,
            spectrogram_shape=(n_mels, num_frames),
        )
        if vector_field_architecture == "mlp":
            self.vector_field: nn.Module = ConditionalResidualMLPField(
                num_params=ml_dimension,
                d_model=field_d_model,
                time_encoding_dimension=time_encoding_dimension,
                conditioning_dim=encoder_d_model,
                num_layers=field_num_layers,
            )
        elif vector_field_architecture == "param2tok":
            self.vector_field = EquivariantTransformerField(
                num_params=ml_dimension,
                d_model=field_d_model,
                time_encoding_dimension=time_encoding_dimension,
                conditioning_dim=encoder_d_model,
                num_layers=field_num_layers,
                num_heads=field_num_heads,
                num_tokens=num_parameter_tokens,
                projection_penalty=projection_penalty,
            )
        else:
            raise ValueError(
                f"Unknown vector_field_architecture '{vector_field_architecture}' "
                "(use 'mlp' or 'param2tok')."
            )

    def featurize(self, audio: torch.Tensor) -> torch.Tensor:
        """Audio ``[batch, num_samples]`` -> standardized mel-dB ``[batch, 1, n_mels, frames]``."""
        decibels = _compute_mel_db_spectrogram(
            audio, self.window, self.mel_filterbank, self._n_fft, self._hop_length
        )
        standardized = (decibels - self._mel_mean_db) / self._mel_std_db
        return standardized.unsqueeze(1)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Audio -> conditioning ``[batch, num_conditioning_outputs, encoder_d_model]``."""
        return self.encoder(self.featurize(audio))

    def velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, conditioning: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.vector_field(x_t, t, conditioning)

    def sample(
        self,
        audio: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        num_steps: Optional[int] = None,
        cfg_strength: Optional[float] = None,
    ) -> torch.Tensor:
        """Draw one parameter sample per clip; ``[batch, ml_dimension]`` in flow space.

        Noise is drawn on CPU from ``generator`` (when given) so a fixed seed yields
        the same sample on any device; integration itself is deterministic.
        """
        conditioning = self.encode(audio)
        noise = torch.randn(
            audio.shape[0], self.ml_dimension, generator=generator, device="cpu"
        ).to(device=conditioning.device, dtype=conditioning.dtype)
        return rk4_sample(
            self.vector_field,
            noise,
            conditioning,
            num_steps=num_steps if num_steps is not None else self.sample_steps,
            cfg_strength=cfg_strength if cfg_strength is not None else self.sample_cfg_strength,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.sample(audio)
