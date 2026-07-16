"""The paper's ``FlowVAE`` network: mel-dB front-end + ``speccnn8l1_bn`` VAE + regressor head.

The mel-dB front-end and the paper's ``speccnn8l1_bn`` spectrogram encoder emit
``mu``/``logvar``; a reparameterized sample ``z0`` is pushed through the **latent RealNVP
flow** to ``zK``, which feeds both the mirror-image ``speccnn8l1_bn`` spectrogram decoder
(reconstruction) and the regressor head (parameters). The regression gradient flows into the
encoder (joint training), exactly as the paper trains the VAE and regressor together.

This is the architecture of the paper's Figure 1, i.e. both models it reports: the head is
either its MLP (``3l1024``, the "MLP" rows of Table 1) or its RealNVP flow used feed-forward
(``realnvp_6l300``, the "Flow" rows). Setting ``latent_flow_layers=0`` drops the latent flow
back to the paper's plain-Gaussian ``BasicVAE`` code path -- a free ablation, but not a model
the paper reports, so no family registers it.

The head emits **raw** outputs (continuous floats + categorical logits), matching the
``ParameterSpace`` / ``ParameterLoss`` contract -- the paper's ``PresetActivation``
(Hardtanh + softmax) is intentionally dropped, exactly as ``Sound2SynthSpectrogramNetwork``
does. ``ParameterLoss`` applies softmax/cross-entropy to categorical blocks itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.presetgen_vae.realnvp import RealNVP

# Fixed architecture constants, faithful to the paper's speccnn8l1_bn encoder.
_NEGATIVE_SLOPE = 0.1  # LeakyReLU slope used throughout the paper's CNN
_LOG_EPSILON = 1e-7  # floor inside log10 before dB conversion


def _build_mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> torch.Tensor:
    """A ``[n_mels, 1 + n_fft // 2]`` mel filterbank (librosa, un-normalized)."""
    import librosa  # heavy import kept off module load

    filterbank = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm=None
    )
    return torch.from_numpy(np.asarray(filterbank, dtype=np.float32))


def _compute_mel_db_spectrogram(
    audio: torch.Tensor,
    window: torch.Tensor,
    mel_filterbank: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    min_db: float,
    max_db: float,
) -> torch.Tensor:
    """Audio ``[batch, num_samples]`` -> normalized mel-dB ``[batch, 1, n_mels, frames]``.

    STFT -> mel filterbank -> dB -> min-max to [-1, 1]. The autoencoder's decoder targets
    exactly this normalized [-1, 1] spectrogram.
    """
    # torch.stft needs float32/64; under bf16-mixed autocast the input arrives bf16.
    audio = audio.float()
    complex_stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    magnitude = complex_stft.abs()  # [batch, freq, frames]
    mel_magnitude = torch.matmul(mel_filterbank, magnitude)  # [batch, n_mels, frames]
    decibels = 20.0 * torch.log10(mel_magnitude + _LOG_EPSILON)
    decibels = torch.clamp(decibels, min=min_db, max=max_db)
    normalized = 2.0 * (decibels - min_db) / (max_db - min_db) - 1.0
    return normalized.unsqueeze(1)  # add the single input channel


def _center_crop_or_pad(tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    """Center-crop (when bigger) or zero-pad (when smaller) the last two dims to the target.

    The ported decoder's transposed-conv schedule lands within a few pixels of the encoder's
    input spectrogram size (exactly ``(n_mels, frames + 2)`` at the paper's 257-mel / 345-frame
    contract). This makes the reconstruction the exact ``(n_mels, frames)`` target for any
    render length, so the reconstruction MSE is always well-defined.
    """
    height, width = tensor.shape[-2], tensor.shape[-1]
    if height > target_height:
        top = (height - target_height) // 2
        tensor = tensor[:, :, top : top + target_height, :]
    elif height < target_height:
        pad_total = target_height - height
        tensor = F.pad(tensor, (0, 0, pad_total // 2, pad_total - pad_total // 2))
    width = tensor.shape[-1]
    if width > target_width:
        left = (width - target_width) // 2
        tensor = tensor[:, :, :, left : left + target_width]
    elif width < target_width:
        pad_total = target_width - width
        tensor = F.pad(tensor, (pad_total // 2, pad_total - pad_total // 2, 0, 0))
    return tensor


def _conv2d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: int,
    use_batch_norm: bool,
) -> nn.Sequential:
    """One conv layer of the paper's ``layer.Conv2D``: conv -> LeakyReLU -> (BN).

    Batch-norm is applied *after* the activation (the paper's ``batch_norm='after'``),
    and omitted on the first and last conv layers (``batch_norm=None`` there).
    """
    block = nn.Sequential()
    block.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    block.add_module("act", nn.LeakyReLU(_NEGATIVE_SLOPE))
    if use_batch_norm:
        block.add_module("bn", nn.BatchNorm2d(out_channels))
    return block


def _build_spectrogram_cnn() -> nn.Sequential:
    """The paper's ``speccnn8l1_bn`` single-channel encoder CNN (enc1..enc8).

    Eight strided convolutions taking a 1-channel spectrogram to 2048 feature maps.
    No batch-norm on the first (enc1) and last (enc8) layers, per the paper. enc8's width
    follows the paper's composed ``SpectrogramEncoder`` (``mixer_1x1conv_ch`` = 2048 for
    single-channel input), not the raw ``speccnn8l1_bn`` listing (512 -> 1024), which is
    dead code under the paper's own config.
    """
    return nn.Sequential(
        _conv2d_block(1, 8, (5, 5), (2, 2), 2, use_batch_norm=False),  # enc1
        _conv2d_block(8, 16, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc2
        _conv2d_block(16, 32, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc3
        _conv2d_block(32, 64, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc4
        _conv2d_block(64, 128, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc5
        _conv2d_block(128, 256, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc6
        _conv2d_block(256, 512, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc7 (4x4conv)
        _conv2d_block(512, 2048, (1, 1), (1, 1), 0, use_batch_norm=False),  # enc8 (1x1conv)
    )


def _tconv2d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: int,
    output_padding: Tuple[int, int],
    use_batch_norm: bool,
) -> nn.Sequential:
    """One transposed-conv layer of the paper's ``layer.TConv2D``: tconv -> LeakyReLU -> (BN).

    The decoder mirror of :func:`_conv2d_block`. Batch-norm is applied after the activation
    (the paper's ``batch_norm='after'``); ``output_padding`` disambiguates the transposed-conv
    output size where a strided conv would otherwise map several sizes to one.
    """
    block = nn.Sequential()
    block.add_module(
        "tconv",
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
    )
    block.add_module("act", nn.LeakyReLU(_NEGATIVE_SLOPE))
    if use_batch_norm:
        block.add_module("bn", nn.BatchNorm2d(out_channels))
    return block


def _build_decoder_cnn(unmixer_in_channels: int) -> nn.Sequential:
    """The paper's ``speccnn8l1_bn`` single-channel spectrogram decoder (dec1..dec8).

    A mirror of :func:`_build_spectrogram_cnn`: a 1x1 "un-mixer" transposed conv (dec1,
    mirroring the encoder's 1x1 enc8) followed by six 4x4 up-convolutions and a final 5x5
    transposed conv to one channel, ending in ``Hardtanh`` to bound the output to the
    normalized mel-dB target's [-1, 1] range. ``unmixer_in_channels`` is the encoder's deepest
    channel count (2048 here), so the decoder inverts the exact encoder feature map.
    """
    return nn.Sequential(
        _tconv2d_block(unmixer_in_channels, 512, (1, 1), (1, 1), 0, (0, 0), use_batch_norm=True),  # dec1
        _tconv2d_block(512, 256, (4, 4), (2, 2), 2, (1, 1), use_batch_norm=True),  # dec2
        _tconv2d_block(256, 128, (4, 4), (2, 2), 2, (1, 0), use_batch_norm=True),  # dec3
        _tconv2d_block(128, 64, (4, 4), (2, 2), 2, (1, 1), use_batch_norm=True),  # dec4
        _tconv2d_block(64, 32, (4, 4), (2, 2), 2, (1, 1), use_batch_norm=True),  # dec5
        _tconv2d_block(32, 16, (4, 4), (2, 2), 2, (1, 0), use_batch_norm=True),  # dec6
        _tconv2d_block(16, 8, (4, 4), (2, 2), 2, (1, 0), use_batch_norm=True),  # dec7
        nn.ConvTranspose2d(8, 1, (5, 5), (2, 2), 2),  # dec8 (final, no activation / BN)
        nn.Hardtanh(),  # bound reconstruction to the normalized [-1, 1] mel-dB target range
    )


def _build_regressor(
    latent_dimension: int,
    ml_dimension: int,
    hidden_layers: int,
    hidden_width: int,
    dropout: float,
) -> nn.Sequential:
    """The paper's ``MLPRegression`` (e.g. ``3l1024``) minus its ``PresetActivation``.

    ``hidden_layers`` fully-connected layers of ``hidden_width``, each with ReLU; the
    first ``hidden_layers - 1`` also carry BatchNorm + Dropout (the paper omits both on
    the two deepest layers). Ends in a plain ``Linear`` to ``ml_dimension`` raw outputs.
    """
    if hidden_layers < 1:
        raise ValueError(f"regressor hidden_layers must be >= 1, got {hidden_layers}.")
    model = nn.Sequential()
    for layer_index in range(hidden_layers):
        in_features = latent_dimension if layer_index == 0 else hidden_width
        model.add_module(f"fc{layer_index + 1}", nn.Linear(in_features, hidden_width))
        if layer_index < hidden_layers - 1:
            model.add_module(f"bn{layer_index + 1}", nn.BatchNorm1d(hidden_width))
            model.add_module(f"drp{layer_index + 1}", nn.Dropout(dropout))
        model.add_module(f"act{layer_index + 1}", nn.ReLU())
    model.add_module(f"fc{hidden_layers + 1}", nn.Linear(hidden_width, ml_dimension))
    return model


@dataclass(frozen=True)
class VAENetworkOutput:
    """Everything the training loss needs from one ``forward_training`` pass.

    ``prediction`` and ``reconstruction`` both come from ``transformed_latent_sample`` (zK);
    ``target_spectrogram`` is the mel-dB input the decoder is trained to reconstruct;
    ``mu``/``logvar`` parameterize the approximate posterior.

    ``log_abs_determinant`` is the latent flow's per-sample log|det J|, and is ``None`` when
    the latent flow is disabled -- which is how the training step picks its latent term: the
    Monte-Carlo estimate when a flow is present, the closed-form KL when it is not.
    """

    prediction: torch.Tensor  # [batch, ml_dimension] raw floats + categorical logits
    reconstruction: torch.Tensor  # [batch, 1, n_mels, frames] in [-1, 1]
    target_spectrogram: torch.Tensor  # [batch, 1, n_mels, frames] in [-1, 1]
    mu: torch.Tensor  # [batch, latent_dimension]
    logvar: torch.Tensor  # [batch, latent_dimension]
    latent_sample: torch.Tensor  # z0: [batch, latent_dimension]
    transformed_latent_sample: torch.Tensor  # zK: [batch, latent_dimension]
    log_abs_determinant: Optional[torch.Tensor] = None  # [batch], None without a latent flow


class PresetGenVAENetwork(nn.Module):
    """The paper's ``FlowVAE`` -- a spectrogram autoencoder with a normalizing flow on its
    latent and a regressor head, predicting the ML-side vector ``[batch, ml_dimension]``.

    The paper's ``speccnn8l1_bn`` encoder feeds an MLP emitting ``2 * latent_dimension``
    values (``mu`` and ``logvar``; ``latent_dimension`` is the paper's ``dim_z``). A
    reparameterized sample ``z0`` is pushed through the latent RealNVP flow to ``zK``, which
    feeds both the mirror-image ``speccnn8l1_bn`` decoder -- reconstructing the normalized
    mel-dB spectrogram -- and the regressor head.

    The head is ``regressor_architecture``: the paper's MLP, or its RealNVP flow used
    feed-forward -- an invertible map, so ``latent_dimension`` must then equal ``ml_dimension``
    (the paper's build-time assert). For the flow, ``regressor_hidden_layers`` /
    ``regressor_hidden_width`` mean coupling layers / hidden features (the paper's ``6l300``).

    ``latent_flow_layers`` / ``latent_flow_hidden_features`` size the latent flow (the paper's
    ``realnvp_6l300``); ``latent_flow_layers=0`` disables it, restoring the plain-Gaussian
    ``BasicVAE`` path with a closed-form KL.

    ``forward`` returns the regressor prediction only (from the posterior mean, deterministic in
    eval), so the inherited ``BaseDeepModel.predict`` path is unchanged and the decoder is skipped
    at inference. ``forward_training`` additionally runs the decoder and returns everything the
    reconstruction + latent + parameter loss needs.
    """

    def __init__(
        self,
        ml_dimension: int,
        num_audio_samples: int,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 257,
        mel_fmin: float = 30.0,
        mel_fmax: float = 11000.0,
        spectrogram_min_db: float = -120.0,
        spectrogram_max_db: float = 0.0,
        latent_dimension: int = 256,
        encoder_dropout: float = 0.3,
        latent_flow_layers: int = 6,
        latent_flow_hidden_features: int = 300,
        regressor_architecture: str = "mlp",
        regressor_hidden_layers: int = 3,
        regressor_hidden_width: int = 1024,
        regressor_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if spectrogram_max_db <= spectrogram_min_db:
            raise ValueError("spectrogram_max_db must exceed spectrogram_min_db.")
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        self._n_mels = n_mels
        self._min_db = float(spectrogram_min_db)
        self._max_db = float(spectrogram_max_db)
        self._latent_dimension = latent_dimension

        # Deterministic, non-persistent buffers (follow .to(device), out of state_dict).
        self.register_buffer("_window", torch.hann_window(win_length), persistent=False)
        mel_filterbank = _build_mel_filterbank(sample_rate, n_fft, n_mels, mel_fmin, mel_fmax)
        self.register_buffer("_mel_filterbank", mel_filterbank, persistent=False)

        self.spectrogram_cnn = _build_spectrogram_cnn()
        # Infer the encoder's deepest feature-map shape from a dummy render-length input, so the
        # decoder MLP/un-mixer invert exactly that shape and the reconstruction target size is known.
        cnn_output_shape, target_size = self._infer_cnn_output_shape(num_audio_samples)
        self._cnn_output_shape = cnn_output_shape  # (channels, height, width)
        self._target_spectrogram_size = target_size  # (n_mels, frames)
        cnn_output_items = int(np.prod(cnn_output_shape))

        # Encoder MLP emits mu and logvar (2 * latent_dimension), reshaped to [batch, 2, latent_dimension].
        self.encoder_mlp = nn.Sequential(
            nn.Dropout(encoder_dropout), nn.Linear(cnn_output_items, 2 * latent_dimension)
        )
        if latent_flow_layers > 0:
            # The paper's 'lat_in_regularization': with a latent flow, z0 is no longer pulled
            # towards N(0, I) by a KL term, so batch-norm keeps the flow's input near zero.
            self.encoder_mlp.add_module(
                "lat_in_regularization", nn.BatchNorm1d(2 * latent_dimension)
            )
        # Decoder MLP mirrors it: latent -> flattened deepest feature map (paper drops the ReLU here).
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dimension, cnn_output_items), nn.Dropout(encoder_dropout)
        )
        self.decoder_cnn = _build_decoder_cnn(unmixer_in_channels=cnn_output_shape[0])
        # The paper's latent flow is nflows' SimpleRealNVP, not the CustomRealNVP its regressor
        # uses. Under the paper's own settings -- no dropout, no batch-norm *between* couplings
        # ("True would prevent reversibility during train") -- the two build an identical stack,
        # so the ported RealNVP covers both roles.
        self.latent_flow = (
            RealNVP(
                features=latent_dimension,
                hidden_features=latent_flow_hidden_features,
                coupling_layers=latent_flow_layers,
                dropout=0.0,
                batch_norm_between_layers=False,
                batch_norm_within_layers=True,
            )
            if latent_flow_layers > 0
            else None
        )
        if regressor_architecture == "mlp":
            self.regressor: nn.Module = _build_regressor(
                latent_dimension, ml_dimension, regressor_hidden_layers, regressor_hidden_width, regressor_dropout
            )
        elif regressor_architecture == "flow":
            if latent_dimension != ml_dimension:
                raise ValueError(
                    "regressor_architecture='flow' is invertible and needs latent_dimension == "
                    f"ml_dimension, got {latent_dimension} != {ml_dimension}."
                )
            self.regressor = RealNVP(
                features=ml_dimension,
                hidden_features=regressor_hidden_width,
                coupling_layers=regressor_hidden_layers,
                dropout=regressor_dropout,
            )
        else:
            raise ValueError(
                f"Unknown regressor_architecture '{regressor_architecture}' (use 'mlp' or 'flow')."
            )

    def _mel_db_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        return _compute_mel_db_spectrogram(
            audio, self._window, self._mel_filterbank,
            self._n_fft, self._hop_length, self._win_length, self._min_db, self._max_db,
        )

    def _infer_cnn_output_shape(
        self, num_audio_samples: int
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int]]:
        with torch.no_grad():
            dummy_audio = torch.zeros(1, num_audio_samples)
            dummy_spectrogram = self._mel_db_spectrogram(dummy_audio)
            cnn_output = self.spectrogram_cnn(dummy_spectrogram)
        channels, height, width = cnn_output.shape[1:]
        target_size = (int(dummy_spectrogram.shape[-2]), int(dummy_spectrogram.shape[-1]))
        return (int(channels), int(height), int(width)), target_size

    def _encode(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.spectrogram_cnn(spectrogram)
        flattened = torch.flatten(features, start_dim=1)
        mu_logvar = self.encoder_mlp(flattened).view(-1, 2, self._latent_dimension)
        return mu_logvar[:, 0, :], mu_logvar[:, 1, :]

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Sample only while training; eval uses the posterior mean (no randomness), as the paper does.
        if self.training:
            standard_deviation = torch.exp(0.5 * logvar)
            return mu + standard_deviation * torch.randn_like(standard_deviation)
        return mu

    def _apply_latent_flow(
        self, latent_sample: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """z0 -> zK and its log|det J|. The identity (and no log-det) when the flow is disabled."""
        if self.latent_flow is None:
            return latent_sample, None
        return self.latent_flow.forward_with_log_determinant(latent_sample)

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        mixed_features = self.decoder_mlp(latent).view(-1, *self._cnn_output_shape)
        reconstruction = self.decoder_cnn(mixed_features)
        return _center_crop_or_pad(reconstruction, *self._target_spectrogram_size)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Predict path: regress from zK, with the posterior mean as z0 (deterministic in eval).
        No decoder."""
        spectrogram = self._mel_db_spectrogram(audio)
        mu, _ = self._encode(spectrogram)
        transformed_latent_sample, _ = self._apply_latent_flow(mu)
        return self.regressor(transformed_latent_sample)

    def forward_training(self, audio: torch.Tensor) -> VAENetworkOutput:
        """Full VAE pass for the training loss: prediction + reconstruction + latent terms.

        Both the decoder and the regressor consume zK, as the paper's ``FlowVAE`` does.
        """
        spectrogram = self._mel_db_spectrogram(audio)
        mu, logvar = self._encode(spectrogram)
        latent_sample = self._reparameterize(mu, logvar)
        transformed_latent_sample, log_abs_determinant = self._apply_latent_flow(latent_sample)
        return VAENetworkOutput(
            prediction=self.regressor(transformed_latent_sample),
            reconstruction=self._decode(transformed_latent_sample),
            target_spectrogram=spectrogram,
            mu=mu,
            logvar=logvar,
            latent_sample=latent_sample,
            transformed_latent_sample=transformed_latent_sample,
            log_abs_determinant=log_abs_determinant,
        )


def measure_corpus_mel_db_range(
    dataset: RenderedCorpusDataset,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    mel_fmin: float,
    mel_fmax: float,
    db_floor: float = -120.0,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """Global min/max mel-dB over the corpus audio (D-MELNORM normalization endpoints).

    One pass over ``dataset``: STFT -> mel -> dB, floored at ``db_floor``, tracking the running
    min and max. These become the [-1, 1] endpoints so the mel-dB input -- and, from Stage 2, the
    decoder's reconstruction target -- fills the range instead of a fixed dB sub-interval.
    """
    from torch.utils.data import DataLoader

    window = torch.hann_window(win_length)
    mel_filterbank = _build_mel_filterbank(sample_rate, n_fft, n_mels, mel_fmin, mel_fmax)
    running_min, running_max = float("inf"), float("-inf")
    for audio, _ in DataLoader(dataset, batch_size=batch_size):
        complex_stft = torch.stft(
            audio.float(), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=True, pad_mode="reflect", return_complex=True,
        )
        mel_magnitude = torch.matmul(mel_filterbank, complex_stft.abs())
        decibels = torch.clamp(20.0 * torch.log10(mel_magnitude + _LOG_EPSILON), min=db_floor)
        running_min = min(running_min, float(decibels.min()))
        running_max = max(running_max, float(decibels.max()))
    return running_min, running_max
