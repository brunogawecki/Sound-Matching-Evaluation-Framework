"""Parity tests: the preset-gen-vae port vs the paper's own code in ``paper_repos/``.

Each test builds a component from both codebases, transplants the (randomly
initialized) weights from the paper's module into ours in registration order,
feeds both the same input in eval mode, and asserts numerically identical
outputs. Passing means the two implementations compute the same function --
not merely that they look alike. See docs/PRESETGEN_VAE_PORT.md.

The paper's encoder/decoder and ``LinearDynamicParam`` are dependency-free and
always tested. Its flow, regression, and loss modules import ``nflows`` at
module level (not a project dependency -- ``models/presetgen_vae/realnvp.py``
exists to avoid it), so those tests skip unless nflows is installed
(``pip install nflows --no-deps``; dev-only, keep it out of requirements).

The mel-dB front-end is deliberately not parity-tested: the paper's offline
STFT uses a symmetric Hann window, constant padding, and a window-gain
scaling, all documented as deviations in docs/PRESETGEN_VAE_PORT.md.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest
import torch
from torch import nn

from models.presetgen_vae.network import (
    PresetGenVAENetwork,
    _build_regressor,
    _center_crop_or_pad,
)
from models.presetgen_vae.realnvp import RealNVP
from models.training.loss import gaussian_kl_divergence

# Appended (not prepended) so the paper's config.py cannot shadow ours. The paper's
# package names (model, data, utils, logs) don't exist in this repo, so they resolve
# to the paper's code unambiguously.
_PAPER_ROOT = Path(__file__).resolve().parent.parent / "paper_repos" / "preset-gen-vae"
if str(_PAPER_ROOT) not in sys.path:
    sys.path.append(str(_PAPER_ROOT))

from model import decoder as paper_decoder_module  # noqa: E402
from model import encoder as paper_encoder_module  # noqa: E402
from utils.hparams import LinearDynamicParam  # noqa: E402

# The paper's fixed geometry: 257-mel / 347-frame spectrograms (its decoder asserts
# this size). 88576 audio samples -> 347 STFT frames at hop 256 with center padding.
_SPECTROGRAM_SIZE = (257, 347)
_NUM_AUDIO_SAMPLES = 346 * 256


def transplant_parameters(
    source_modules: Sequence[nn.Module], target_module: nn.Module
) -> None:
    """Copy every parameter AND buffer of ``source_modules`` (in registration
    order) into ``target_module``, asserting a one-to-one shape match.

    Buffers matter: both encoders run a dummy shape-inference forward in their
    constructors while still in train mode, so their batch-norm running stats
    already differ from fresh init (and from each other) before any training.
    """
    for collect in (
        lambda module: list(module.parameters()),
        lambda module: list(module.buffers()),
    ):
        source_tensors = [
            tensor for module in source_modules for tensor in collect(module)
        ]
        target_tensors = collect(target_module)
        assert len(source_tensors) == len(target_tensors), (
            f"tensor count mismatch: paper {len(source_tensors)} "
            f"vs ours {len(target_tensors)}"
        )
        with torch.no_grad():
            for source, target in zip(source_tensors, target_tensors):
                assert source.shape == target.shape, (
                    f"tensor shape mismatch: paper {tuple(source.shape)} "
                    f"vs ours {tuple(target.shape)}"
                )
                target.copy_(source)


@pytest.fixture(scope="module")
def our_network() -> PresetGenVAENetwork:
    torch.manual_seed(0)
    return PresetGenVAENetwork(
        ml_dimension=16, num_audio_samples=_NUM_AUDIO_SAMPLES
    )


def test_encoder_matches_paper(our_network: PresetGenVAENetwork) -> None:
    """Our enc1..enc8 + mu/logvar MLP == the paper's composed SpectrogramEncoder."""
    torch.manual_seed(1)
    paper_encoder = paper_encoder_module.SpectrogramEncoder(
        "speccnn8l1_bn",
        dim_z=256,
        input_tensor_size=(2, 1, *_SPECTROGRAM_SIZE),
        fc_dropout=0.3,
        deepest_features_mix=False,  # the shipped config's setting
    )
    transplant_parameters(
        [paper_encoder.single_ch_cnn, paper_encoder.features_mixer_cnn],
        our_network.spectrogram_cnn,
    )
    transplant_parameters([paper_encoder.mlp], our_network.encoder_mlp)
    our_network.eval()
    paper_encoder.eval()

    spectrogram = torch.randn(2, 1, *_SPECTROGRAM_SIZE)
    with torch.no_grad():
        our_mu, our_logvar = our_network._encode(spectrogram)
        paper_mu_logvar = paper_encoder(spectrogram)
    assert torch.allclose(our_mu, paper_mu_logvar[:, 0, :], atol=1e-6)
    assert torch.allclose(our_logvar, paper_mu_logvar[:, 1, :], atol=1e-6)


def test_decoder_matches_paper(our_network: PresetGenVAENetwork) -> None:
    """Our decoder MLP + dec1..dec8 == the paper's SpectrogramDecoder.

    The only intentional difference is our final center crop to the input
    geometry, so the paper's raw output is cropped identically before comparing.
    """
    torch.manual_seed(2)
    paper_decoder = paper_decoder_module.SpectrogramDecoder(
        "speccnn8l1_bn",
        dim_z=256,
        output_tensor_size=(2, 1, *_SPECTROGRAM_SIZE),
        fc_dropout=0.3,
    )
    transplant_parameters([paper_decoder.mlp], our_network.decoder_mlp)
    transplant_parameters(
        [paper_decoder.features_unmixer_cnn, paper_decoder.single_ch_cnn],
        our_network.decoder_cnn,
    )
    our_network.eval()
    paper_decoder.eval()

    latent = torch.randn(2, 256)
    with torch.no_grad():
        our_reconstruction = our_network._decode(latent)
        paper_reconstruction = paper_decoder(latent)
    assert our_reconstruction.shape == (2, 1, *_SPECTROGRAM_SIZE)
    assert torch.allclose(
        our_reconstruction,
        _center_crop_or_pad(paper_reconstruction, *_SPECTROGRAM_SIZE),
        atol=1e-6,
    )


def test_beta_warmup_matches_paper() -> None:
    """linear_warmup == the paper's LinearDynamicParam beta schedule."""
    from models.presetgen_vae.lightning_module import linear_warmup

    paper_schedule = LinearDynamicParam(0.1, 0.2, end_epoch=25)
    for epoch in range(40):
        assert linear_warmup(epoch, 0.1, 0.2, 25) == pytest.approx(
            paper_schedule.get(epoch)
        )


def test_mlp_regressor_matches_paper() -> None:
    """Our MLP head == the paper's MLPRegression minus its PresetActivation."""
    pytest.importorskip("nflows")  # model/regression.py imports nflows at module level
    from model import regression as paper_regression_module

    torch.manual_seed(3)
    # MLPRegression only reads learnable_preset_size from the indexes helper
    # (PresetActivation ignores it entirely when cat_softmax_activation=False).
    fake_indexes_helper = SimpleNamespace(learnable_preset_size=16)
    paper_head = paper_regression_module.MLPRegression(
        "3l1024", dim_z=256, idx_helper=fake_indexes_helper, dropout_p=0.4,
        cat_softmax_activation=False,
    )
    our_head = _build_regressor(
        latent_dimension=256, ml_dimension=16, hidden_layers=3,
        hidden_width=1024, dropout=0.4,
    )
    # Drop the paper's trailing PresetActivation (we emit raw outputs by design).
    paper_head_without_activation = paper_head.reg_model[:-1]
    transplant_parameters([paper_head_without_activation], our_head)
    our_head.eval()
    paper_head_without_activation.eval()

    latent = torch.randn(4, 256)
    with torch.no_grad():
        assert torch.allclose(
            our_head(latent), paper_head_without_activation(latent), atol=1e-6
        )


def test_flow_regressor_matches_paper() -> None:
    """Our RealNVP == the paper's CustomRealNVP (nflows), outputs and log-dets."""
    pytest.importorskip("nflows")
    from model.flows import CustomRealNVP

    torch.manual_seed(4)
    paper_flow = CustomRealNVP(
        features=20, hidden_features=32, num_layers=6, num_blocks_per_layer=2,
        dropout_probability=0.4, batch_norm_within_layers=True,
        batch_norm_between_layers=True,
    )
    our_flow = RealNVP(
        features=20, hidden_features=32, coupling_layers=6, dropout=0.4
    )
    transplant_parameters([paper_flow], our_flow)
    our_flow.eval()
    paper_flow.eval()

    inputs = torch.randn(8, 20)
    with torch.no_grad():
        paper_outputs, paper_log_determinant = paper_flow(inputs)
        our_outputs, our_log_determinant = our_flow.forward_with_log_determinant(inputs)
    assert torch.allclose(our_outputs, paper_outputs, atol=1e-6)
    assert torch.allclose(our_log_determinant, paper_log_determinant, atol=1e-6)


def test_flow_train_mode_matches_paper() -> None:
    """Train-mode parity for the flow batch-norm path (batch statistics + their
    log-det term + running-stat updates). Dropout is set to 0 on both sides so
    the comparison stays deterministic."""
    pytest.importorskip("nflows")
    from model.flows import CustomRealNVP

    torch.manual_seed(5)
    paper_flow = CustomRealNVP(
        features=20, hidden_features=32, num_layers=6, num_blocks_per_layer=2,
        dropout_probability=0.0, batch_norm_within_layers=True,
        batch_norm_between_layers=True,
    )
    our_flow = RealNVP(
        features=20, hidden_features=32, coupling_layers=6, dropout=0.0
    )
    transplant_parameters([paper_flow], our_flow)
    our_flow.train()
    paper_flow.train()

    inputs = torch.randn(16, 20)
    paper_outputs, paper_log_determinant = paper_flow(inputs)
    our_outputs, our_log_determinant = our_flow.forward_with_log_determinant(inputs)
    assert torch.allclose(our_outputs, paper_outputs, atol=1e-5)
    assert torch.allclose(our_log_determinant, paper_log_determinant, atol=1e-5)


def test_kl_divergence_matches_paper() -> None:
    """gaussian_kl_divergence == the paper's GaussianDkl, both normalizations."""
    pytest.importorskip("nflows")  # model/loss.py imports nflows at module level
    from model.loss import GaussianDkl

    torch.manual_seed(6)
    mu = torch.randn(8, 16)
    logvar = torch.randn(8, 16)
    for normalize in (True, False):
        ours = gaussian_kl_divergence(mu, logvar, normalize=normalize)
        paper = GaussianDkl(normalize=normalize)(mu, logvar)
        assert torch.allclose(ours, paper, atol=1e-6)
