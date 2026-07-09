"""Tests for the preset-gen-vae port (Stage 1: mel-dB encoder -> MLP regressor).

Forward-shape, front-end, and rebuild-determinism checks on the network. The
end-to-end fit -> save -> load -> predict path is covered once the BaseDeepModel
family lands (Task 2). Skips cleanly when ``torch`` is absent (training-only dep).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from models.presetgen_vae import PresetGenVaeNetwork

SAMPLE_RATE = 8000
NUM_AUDIO_SAMPLES = 4000  # 0.5 s

# A small STFT / mel / latent so the CNN does not collapse the tiny spectrogram and
# the CPU test stays fast. The 8-layer conv schedule itself is byte-faithful to the paper.
TINY_KWARGS = dict(
    num_audio_samples=NUM_AUDIO_SAMPLES,
    sample_rate=SAMPLE_RATE,
    n_fft=256,
    hop_length=128,
    win_length=256,
    n_mels=32,
    mel_fmin=0.0,
    mel_fmax=4000.0,
    dim_z=16,
    encoder_dropout=0.0,
    regressor_hidden_layers=2,
    regressor_hidden_width=16,
    regressor_dropout=0.0,
)


def test_forward_maps_audio_batch_to_ml_dimension():
    ml_dimension = 137
    network = PresetGenVaeNetwork(ml_dimension=ml_dimension, **TINY_KWARGS)
    audio = torch.randn(3, NUM_AUDIO_SAMPLES)
    output = network(audio)
    assert output.shape == (3, ml_dimension)


def test_mel_db_front_end_is_bounded_and_correctly_shaped():
    network = PresetGenVaeNetwork(ml_dimension=5, **TINY_KWARGS)
    audio = torch.randn(2, NUM_AUDIO_SAMPLES)
    spectrogram = network._mel_db_spectrogram(audio)
    assert spectrogram.shape[:2] == (2, 1)          # [batch, 1 channel, ...]
    assert spectrogram.shape[2] == TINY_KWARGS["n_mels"]
    # Min-max normalization keeps the mel-dB input inside [-1, 1] for the decoder target.
    assert torch.all(spectrogram >= -1.0) and torch.all(spectrogram <= 1.0)


def test_build_is_deterministic_in_hparams():
    first = PresetGenVaeNetwork(ml_dimension=12, **TINY_KWARGS)
    second = PresetGenVaeNetwork(ml_dimension=12, **TINY_KWARGS)
    # Same structure -> identical parameter-tensor names and shapes.
    first_shapes = {name: tuple(p.shape) for name, p in first.state_dict().items()}
    second_shapes = {name: tuple(p.shape) for name, p in second.state_dict().items()}
    assert first_shapes == second_shapes


def test_encoder_cnn_has_no_batch_norm_on_first_and_last_layers():
    network = PresetGenVaeNetwork(ml_dimension=5, **TINY_KWARGS)
    first_block = network.spectrogram_cnn[0]
    last_block = network.spectrogram_cnn[-1]
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in first_block)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in last_block)
    # An interior block does carry batch-norm.
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in network.spectrogram_cnn[1])
