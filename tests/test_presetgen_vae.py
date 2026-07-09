"""Tests for the preset-gen-vae port (Stage 1: mel-dB encoder -> MLP regressor).

Forward-shape, front-end, and rebuild-determinism checks on the network, plus an
end-to-end fit -> save -> load -> predict smoke test on a tiny synthetic corpus
(mirroring ``tests/test_sound2synth.py``). The network checks need only ``torch``; the
end-to-end test additionally skips when ``lightning`` is absent (training-only deps).
"""
import os
import sys
from typing import Dict

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from models.presetgen_vae import (
    PresetGenVaeAutoencoderNetwork,
    PresetGenVaeNetwork,
    PresetGenVaeRegressor,
    VaeNetworkOutput,
)

SAMPLE_RATE = 8000
NUM_AUDIO_SAMPLES = 4000  # 0.5 s
DURATION_SEC = NUM_AUDIO_SAMPLES / SAMPLE_RATE  # keep the corpus render length tiny

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


# -- Stage 2 autoencoder network (VAE) ---------------------------------------------
# Same tiny STFT/mel/latent as Stage 1; the encoder + regressor are shared, so only the
# added VAE machinery (mu/logvar split, reparameterization, decoder) is exercised here.


def test_autoencoder_forward_maps_audio_batch_to_ml_dimension():
    ml_dimension = 137
    network = PresetGenVaeAutoencoderNetwork(ml_dimension=ml_dimension, **TINY_KWARGS)
    network.eval()
    audio = torch.randn(3, NUM_AUDIO_SAMPLES)
    output = network(audio)
    assert output.shape == (3, ml_dimension)


def test_autoencoder_forward_training_returns_all_terms_with_correct_shapes():
    ml_dimension = 41
    dim_z = TINY_KWARGS["dim_z"]
    network = PresetGenVaeAutoencoderNetwork(ml_dimension=ml_dimension, **TINY_KWARGS)
    audio = torch.randn(2, NUM_AUDIO_SAMPLES)
    output = network.forward_training(audio)
    assert isinstance(output, VaeNetworkOutput)
    assert output.prediction.shape == (2, ml_dimension)
    assert output.mu.shape == (2, dim_z)
    assert output.logvar.shape == (2, dim_z)
    # The decoder reconstructs exactly the encoder's input spectrogram (its training target).
    assert output.reconstruction.shape == output.target_spectrogram.shape
    assert output.reconstruction.shape[1] == 1  # single spectrogram channel
    assert output.reconstruction.shape[2] == TINY_KWARGS["n_mels"]


def test_autoencoder_reconstruction_is_bounded_to_normalized_range():
    network = PresetGenVaeAutoencoderNetwork(ml_dimension=5, **TINY_KWARGS)
    audio = torch.randn(2, NUM_AUDIO_SAMPLES)
    reconstruction = network.forward_training(audio).reconstruction
    # Hardtanh output must match the [-1, 1] range of the normalized mel-dB target.
    assert torch.all(reconstruction >= -1.0) and torch.all(reconstruction <= 1.0)


def test_autoencoder_eval_forward_is_deterministic():
    network = PresetGenVaeAutoencoderNetwork(ml_dimension=7, **TINY_KWARGS)
    network.eval()  # eval uses the posterior mean -> no sampling
    audio = torch.randn(2, NUM_AUDIO_SAMPLES)
    assert torch.allclose(network(audio), network(audio))


def test_autoencoder_reparameterizes_only_in_training_mode():
    torch.manual_seed(0)
    network = PresetGenVaeAutoencoderNetwork(ml_dimension=7, **TINY_KWARGS)
    audio = torch.randn(2, NUM_AUDIO_SAMPLES)
    network.train()  # training samples z ~ N(mu, sigma) -> reconstructions differ across calls
    first = network.forward_training(audio).reconstruction
    second = network.forward_training(audio).reconstruction
    assert not torch.allclose(first, second)


# -- end-to-end family (fit -> save -> load -> predict) ----------------------------
# The family constructor takes only architecture knobs; ml_dimension, num_audio_samples
# and sample_rate come from the corpus at fit time. mel_fmax stays <= sample_rate / 2.
FAMILY_TINY_KWARGS = dict(
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


def make_space():
    from synth.parameter_space import ParameterSpace, ParameterSpecification

    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.8),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 0.5, 1.0], default=0.0),
    ])


class FakeSynth:
    """A no-VST sine synth (mirrors tests/test_sound2synth.py)."""

    renderer_name = "fake"

    def __init__(self, space, sample_rate: int = SAMPLE_RATE):
        self._space = space
        self._sample_rate = sample_rate
        self._state: Dict[str, float] = {s.name: s.default for s in space.parameter_specs}

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def parameter_space(self):
        return self._space

    def get_parameter_defaults(self) -> Dict[str, float]:
        return {s.name: s.default for s in self._space.parameter_specs}

    def set_parameters(self, params: Dict[str, float]) -> None:
        self._state.update(params)

    def get_parameters(self) -> Dict[str, float]:
        return dict(self._state)

    def render_audio(self, midi_note, velocity, duration_sec, note_duration_sec=None) -> np.ndarray:
        samples = int(duration_sec * self._sample_rate)
        time = np.arange(samples) / self._sample_rate
        return float(self._state["AMP"]) * np.sin(2.0 * np.pi * 220.0 * time)


def build_corpus(tmp_path, run_name, count, seed):
    from dataset.builder import DatasetBuilder, RenderSettings
    from dataset.preset_sources import SyntheticPresetSource
    from dataset.torch_dataset import RenderedCorpusDataset

    synth = FakeSynth(make_space())
    settings = RenderSettings(
        midi_note=60, velocity=100, duration_sec=DURATION_SEC, note_duration_sec=DURATION_SEC
    )
    source = SyntheticPresetSource(
        make_space(), count=count, seed=seed, sampling_ranges={"AMP": (0.7, 1.0)}
    )
    DatasetBuilder(synth, render_settings=settings).build(
        source, run_name=run_name, output_root=tmp_path
    )
    return RenderedCorpusDataset.load(tmp_path / run_name)


def training_config(seed=0):
    return {
        "seed": seed,
        "optimizer": {"learning_rate": 1e-2},
        "loss": {"categorical_loss_weight": 0.2},
        "data": {"batch_size": 4, "val_fraction": 0.25},
        "trainer": {
            "max_epochs": 5,
            "precision": "32-true",
            "accelerator": "cpu",
            "devices": 1,
            "log_every_n_steps": 1,
        },
    }


def test_fit_save_load_predict_end_to_end(tmp_path):
    pytest.importorskip("lightning")  # training-only dependency (cluster-side)

    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    log_dir = tmp_path / "logs"

    model = PresetGenVaeRegressor(default_root_dir=str(log_dir), **FAMILY_TINY_KWARGS)
    model.fit(train_dataset, config=training_config())

    checkpoint_path = tmp_path / "presetgen_vae.pt"
    model.save(checkpoint_path)
    assert checkpoint_path.exists()

    # Fresh instance loads with no dataset and no VST, then predicts.
    reloaded = PresetGenVaeRegressor(**FAMILY_TINY_KWARGS)
    reloaded.load(checkpoint_path)

    audio, _ = train_dataset[0]
    prediction = reloaded.predict(audio)

    space = train_dataset.parameter_space
    assert set(prediction) == set(space.names)              # all parameters present
    assert prediction["CAT"] in (0.0, 0.5, 1.0)             # categorical snapped to a grid option
    assert 0.0 <= prediction["AMP"] <= 1.0                  # continuous clipped into bounds
