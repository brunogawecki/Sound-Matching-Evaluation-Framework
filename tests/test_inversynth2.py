"""Tests for the InverSynth II ``IS`` (Stage 1), ``IS2xITF`` (Stage 2) and ``IS2`` (Stage 3) models.

Forward-shape and rebuild-determinism checks on the encoder network, an ``IS2Network``
forward_training shape check (encoder + training-only proxy), and end-to-end
fit -> save -> load -> predict smoke tests on a tiny synthetic corpus for every model
(mirroring ``tests/test_sound2synth.py``). ``IS2``'s inference-time finetuning is covered both
in its default proxy-monitored form and with an injected real-synth render callback (the paper's
``L_t^f``) faked with an in-test waveform so no VST is needed. A small STFT + few mels keep the
deep conv stack from collapsing the tiny spectrogram and the run fast on CPU. Skips cleanly when
``torch``/``lightning``/``librosa`` are absent (training-only deps).
"""
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")  # training-only dependency (cluster-side); skip locally if absent
pytest.importorskip("librosa")  # mel filterbank (front-end dependency)

from dataset.builder import DatasetBuilder, RenderSettings
from dataset.preset_sources import SyntheticPresetSource
from dataset.torch_dataset import RenderedCorpusDataset
from models.inversynth2 import IS, IS2, IS2Network, IS2xITF, InverSynthEncoderNetwork
from synth.parameter_space import ParameterSpace, ParameterSpecification

SAMPLE_RATE = 16000
DURATION_SEC = 1.0
EXPECTED_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

# The encoder halves the spectrogram six times, so it needs a spectrogram wide/tall enough not
# to collapse to zero. A 1 s render at 16 kHz with a 512 STFT / 64 mels gives ~64x125, which
# survives the strided stack. Smaller n_mels also keeps the CPU smoke test fast.
TINY_KWARGS = dict(
    n_fft=512,
    hop_length=128,
    win_length=512,
    n_mels=64,
    mel_fmax=8000.0,
    dropout=0.0,
)


def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.8),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 0.5, 1.0], default=0.0),
    ])


class FakeSynth:
    """A no-VST sine synth (mirrors tests/test_sound2synth.py)."""

    renderer_name = "fake"

    def __init__(self, space: ParameterSpace, sample_rate: int = SAMPLE_RATE):
        self._space = space
        self._sample_rate = sample_rate
        self._state: Dict[str, float] = {s.name: s.default for s in space.parameter_specs}

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def parameter_space(self) -> ParameterSpace:
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
        "loss": {"categorical_loss_weight": 1.0},  # the paper's equal CE/L2 average
        "data": {"batch_size": 4, "val_fraction": 0.25},
        "trainer": {
            "max_epochs": 5,  # enough epochs for a reliable train-loss decrease
            "precision": "32-true",
            "accelerator": "cpu",
            "devices": 1,
            "log_every_n_steps": 1,
        },
    }


def logged_metric(log_dir, name) -> list:
    metrics_files = list(log_dir.rglob("metrics.csv"))
    assert metrics_files, "CSVLogger should have written a metrics.csv"
    return pd.read_csv(metrics_files[0])[name].dropna().tolist()


def test_forward_maps_audio_batch_to_ml_dimension():
    ml_dimension = 41
    network = InverSynthEncoderNetwork(
        ml_dimension=ml_dimension, num_audio_samples=EXPECTED_SAMPLES,
        sample_rate=SAMPLE_RATE, **TINY_KWARGS,
    )
    network.eval()  # BatchNorm1d head needs eval mode (or batch > 1) for a batch of any size
    audio = torch.randn(3, EXPECTED_SAMPLES)
    output = network(audio)
    assert output.shape == (3, ml_dimension)


def test_is2_network_forward_training_shapes():
    ml_dimension = 41
    network = IS2Network(
        ml_dimension=ml_dimension, num_audio_samples=EXPECTED_SAMPLES,
        sample_rate=SAMPLE_RATE, proxy_dropout=0.0, **TINY_KWARGS,
    )
    network.eval()  # BatchNorm in the head + proxy needs eval mode (or batch > 1)
    audio = torch.randn(2, EXPECTED_SAMPLES)

    output = network.forward_training(audio)
    assert output.prediction.shape == (2, ml_dimension)
    # The proxy reconstructs the encoder's mel-dB input, so their shapes match exactly.
    assert output.proxy_spectrogram.shape == output.target_spectrogram.shape
    assert output.target_spectrogram.shape[-2] == TINY_KWARGS["n_mels"]
    # The eval path (forward) is the encoder alone -- same prediction width, proxy skipped.
    assert network(audio).shape == (2, ml_dimension)


def test_build_network_is_deterministic_in_hparams():
    model = IS(**TINY_KWARGS)
    hparams = {
        "ml_dimension": 12,
        "num_audio_samples": EXPECTED_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "n_fft": TINY_KWARGS["n_fft"],
        "hop_length": TINY_KWARGS["hop_length"],
        "win_length": TINY_KWARGS["win_length"],
        "n_mels": TINY_KWARGS["n_mels"],
        "mel_fmin": 30.0,
        "mel_fmax": TINY_KWARGS["mel_fmax"],
        "spectrogram_min_db": -120.0,
        "spectrogram_max_db": 0.0,
        "dropout": TINY_KWARGS["dropout"],
    }
    first = model._build_network(hparams)
    second = model._build_network(hparams)
    # Same structure -> identical parameter-tensor names and shapes.
    first_shapes = {name: tuple(p.shape) for name, p in first.state_dict().items()}
    second_shapes = {name: tuple(p.shape) for name, p in second.state_dict().items()}
    assert first_shapes == second_shapes


def test_fit_export_load_predict_end_to_end(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    log_dir = tmp_path / "logs"

    model = IS(default_root_dir=str(log_dir), **TINY_KWARGS)
    model.fit(train_dataset, config=training_config())

    # Training actually learned: the epoch-mean train_loss fell over the run.
    train_losses = logged_metric(log_dir, "train_loss")
    assert train_losses[-1] < train_losses[0]

    checkpoint_path = tmp_path / "inversynth_is.pt"
    model.save(checkpoint_path)
    assert checkpoint_path.exists()

    # Fresh instance loads with no dataset and no VST, then predicts.
    reloaded = IS(**TINY_KWARGS)
    reloaded.load(checkpoint_path)

    audio, _ = train_dataset[0]
    prediction = reloaded.predict(audio)

    space = train_dataset.parameter_space
    assert set(prediction) == set(space.names)              # all parameters present
    assert prediction["CAT"] in (0.0, 0.5, 1.0)             # categorical snapped to a grid option
    assert 0.0 <= prediction["AMP"] <= 1.0                  # continuous clipped into bounds


# -- Stage 3: IS2 (inference-time finetuning) --------------------------------

# Tiny ITF settings so the per-sample finetuning loop stays fast on CPU.
TINY_ITF_KWARGS = dict(itf_steps=2, itf_batch_size=4, itf_pool_size=8)


def test_is2_fit_export_load_predict_end_to_end(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    log_dir = tmp_path / "logs"

    model = IS2(default_root_dir=str(log_dir), proxy_dropout=0.0, **TINY_ITF_KWARGS, **TINY_KWARGS)
    model.fit(train_dataset, config=training_config())

    checkpoint_path = tmp_path / "inversynth_is2.pt"
    model.save(checkpoint_path)

    reloaded = IS2(proxy_dropout=0.0, **TINY_ITF_KWARGS, **TINY_KWARGS)
    reloaded.load(checkpoint_path)

    # The checkpoint carries the cached ITF training pool (L_B needs it offline).
    assert reloaded._itf_pool_audio is not None
    assert reloaded._itf_pool_audio.shape[0] == TINY_ITF_KWARGS["itf_pool_size"]

    audio, _ = train_dataset[0]
    prediction = reloaded.predict(audio)

    space = train_dataset.parameter_space
    assert set(prediction) == set(space.names)              # all parameters present
    assert prediction["CAT"] in (0.0, 0.5, 1.0)             # categorical snapped to a grid option
    assert 0.0 <= prediction["AMP"] <= 1.0                  # continuous clipped into bounds


def test_is2_restores_encoder_weights_after_predict(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    model = IS2(default_root_dir=str(tmp_path / "logs"), proxy_dropout=0.0,
                **TINY_ITF_KWARGS, **TINY_KWARGS)
    model.fit(train_dataset, config=training_config())

    encoder = model._network.encoder
    before = {name: tensor.clone() for name, tensor in encoder.state_dict().items()}
    model.predict(train_dataset[0][0])
    after = encoder.state_dict()

    # ITF is per-sample: theta* is restored, so the model is unchanged for the next call.
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name])


def test_is2_zero_steps_falls_back_to_plain_encoder(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    model = IS2(default_root_dir=str(tmp_path / "logs"), proxy_dropout=0.0,
                itf_steps=0, itf_batch_size=4, itf_pool_size=8, **TINY_KWARGS)
    model.fit(train_dataset, config=training_config())

    audio, _ = train_dataset[0]
    space = train_dataset.parameter_space
    encoder = model._network.encoder
    encoder.eval()
    with torch.no_grad():
        plain_vector = encoder(audio.unsqueeze(0)).squeeze(0).cpu().numpy()
    plain_prediction = space.ml_vector_to_synth_dict(plain_vector)

    # With no ITF alternations, predict keeps theta* -- the plain-encoder prediction.
    assert model.predict(audio) == plain_prediction


def test_is2_render_callback_is_used_and_receives_synth_dicts(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    model = IS2(default_root_dir=str(tmp_path / "logs"), proxy_dropout=0.0,
                **TINY_ITF_KWARGS, **TINY_KWARGS)
    model.fit(train_dataset, config=training_config())

    audio, _ = train_dataset[0]
    space = train_dataset.parameter_space
    calls = []

    def fake_render(predicted_dict):
        calls.append(predicted_dict)
        return audio.numpy()  # any waveform of the corpus length; contents don't matter here

    model.set_itf_render_callback(fake_render)
    prediction = model.predict(audio)

    # The real-synth monitor scores theta* once, then once per ITF alternation.
    assert len(calls) == TINY_ITF_KWARGS["itf_steps"] + 1
    # Each render request is a full, valid synth-side dict (default params merged in by the caller).
    for predicted_dict in calls:
        assert set(predicted_dict) == set(space.names)
        assert predicted_dict["CAT"] in (0.0, 0.5, 1.0)
    assert set(prediction) == set(space.names)


def test_is2_render_callback_governs_step_selection(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    model = IS2(default_root_dir=str(tmp_path / "logs"), proxy_dropout=0.0,
                **TINY_ITF_KWARGS, **TINY_KWARGS)
    model.fit(train_dataset, config=training_config())

    audio, _ = train_dataset[0]
    space = train_dataset.parameter_space
    encoder = model._network.encoder
    encoder.eval()
    with torch.no_grad():
        plain_vector = encoder(audio.unsqueeze(0)).squeeze(0).cpu().numpy()
    plain_prediction = space.ml_vector_to_synth_dict(plain_vector)

    # A monitor that renders the target itself scores a perfect MAE 0 at theta*, so no ITF
    # alternation can beat it: selection falls back to theta* -- the plain-encoder prediction.
    # Proves the injected render (not the proxy) drives step selection and fallback.
    model.set_itf_render_callback(lambda predicted_dict: audio.numpy())
    assert model.predict(audio) == plain_prediction


def test_is2_load_rejects_checkpoint_without_itf_pool(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    # An IS2xITF checkpoint trains the same network but carries no ITF pool.
    is2xitf = IS2xITF(default_root_dir=str(tmp_path / "logs"), proxy_dropout=0.0, **TINY_KWARGS)
    is2xitf.fit(train_dataset, config=training_config())
    checkpoint_path = tmp_path / "inversynth_is2xitf.pt"
    is2xitf.save(checkpoint_path)

    with pytest.raises(RuntimeError, match="ITF training pool"):
        IS2(proxy_dropout=0.0, **TINY_KWARGS).load(checkpoint_path)


def test_is2xitf_fit_export_load_predict_end_to_end(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    log_dir = tmp_path / "logs"

    model = IS2xITF(default_root_dir=str(log_dir), proxy_dropout=0.0, **TINY_KWARGS)
    model.fit(train_dataset, config=training_config())

    # The combined loss actually fell, and the proxy's audio loss was part of training.
    train_losses = logged_metric(log_dir, "train_loss")
    assert train_losses[-1] < train_losses[0]
    assert logged_metric(log_dir, "train_audio_loss")

    checkpoint_path = tmp_path / "inversynth_is2xitf.pt"
    model.save(checkpoint_path)
    assert checkpoint_path.exists()

    reloaded = IS2xITF(proxy_dropout=0.0, **TINY_KWARGS)
    reloaded.load(checkpoint_path)

    # The checkpoint carries both the encoder and the training-only proxy (Stage 3 needs it).
    state_keys = reloaded._network.state_dict().keys()
    assert any(key.startswith("encoder.") for key in state_keys)
    assert any(key.startswith("proxy.") for key in state_keys)

    audio, _ = train_dataset[0]
    prediction = reloaded.predict(audio)

    space = train_dataset.parameter_space
    assert set(prediction) == set(space.names)              # all parameters present
    assert prediction["CAT"] in (0.0, 0.5, 1.0)             # categorical snapped to a grid option
    assert 0.0 <= prediction["AMP"] <= 1.0                  # continuous clipped into bounds
