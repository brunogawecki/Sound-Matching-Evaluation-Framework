"""Tests for the InverSynth II ``IS`` model (Stage 1 of the neural-proxy family).

A forward-shape and rebuild-determinism check on the encoder network, plus an end-to-end
fit -> save -> load -> predict smoke test on a tiny synthetic corpus (mirroring
``tests/test_sound2synth.py``). A small STFT + few mels keep the deep conv stack from
collapsing the tiny spectrogram and the run fast on CPU. Skips cleanly when
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
from models.inversynth2 import IS, InverSynthEncoderNetwork
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
