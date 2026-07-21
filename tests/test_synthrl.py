"""End-to-end tests for the SynthRL-p family (stage 1, no RL).

Registry wiring, a fit -> save -> load -> predict smoke test on a tiny synthetic corpus,
and a full run through the ``Evaluator`` (the Step 3 milestone: a real, evaluable family).
A small STFT + few mels + a small transformer keep it fast on CPU. Skips cleanly when
``torch`` / ``lightning`` / ``librosa`` are absent (training-only deps).
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

import evaluation.evaluator as evaluator_module
from dataset.builder import DatasetBuilder, RenderSettings
from dataset.preset_sources import SyntheticPresetSource
from dataset.torch_dataset import RenderedCorpusDataset
from evaluation.evaluator import Evaluator
from evaluation.registry import metric_names
from models.registry import MODEL_REGISTRY
from models.synthrl import SynthRLp
from synth.parameter_space import ParameterSpace, ParameterSpecification

SAMPLE_RATE = 16000
DURATION_SEC = 1.0
EXPECTED_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

# Small front-end + transformer so the run stays fast on CPU.
TINY_KWARGS = dict(
    num_bins=8,
    n_fft=512,
    hop_length=128,
    win_length=512,
    n_mels=64,
    mel_fmax=8000.0,
    d_model=32,
    num_conv_layers=3,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    feedforward_dim=64,
    dropout=0.0,
)


def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.8),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 0.5, 1.0], default=0.0),
    ])


class FakeSynth:
    """A no-VST sine synth (mirrors tests/test_inversynth2.py)."""

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


class _FakeBackend:
    """Stands in for FreshProcessRenderBackend at eval time: a fixed sine, no VST."""

    def __init__(self, settings, renderer="dawdreamer"):
        self.settings = settings
        self.renderer = renderer
        self.closed = False

    def render(self, params):
        time = np.arange(EXPECTED_SAMPLES) / SAMPLE_RATE
        return (0.5 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)

    def close(self):
        self.closed = True


def build_corpus(tmp_path, run_name, count, seed) -> RenderedCorpusDataset:
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
        "data": {"batch_size": 4, "val_fraction": 0.25},
        "trainer": {
            "max_epochs": 5,
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


def test_registry_entry_constructs_synthrlp():
    registration = MODEL_REGISTRY["SynthRLp"]
    assert registration.model_class is SynthRLp
    assert registration.default_checkpoint_filename.endswith(".pt")


def test_fit_export_load_predict_end_to_end(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    log_dir = tmp_path / "logs"

    model = SynthRLp(default_root_dir=str(log_dir), **TINY_KWARGS)
    model.fit(train_dataset, config=training_config())

    # Training actually learned: the epoch-mean train_loss fell over the run.
    train_losses = logged_metric(log_dir, "train_loss")
    assert train_losses[-1] < train_losses[0]

    checkpoint_path = tmp_path / "synthrl_p.pt"
    model.save(checkpoint_path)
    assert checkpoint_path.exists()

    # Fresh instance loads with no dataset and no VST, then predicts through the representation.
    reloaded = SynthRLp(**TINY_KWARGS)
    reloaded.load(checkpoint_path)

    audio, _ = train_dataset[0]
    prediction = reloaded.predict(audio)

    space = train_dataset.parameter_space
    assert set(prediction) == set(space.names)     # all parameters present
    assert prediction["CAT"] in (0.0, 0.5, 1.0)    # categorical decoded to a grid option
    # Continuous decodes to a bin center, inside bounds.
    assert 0.0 <= prediction["AMP"] <= 1.0


def test_fit_then_evaluate_through_the_evaluator(tmp_path, monkeypatch):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    model = SynthRLp(default_root_dir=str(tmp_path / "logs"), **TINY_KWARGS)
    model.fit(train_dataset, config=training_config())

    # The Evaluator re-renders predictions; stub the fresh-process backend (no VST).
    monkeypatch.setattr(evaluator_module, "FreshProcessRenderBackend", _FakeBackend)
    corpus = build_corpus(tmp_path, "eval", count=4, seed=1)
    result = Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results")

    assert list(result.per_sample_metrics.columns) == ["sample_id"] + metric_names()
    assert len(result.per_sample_metrics) == 4
    assert result.summary_path.exists()
    assert result.summary["model_class"] == "SynthRLp"
