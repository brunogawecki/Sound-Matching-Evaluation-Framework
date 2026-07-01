"""End-to-end CPU smoke test for the training harness (issue #22).

Drives every harness component once with a tiny dummy network (``tests.tiny_deep_model``)
on a small corpus built by the real ``DatasetBuilder``: config -> DataModule ->
LightningRegressor -> build_trainer -> trainer.fit -> export checkpoint -> BaseDeepModel.load
-> predict. No GPU, no VST. Skips cleanly when Lightning is not installed (the
Lightning dep is cluster-only, D-FRAMEWORK).
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

from dataset.builder import DatasetBuilder, RenderSettings
from dataset.preset_sources import SyntheticPresetSource
from dataset.torch_dataset import RenderedCorpusDataset
from synth.parameter_space import ParameterSpace, ParameterSpecification

from tests.tiny_deep_model import TinyDeepModel

SAMPLE_RATE = 8000
DURATION_SEC = 0.5
EXPECTED_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)


def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.8),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 0.5, 1.0], default=0.0),
    ])


class FakeSynth:
    """A no-VST sine synth (mirrors tests/test_torch_dataset.py)."""

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


def training_config(tmp_path, seed=0):
    return {
        "seed": seed,
        "optimizer": {"learning_rate": 1e-2},
        "loss": {"categorical_loss_weight": 0.2},
        "data": {"batch_size": 4, "val_fraction": 0.25},
        "trainer": {
            "max_epochs": 2,
            "precision": "32-true",
            "accelerator": "cpu",
            "devices": 1,
            "log_every_n_steps": 1,
        },
    }


def logged_metric(log_dir, name) -> list:
    """The non-NaN values logged for ``name`` across a run's metrics.csv."""
    metrics_files = list(log_dir.rglob("metrics.csv"))
    assert metrics_files, "CSVLogger should have written a metrics.csv"
    return pd.read_csv(metrics_files[0])[name].dropna().tolist()


def test_fit_export_load_predict_end_to_end(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    log_dir = tmp_path / "logs"

    config = training_config(tmp_path)
    config["trainer"]["max_epochs"] = 5  # enough epochs for a reliable loss decrease
    model = TinyDeepModel(default_root_dir=str(log_dir))
    model.fit(train_dataset, config=config)

    # Training actually learned: the epoch-mean train_loss fell over the run.
    train_losses = logged_metric(log_dir, "train_loss")
    assert train_losses[-1] < train_losses[0]

    checkpoint_path = tmp_path / "tiny.pt"
    model.save(checkpoint_path)
    assert checkpoint_path.exists()

    # Fresh instance loads with no dataset and no VST, then predicts.
    reloaded = TinyDeepModel()
    reloaded.load(checkpoint_path)

    audio, _ = train_dataset[0]
    prediction = reloaded.predict(audio)

    space = train_dataset.parameter_space
    assert set(prediction) == set(space.names)              # all parameters present
    assert prediction["CAT"] in (0.0, 0.5, 1.0)             # categorical snapped to a grid option
    assert 0.0 <= prediction["AMP"] <= 1.0                  # continuous clipped into bounds


def test_explicit_validation_dataset_is_used(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=12, seed=0)
    validation_dataset = build_corpus(tmp_path, "val", count=8, seed=1)
    config = training_config(tmp_path)
    config["data"].pop("val_fraction")  # validation comes from the explicit corpus instead

    log_dir = tmp_path / "logs"
    model = TinyDeepModel(default_root_dir=str(log_dir))
    model.fit(train_dataset, validation_dataset=validation_dataset, config=config)
    # val_loss is only logged if the explicit validation dataset was actually consumed.
    assert logged_metric(log_dir, "val_loss")
    prediction = model.predict(train_dataset[0][0])
    assert set(prediction) == set(train_dataset.parameter_space.names)


def test_trains_without_validation(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    config = training_config(tmp_path)
    config["data"].pop("val_fraction")  # no explicit val set and no split -> no validation

    log_dir = tmp_path / "logs"
    model = TinyDeepModel(default_root_dir=str(log_dir))
    # Exercises monitor="train_loss" and val_dataloader() -> None (the default DataConfig).
    model.fit(train_dataset, config=config)

    train_losses = logged_metric(log_dir, "train_loss")
    assert train_losses  # training ran and logged an epoch-level train_loss
    prediction = model.predict(train_dataset[0][0])
    assert set(prediction) == set(train_dataset.parameter_space.names)


def test_csv_logger_writes_metrics(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    log_dir = tmp_path / "logs"
    model = TinyDeepModel(default_root_dir=str(log_dir))
    model.fit(train_dataset, config=training_config(tmp_path))
    metrics_files = list(log_dir.rglob("metrics.csv"))
    assert metrics_files, "CSVLogger should have written a metrics.csv"
    contents = metrics_files[0].read_text()
    assert "train_loss" in contents and "val_loss" in contents
