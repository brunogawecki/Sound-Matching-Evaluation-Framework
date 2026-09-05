"""The Evaluator on an out-of-domain, audio-only corpus (D-OOD).

Same shape as ``tests/test_evaluator.py`` -- tiny on-disk corpus, fake render backend,
fake model, no VST -- but the corpus carries no ground-truth parameters. The contract
under test: the ten audio metrics stay finite and the three parameter metrics report
``NaN`` with a ``valid_count`` of 0, rather than being silently filled with a placeholder.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation.evaluator as evaluator_module
from dataset.ood_corpus import NO_TARGETS, load_eval_corpus
from evaluation.evaluator import Evaluator
from evaluation.registry import METRIC_PANEL, metric_names
from synth.parameter_space import ParameterSpace, ParameterSpecification

SAMPLE_RATE = 22050
DURATION_SAMPLES = SAMPLE_RATE  # 1 second

PARAMETER_METRICS = [spec.name for spec in METRIC_PANEL if spec.input_type == "parameter"]
AUDIO_METRICS = [spec.name for spec in METRIC_PANEL if spec.input_type == "audio"]


def _sine(frequency: float = 440.0) -> np.ndarray:
    time = np.arange(DURATION_SAMPLES) / SAMPLE_RATE
    return (0.5 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def _small_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.5),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 1.0], default=0.0),
    ])


def _write_ood_corpus(corpus_dir, waveforms) -> None:
    """A minimal out-of-domain corpus: audio + provenance metadata, no parameter columns."""
    space = _small_space()
    (corpus_dir / "audio").mkdir(parents=True, exist_ok=True)

    rows = []
    for index, waveform in enumerate(waveforms):
        sample_id = f"sample_{index:06d}"
        relative_path = f"audio/{sample_id}.wav"
        wavfile.write(str(corpus_dir / relative_path), SAMPLE_RATE, waveform.astype(np.float32))
        rows.append({
            "sample_id": sample_id, "audio_path": relative_path,
            "nsynth_note_str": f"flute_acoustic_{index:03d}-060-100",
            "instrument_family_str": "flute",
        })
    pd.DataFrame(rows).to_csv(corpus_dir / "metadata.csv", index=False)

    summary = {
        "run_name": corpus_dir.name,
        "num_samples": len(rows),
        "targets": NO_TARGETS,
        "domain": "out_of_domain",
        "render_settings": {
            "midi_note": 60, "velocity": 100, "duration_sec": 1.0, "note_duration_sec": 0.8,
        },
        "renderer": "dawdreamer",
        "sample_rate": SAMPLE_RATE,
        "synth": "dexed",
        "parameter_space": space.to_dict(),
        "default_params": {"AMP": 0.5, "CAT": 0.0},
        "git_revision": "deadbeef",
    }
    with open(corpus_dir / "run_summary.json", "w") as summary_file:
        json.dump(summary, summary_file)


class _FakeBackend:
    """Stands in for FreshProcessRenderBackend: returns a fixed waveform, no VST."""

    def __init__(self, settings, renderer="dawdreamer", synth_name="dexed"):
        self.rendered_params = []
        self.closed = False

    def render(self, params):
        self.rendered_params.append(params)
        return _sine(660.0)

    def close(self):
        self.closed = True


class _FakeModel:
    """Returns a fixed synth-side prediction, ignoring the audio."""

    def __init__(self, prediction):
        self._prediction = prediction
        self.seen_audio = []

    def predict(self, audio):
        self.seen_audio.append(audio)
        return dict(self._prediction)


@pytest.fixture
def ood_corpus(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "nsynth_fake_test"
    _write_ood_corpus(corpus_dir, [_sine(440.0), _sine(880.0)])
    monkeypatch.setattr(evaluator_module, "FreshProcessRenderBackend", _FakeBackend)
    return load_eval_corpus(corpus_dir)


def test_parameter_metrics_are_nan_and_audio_metrics_are_finite(ood_corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(ood_corpus).evaluate(model, out_dir=tmp_path / "results")

    assert list(result.per_sample_metrics.columns) == ["sample_id"] + metric_names()
    for name in PARAMETER_METRICS:
        assert result.per_sample_metrics[name].isna().all(), name
    for name in AUDIO_METRICS:
        assert np.isfinite(result.per_sample_metrics[name]).all(), name


def test_summary_reports_zero_valid_count_for_the_parameter_axis(ood_corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(ood_corpus).evaluate(model, out_dir=tmp_path / "results")
    per_metric = result.summary["per_metric"]

    for name in PARAMETER_METRICS:
        assert per_metric[name]["valid_count"] == 0, name
        assert np.isnan(per_metric[name]["mean"]), name
    for name in AUDIO_METRICS:
        assert per_metric[name]["valid_count"] == 2, name


def test_prediction_is_still_rendered_under_the_copied_contract(ood_corpus, tmp_path):
    """No ground truth, but predictions still re-render with default_params merged in."""
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    Evaluator(ood_corpus).evaluate(model, out_dir=tmp_path / "results")
    summary_path = tmp_path / "results" / "nsynth_fake_test" / "_FakeModel" / "eval_summary.json"
    with open(summary_path) as summary_file:
        summary = json.load(summary_file)
    assert summary["render_contract"]["render_settings"]["midi_note"] == 60
    assert summary["render_contract"]["sample_rate"] == SAMPLE_RATE
    assert summary["num_samples"] == 2


def test_results_land_under_the_ood_corpus_name(ood_corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(ood_corpus).evaluate(model, out_dir=tmp_path / "results")
    assert result.per_sample_metrics_path.parent.parent.name == "nsynth_fake_test"
    assert result.per_sample_metrics_path.exists()
    assert result.summary_path.exists()
