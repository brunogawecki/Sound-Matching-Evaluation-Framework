"""Tests for the Evaluator (Layer 4).

Logic tests (metric routing, nanmean/summary, persistence, NaN handling) run on a
tiny on-disk corpus with a **fake** render backend and a **fake** model -- no VST.
The real re-render path gets a plugin-dependent test that auto-skips when Dexed (or
the reference fresh-process corpus) is absent.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import evaluation.evaluator as evaluator_module
from evaluation.evaluator import Evaluator
from evaluation.registry import metric_names
from synth.parameter_space import ParameterSpace, ParameterSpecification

SAMPLE_RATE = 22050
DURATION_SAMPLES = SAMPLE_RATE  # 1 second


def _sine(frequency: float = 440.0) -> np.ndarray:
    time = np.arange(DURATION_SAMPLES) / SAMPLE_RATE
    return (0.5 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def _small_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.5),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 1.0], default=0.0),
    ])


def _write_corpus(corpus_dir, waveforms, params_rows) -> None:
    """Write a minimal builder-shaped corpus: run_summary + metadata + audio WAVs."""
    space = _small_space()
    audio_dir = corpus_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (waveform, params) in enumerate(zip(waveforms, params_rows)):
        sample_id = f"sample_{index:06d}"
        relative_path = f"audio/{sample_id}.wav"
        wavfile.write(str(corpus_dir / relative_path), SAMPLE_RATE, waveform.astype(np.float32))
        rows.append({"sample_id": sample_id, "audio_path": relative_path, **params})
    pd.DataFrame(rows, columns=["sample_id", "audio_path"] + space.names).to_csv(
        corpus_dir / "metadata.csv", index=False
    )

    summary = {
        "run_name": corpus_dir.name,
        "num_samples": len(rows),
        "render_settings": {
            "midi_note": 60, "velocity": 100, "duration_sec": 1.0, "note_duration_sec": 0.8,
        },
        "renderer": "dawdreamer",
        "sample_rate": SAMPLE_RATE,
        "parameter_space": space.to_dict(),
        "default_params": {"AMP": 0.5, "CAT": 0.0},
        "git_revision": "deadbeef",
    }
    with open(corpus_dir / "run_summary.json", "w") as summary_file:
        json.dump(summary, summary_file)


class _FakeBackend:
    """Stands in for FreshProcessRenderBackend: returns a fixed waveform, no VST."""

    def __init__(self, settings, renderer="dawdreamer"):
        self.settings = settings
        self.renderer = renderer
        self.closed = False
        self.rendered_params = []

    def render(self, params):
        self.rendered_params.append(params)
        return _sine()

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
def corpus(tmp_path, monkeypatch):
    """A 3-sample corpus (two audible sines, one silent) + a patched fake backend."""
    from dataset.torch_dataset import RenderedCorpusDataset

    corpus_dir = tmp_path / "run_fake_test"
    waveforms = [_sine(440.0), _sine(440.0), np.zeros(DURATION_SAMPLES, dtype=np.float32)]
    params_rows = [
        {"AMP": 0.2, "CAT": 0.0},
        {"AMP": 0.8, "CAT": 1.0},
        {"AMP": 0.5, "CAT": 0.0},
    ]
    _write_corpus(corpus_dir, waveforms, params_rows)
    monkeypatch.setattr(evaluator_module, "FreshProcessRenderBackend", _FakeBackend)
    return RenderedCorpusDataset.load(corpus_dir)


# -- routing + persistence ---------------------------------------------------

def test_per_sample_matrix_shape_and_columns(corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results")

    assert list(result.per_sample_metrics.columns) == ["sample_id"] + metric_names()
    assert len(result.per_sample_metrics) == 3
    assert list(result.per_sample_metrics["sample_id"]) == ["sample_000000", "sample_000001", "sample_000002"]


def test_predict_called_once_per_sample(corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results")
    assert len(model.seen_audio) == 3
    # predict receives the audio tensor, not the target vector.
    assert all(isinstance(audio, torch.Tensor) for audio in model.seen_audio)


def test_re_render_merges_prediction_over_defaults(corpus, tmp_path, monkeypatch):
    # Re-instantiate the patched backend so we can inspect what it was asked to render.
    captured = {}

    def _factory(settings, renderer="dawdreamer"):
        backend = _FakeBackend(settings, renderer)
        captured["backend"] = backend
        return backend

    monkeypatch.setattr(evaluator_module, "FreshProcessRenderBackend", _factory)
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results")

    rendered = captured["backend"].rendered_params[0]
    # default_params ({"AMP":0.5,"CAT":0.0}) overridden by the prediction.
    assert rendered == {"AMP": 0.6, "CAT": 1.0}
    assert captured["backend"].closed is True


def test_writes_both_files_under_corpus_and_model(corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results")

    expected_dir = tmp_path / "results" / "run_fake_test" / "_FakeModel"
    assert result.per_sample_metrics_path == expected_dir / "per_sample.csv"
    assert result.summary_path == expected_dir / "eval_summary.json"
    assert result.per_sample_metrics_path.exists()
    assert result.summary_path.exists()

    reloaded = pd.read_csv(result.per_sample_metrics_path)
    assert len(reloaded) == 3
    assert list(reloaded.columns) == ["sample_id"] + metric_names()


# -- prediction audio persistence (D-EVAL update) -----------------------------

def test_save_audio_off_by_default_writes_no_audio_dir(corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results")
    assert not (result.per_sample_metrics_path.parent / "audio").exists()


def test_save_audio_writes_capped_seeded_random_subset(corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(corpus).evaluate(
        model, out_dir=tmp_path / "results", save_audio=True, save_audio_n=2
    )
    audio_dir = result.per_sample_metrics_path.parent / "audio"
    wav_files = sorted(audio_dir.glob("*.wav"))
    assert len(wav_files) == 2
    for wav_path in wav_files:
        sample_rate, waveform = wavfile.read(wav_path)
        assert sample_rate == SAMPLE_RATE
        assert waveform.dtype == np.float32


def test_save_audio_n_caps_at_corpus_size(corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(corpus).evaluate(
        model, out_dir=tmp_path / "results", save_audio=True, save_audio_n=1000
    )
    audio_dir = result.per_sample_metrics_path.parent / "audio"
    assert len(list(audio_dir.glob("*.wav"))) == 3


# -- progress reporting -------------------------------------------------------

def test_show_progress_draws_a_bar_over_the_samples(corpus, tmp_path, capsys):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results", show_progress=True)

    # tqdm writes to stderr, which the dashboard merges into the log tail it shows.
    progress_output = capsys.readouterr().err
    assert "Evaluating" in progress_output
    assert "3/3" in progress_output


def test_no_progress_bar_by_default(corpus, tmp_path, capsys):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results")
    assert "Evaluating" not in capsys.readouterr().err


# -- NaN handling + aggregation ----------------------------------------------

def test_nan_valid_counts_reflect_silent_sample(corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results")
    per_metric = result.summary["per_metric"]

    # The silent third target makes spectral_convergence / f0_rmse undefined there,
    # so two of three samples are valid; magnitude metrics stay finite for all three.
    assert per_metric["spectral_convergence"]["valid_count"] == 2
    assert per_metric["f0_rmse"]["valid_count"] == 2
    assert per_metric["lsd"]["valid_count"] == 3
    assert np.isnan(result.per_sample_metrics["spectral_convergence"].iloc[2])
    assert np.isfinite(result.per_sample_metrics["lsd"]).all()


def test_summary_mean_ignores_nan(corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results")

    for name in metric_names():
        values = result.per_sample_metrics[name].to_numpy(dtype=float)
        valid = values[~np.isnan(values)]
        reported = result.summary["per_metric"][name]
        if valid.size:
            assert reported["mean"] == pytest.approx(float(np.mean(valid)))
            assert reported["std"] == pytest.approx(float(np.std(valid)))
        assert reported["valid_count"] == int(valid.size)


# -- summary metadata --------------------------------------------------------

def test_summary_echoes_corpus_contract_and_fingerprint(corpus, tmp_path):
    checkpoint = tmp_path / "ckpt.json"
    checkpoint.write_text('{"hello": 1}')
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(corpus).evaluate(
        model, checkpoint_path=checkpoint, out_dir=tmp_path / "results"
    )
    summary = result.summary

    assert summary["model_class"] == "_FakeModel"
    assert summary["num_samples"] == 3
    assert summary["render_contract"]["sample_rate"] == SAMPLE_RATE
    assert summary["render_contract"]["renderer"] == "dawdreamer"
    assert summary["render_contract"]["render_settings"]["midi_note"] == 60
    assert summary["corpus"]["git_revision"] == "deadbeef"
    assert summary["corpus"]["name"] == "run_fake_test"
    assert summary["metrics"] == metric_names()
    assert summary["checkpoint"]["sha256"] is not None
    assert summary["checkpoint"]["path"] == str(checkpoint)


def test_missing_checkpoint_records_null_hash(corpus, tmp_path):
    model = _FakeModel({"AMP": 0.6, "CAT": 1.0})
    result = Evaluator(corpus).evaluate(model, checkpoint_path=None, out_dir=tmp_path / "results")
    assert result.summary["checkpoint"]["sha256"] is None
    assert result.summary["checkpoint"]["note"]


# -- contract validation -----------------------------------------------------

def test_missing_render_field_is_a_hard_error(tmp_path, monkeypatch):
    from dataset.torch_dataset import RenderedCorpusDataset

    corpus_dir = tmp_path / "run_bad"
    _write_corpus(corpus_dir, [_sine()], [{"AMP": 0.2, "CAT": 0.0}])
    # Drop a required render-contract field from the summary.
    summary = json.loads((corpus_dir / "run_summary.json").read_text())
    del summary["renderer"]
    (corpus_dir / "run_summary.json").write_text(json.dumps(summary))

    monkeypatch.setattr(evaluator_module, "FreshProcessRenderBackend", _FakeBackend)
    corpus = RenderedCorpusDataset.load(corpus_dir)
    with pytest.raises(ValueError, match="render-contract fields"):
        Evaluator(corpus)


# -- plugin-dependent: perfect-prediction floor (D-REPRO) --------------------

REFERENCE_CORPUS = os.path.join(config.DATASET_DIR, "run_A_test")
DEXED_PRESENT = os.path.exists(os.path.expanduser(config.DEXED_PATH))
REFERENCE_PRESENT = os.path.exists(os.path.join(REFERENCE_CORPUS, "run_summary.json"))


@pytest.mark.skipif(
    not (DEXED_PRESENT and REFERENCE_PRESENT),
    reason="needs the Dexed VST and the run_A_test fresh-process corpus",
)
def test_true_parameters_floor_audio_metrics_at_zero(tmp_path):
    """Feeding a sample's true parameters back through the real re-render path should
    score the audio metrics at ~0: target and re-render share a clean pos-0 context."""
    from dataset.torch_dataset import RenderedCorpusDataset

    reference = RenderedCorpusDataset.load(REFERENCE_CORPUS)
    # Build a 1-sample corpus from the reference's first sample.
    one_dir = tmp_path / "run_one"
    (one_dir / "audio").mkdir(parents=True)
    shutil.copy(reference.corpus_dir / "run_summary.json", one_dir / "run_summary.json")
    first = reference.metadata.iloc[[0]].copy()
    relative_path = first.iloc[0]["audio_path"]
    (one_dir / os.path.dirname(relative_path)).mkdir(parents=True, exist_ok=True)
    shutil.copy(reference.corpus_dir / relative_path, one_dir / relative_path)
    first.to_csv(one_dir / "metadata.csv", index=False)

    corpus = RenderedCorpusDataset.load(one_dir)
    true_params = {name: float(first.iloc[0][name]) for name in corpus.parameter_space.names}
    model = _FakeModel(true_params)

    result = Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results")
    assert result.summary["per_metric"]["lsd"]["mean"] == pytest.approx(0.0, abs=1e-3)
    assert result.summary["per_metric"]["mel_mae"]["mean"] == pytest.approx(0.0, abs=1e-3)
