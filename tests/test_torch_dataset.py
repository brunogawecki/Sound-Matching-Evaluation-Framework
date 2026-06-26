import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")  # the Dataset is the framework's first torch user

from synth.parameter_space import ParameterSpecification, ParameterSpace
from dataset.builder import DatasetBuilder, RenderSettings
from dataset.preset_sources import METHOD_SYNTHETIC, SyntheticPresetSource
from dataset.torch_dataset import RenderedCorpusDataset


# A tiny no-VST synth: a 220 Hz sine scaled by "AMP" (so loudness is controllable),
# plus a categorical "CAT". Mirrors the fake used in test_dataset_builder.
def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.8),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 0.5, 1.0], default=0.0),
    ])


class FakeSynth:
    renderer_name = "fake"

    def __init__(self, space: ParameterSpace, sample_rate: int = 8000):
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


SAMPLE_RATE = 8000
DURATION_SEC = 0.5
EXPECTED_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)


def build_corpus(tmp_path, count=6, seed=0):
    """Build a small real corpus with the actual DatasetBuilder; return its dir."""
    synth = FakeSynth(make_space(), sample_rate=SAMPLE_RATE)
    settings = RenderSettings(midi_note=60, velocity=100, duration_sec=DURATION_SEC, note_duration_sec=DURATION_SEC)
    # AMP is pinned high so renders clear the loudness gate (no near-silent redraw churn).
    source = SyntheticPresetSource(make_space(), count=count, seed=seed, sampling_ranges={"AMP": (0.7, 1.0)})
    DatasetBuilder(synth, render_settings=settings).build(source, run_name="corpus", output_root=tmp_path)
    return tmp_path / "corpus"


# -- contract ----------------------------------------------------------------

def test_len_matches_metadata_rows(tmp_path):
    corpus_dir = build_corpus(tmp_path, count=6)
    dataset = RenderedCorpusDataset.load(corpus_dir)
    assert len(dataset) == 6 == len(pd.read_csv(corpus_dir / "metadata.csv"))


def test_getitem_returns_audio_and_target_tensors(tmp_path):
    corpus_dir = build_corpus(tmp_path)
    dataset = RenderedCorpusDataset.load(corpus_dir)
    audio, target = dataset[0]
    assert audio.dtype == torch.float32 and target.dtype == torch.float32
    assert audio.shape == (EXPECTED_SAMPLES,)
    assert target.shape == (dataset.parameter_space.ml_dimension,)  # 1 + 3


def test_audio_matches_on_disk_wav(tmp_path):
    corpus_dir = build_corpus(tmp_path)
    dataset = RenderedCorpusDataset.load(corpus_dir)
    audio, _ = dataset[2]
    _, on_disk = wavfile.read(corpus_dir / dataset.metadata.iloc[2]["audio_path"])
    np.testing.assert_array_equal(audio.numpy(), on_disk.astype(np.float32))


def test_target_roundtrips_to_the_rows_parameters(tmp_path):
    corpus_dir = build_corpus(tmp_path)
    dataset = RenderedCorpusDataset.load(corpus_dir)
    space = dataset.parameter_space
    for index in range(len(dataset)):
        _, target = dataset[index]
        decoded = space.ml_vector_to_synth_dict(target.numpy())
        row = dataset.metadata.iloc[index]  # the row's true synth-side params
        assert decoded["CAT"] == row["CAT"]               # categorical exact
        assert decoded["AMP"] == pytest.approx(row["AMP"])  # continuous in place


# -- offline / no-VST path ---------------------------------------------------

def test_load_needs_no_parameter_space_argument(tmp_path):
    corpus_dir = build_corpus(tmp_path, count=3)
    dataset = RenderedCorpusDataset.load(corpus_dir)
    assert dataset.parameter_space.names == ["AMP", "CAT"]
    assert dataset.parameter_space.ml_dimension == 4


def test_load_errors_clearly_when_space_not_serialized(tmp_path):
    import json
    corpus_dir = build_corpus(tmp_path, count=2)
    summary_path = corpus_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.pop("parameter_space")
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="parameter_space"):
        RenderedCorpusDataset.load(corpus_dir)


# -- target-only access ------------------------------------------------------

def test_targets_property_is_the_full_matrix(tmp_path):
    corpus_dir = build_corpus(tmp_path, count=5)
    dataset = RenderedCorpusDataset.load(corpus_dir)
    assert dataset.targets.shape == (5, dataset.parameter_space.ml_dimension)


# -- validation --------------------------------------------------------------

def test_mismatched_parameter_space_raises(tmp_path):
    corpus_dir = build_corpus(tmp_path, count=2)
    wrong_space = ParameterSpace([ParameterSpecification(name="NOPE", kind="continuous")])
    with pytest.raises(ValueError, match="missing parameter columns"):
        RenderedCorpusDataset(corpus_dir, wrong_space)


# -- batching ----------------------------------------------------------------

def test_default_collate_batches_cleanly(tmp_path):
    corpus_dir = build_corpus(tmp_path, count=8)
    dataset = RenderedCorpusDataset.load(corpus_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    audio, target = next(iter(loader))
    assert audio.shape == (4, EXPECTED_SAMPLES)
    assert target.shape == (4, dataset.parameter_space.ml_dimension)
