import json
import os
import sys
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from synth.parameter_space import ParameterSpecification, ParameterSpace
from dataset.builder import DatasetBuilder, RenderSettings
from dataset.preset_sources import METHOD_SYNTHETIC, PresetRecord, PresetSource, SyntheticPresetSource


# ---------------------------------------------------------------------------
# A fake synth: deterministic, fast, no VST. The rendered signal is a sine whose
# amplitude is the value of "AMP", so a preset's loudness is controllable (AMP=0
# is digital silence). A plain DC constant would read as silent under LUFS
# (K-weighting removes DC), so a real tone is used; the render is long enough and
# the sample rate high enough that one pyloudnorm loudness block (400 ms) fits.
# ---------------------------------------------------------------------------

def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.5),
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


def small_settings() -> RenderSettings:
    return RenderSettings(midi_note=60, velocity=100, duration_sec=0.5, note_duration_sec=0.5)


class ScriptedSource(PresetSource):
    """Yields one preset with a given AMP, then resamples to a fixed replacement."""

    def __init__(self, first_amp: float, resample_amp: Optional[float]):
        self._first_amp = first_amp
        self._resample_amp = resample_amp

    def _preset(self, amp: float) -> PresetRecord:
        return PresetRecord(params={"AMP": amp, "CAT": 0.0}, method=METHOD_SYNTHETIC, partition="train", slot=0)

    def iter_presets(self) -> Iterator[PresetRecord]:
        yield self._preset(self._first_amp)

    def resample(self, record, attempt) -> Optional[PresetRecord]:
        if self._resample_amp is None:
            return self._preset(self._first_amp)  # still silent
        return self._preset(self._resample_amp)

    def describe(self):
        return {"method": "scripted"}


def build(tmp_path, source, run_name="run", **kwargs) -> dict:
    synth = FakeSynth(make_space())
    builder = DatasetBuilder(synth, render_settings=small_settings(), **kwargs)
    return builder.build(source, run_name=run_name, output_root=tmp_path)


# -- corpus shape ------------------------------------------------------------

def test_build_writes_one_wav_per_row_all_mono(tmp_path):
    run_summary = build(tmp_path, SyntheticPresetSource(make_space(), count=6, seed=0))
    run_dir = tmp_path / "run"
    frame = pd.read_csv(run_dir / "metadata.csv")
    wavs = sorted((run_dir / "audio").glob("*.wav"))
    assert len(frame) == 6 == len(wavs)
    assert run_summary["num_samples"] == 6
    for _, row in frame.iterrows():
        rate, audio = wavfile.read(run_dir / row["audio_path"])
        assert audio.ndim == 1


def test_metadata_has_subset_columns_and_provenance(tmp_path):
    build(tmp_path, SyntheticPresetSource(make_space(), count=3, seed=1))
    frame = pd.read_csv(tmp_path / "run" / "metadata.csv")
    for column in ["sample_id", "audio_path", "AMP", "CAT", "method", "partition", "rms", "loudness_lufs", "near_silent"]:
        assert column in frame.columns
    assert (frame["method"] == METHOD_SYNTHETIC).all()
    assert frame["sample_id"].tolist() == ["sample_000000", "sample_000001", "sample_000002"]


def test_run_summary_records_settings_seed_subset_and_source(tmp_path):
    run_summary = build(tmp_path, SyntheticPresetSource(make_space(), count=2, seed=5))
    assert run_summary["sample_rate"] == 8000
    assert run_summary["renderer"] == "fake"
    assert run_summary["subset_names"] == ["AMP", "CAT"]
    assert run_summary["render_settings"]["duration_sec"] == 0.5
    assert run_summary["source"]["method"] == METHOD_SYNTHETIC
    assert run_summary["source"]["seed"] == 5
    assert set(run_summary["default_params"]) == {"AMP", "CAT"}
    # run_summary.json is on disk and valid JSON.
    with open(tmp_path / "run" / "run_summary.json") as handle:
        assert json.load(handle)["num_samples"] == 2


def test_run_summary_carries_a_reconstructable_parameter_space(tmp_path):
    build(tmp_path, SyntheticPresetSource(make_space(), count=2, seed=5))
    with open(tmp_path / "run" / "run_summary.json") as handle:
        summary = json.load(handle)
    restored = ParameterSpace.from_dict(summary["parameter_space"])
    assert restored.parameter_specs == make_space().parameter_specs


# -- determinism -------------------------------------------------------------

def test_same_seed_reproduces_identical_metadata_and_wavs(tmp_path):
    build(tmp_path, SyntheticPresetSource(make_space(), count=5, seed=99), run_name="a")
    build(tmp_path, SyntheticPresetSource(make_space(), count=5, seed=99), run_name="b")
    a_csv = (tmp_path / "a" / "metadata.csv").read_bytes()
    b_csv = (tmp_path / "b" / "metadata.csv").read_bytes()
    assert a_csv == b_csv
    for sample in range(5):
        name = f"audio/sample_{sample:06d}.wav"
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


# -- near-silence redraw -----------------------------------------------------

def test_near_silent_preset_is_redrawn_to_an_audible_one(tmp_path):
    build(tmp_path, ScriptedSource(first_amp=0.0, resample_amp=0.8))
    row = pd.read_csv(tmp_path / "run" / "metadata.csv").iloc[0]
    assert not bool(row["near_silent"])
    assert row["AMP"] == pytest.approx(0.8)


def test_persistently_silent_preset_is_kept_and_flagged(tmp_path):
    build(tmp_path, ScriptedSource(first_amp=0.0, resample_amp=None), max_redraw_attempts=3)
    row = pd.read_csv(tmp_path / "run" / "metadata.csv").iloc[0]
    assert bool(row["near_silent"])
    assert row["AMP"] == pytest.approx(0.0)


def test_audible_preset_is_not_resampled(tmp_path):
    # resample_amp would change AMP if it were ever called; it must not be.
    build(tmp_path, ScriptedSource(first_amp=0.7, resample_amp=0.1))
    row = pd.read_csv(tmp_path / "run" / "metadata.csv").iloc[0]
    assert row["AMP"] == pytest.approx(0.7)
    assert not bool(row["near_silent"])


def test_builder_rejects_preset_with_non_subset_params(tmp_path):
    class BadSource(PresetSource):
        def iter_presets(self):
            yield PresetRecord(params={"AMP": 0.5, "CAT": 0.0, "EXTRA": 1.0},
                              method=METHOD_SYNTHETIC, partition="train")
        def describe(self):
            return {}

    with pytest.raises(KeyError):
        build(tmp_path, BadSource())


# ---------------------------------------------------------------------------
# End-to-end with the live Dexed plugin (skips without the VST / cartridge).
# ---------------------------------------------------------------------------

PLUGIN_PATH = os.path.expanduser(config.DEXED_PATH)
CARTRIDGE_PATH = os.path.expanduser(
    os.getenv(
        "DEXED_TEST_CARTRIDGE",
        "~/Library/Application Support/DigitalSuburban/Dexed/Cartridges/Dexed_01.syx",
    )
)
needs_plugin = pytest.mark.skipif(
    not os.path.exists(PLUGIN_PATH), reason=f"Dexed plugin not found at {PLUGIN_PATH}"
)
needs_cartridge = pytest.mark.skipif(
    not os.path.exists(CARTRIDGE_PATH), reason=f"No test cartridge at {CARTRIDGE_PATH}"
)


@needs_plugin
def test_builds_synthetic_corpus_end_to_end_with_dexed(tmp_path):
    from synth.dexed import DexedWrapper

    synth = DexedWrapper(PLUGIN_PATH, sample_rate=config.SAMPLE_RATE, buffer_size=config.BUFFER_SIZE)
    source = SyntheticPresetSource(
        synth.parameter_space, count=8, seed=0, sampling_ranges=synth.audible_sampling_ranges
    )
    run_summary = DatasetBuilder(synth).build(source, run_name="synthetic", output_root=tmp_path)

    run_dir = tmp_path / "synthetic"
    frame = pd.read_csv(run_dir / "metadata.csv")
    wavs = sorted((run_dir / "audio").glob("*.wav"))
    assert len(frame) == 8 == len(wavs) == run_summary["num_samples"]
    assert len(run_summary["subset_names"]) == 103
    for _, row in frame.iterrows():
        _, audio = wavfile.read(run_dir / row["audio_path"])
        assert audio.ndim == 1
        assert len(audio) == int(config.DURATION_SEC * config.SAMPLE_RATE)


@needs_plugin
@needs_cartridge
def test_builds_human_corpus_end_to_end_with_dexed(tmp_path):
    from synth.dexed import DexedWrapper
    from dataset.dexed_preset_loader import DexedPresetLoader
    from dataset.preset_sources import HumanPresetSource

    synth = DexedWrapper(PLUGIN_PATH, sample_rate=config.SAMPLE_RATE, buffer_size=config.BUFFER_SIZE)
    split = DexedPresetLoader(synth.parameter_space, test_fraction=0.5).load([CARTRIDGE_PATH])
    presets = split.test[:4]
    source = HumanPresetSource(presets, synth.parameter_space, partition="test")
    run_summary = DatasetBuilder(synth).build(source, run_name="human", output_root=tmp_path)

    frame = pd.read_csv(tmp_path / "human" / "metadata.csv")
    assert len(frame) == len(presets) == run_summary["num_samples"]
    assert (frame["method"] == "human").all()
    assert (frame["partition"] == "test").all()
    assert frame["source_file"].notna().all()
