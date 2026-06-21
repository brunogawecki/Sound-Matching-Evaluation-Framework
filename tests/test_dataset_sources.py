import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.parameter_space import ParameterSpecification, ParameterSpace
from dataset.dexed_preset_loader import LoadedPreset
from dataset.sources import (
    METHOD_AUGMENT,
    METHOD_HUMAN,
    METHOD_SYNTHETIC,
    HybridSource,
    PresetRecord,
    HumanPresetSource,
    SyntheticSampler,
)


# ---------------------------------------------------------------------------
# All pure-Python: no plugin required.
# ---------------------------------------------------------------------------

def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="CONT A", kind="continuous", default=0.25),
        ParameterSpecification(name="CAT B", kind="categorical", options=[0.0, 0.5, 1.0], default=0.5),
        ParameterSpecification(name="CONT C", kind="continuous", bounds=(0.2, 0.8), default=0.5),
        ParameterSpecification(name="CAT D", kind="categorical", options=[0.0, 1.0], default=0.0),
    ])


def human_preset(name: str = "VOICE", source: str = "bank.syx", voice_index: int = 0) -> PresetRecord:
    return PresetRecord(
        params={"CONT A": 0.4, "CAT B": 1.0, "CONT C": 0.5, "CAT D": 0.0},
        method=METHOD_HUMAN,
        partition="train",
        source_file=source,
        voice_index=voice_index,
        voice_name=name,
    )


# -- SyntheticSampler --------------------------------------------------------

def test_synthetic_sampler_yields_exactly_count_subset_presets():
    space = make_space()
    presets = list(SyntheticSampler(space, count=7, seed=0).iter_presets())
    assert len(presets) == 7
    for preset in presets:
        assert set(preset.params) == set(space.names)
        assert preset.method == METHOD_SYNTHETIC
        assert preset.partition == "train"


def test_synthetic_sampler_is_deterministic_in_seed_and_order_independent():
    space = make_space()
    first = list(SyntheticSampler(space, count=5, seed=42).iter_presets())
    second = list(SyntheticSampler(space, count=5, seed=42).iter_presets())
    assert [p.params for p in first] == [p.params for p in second]
    # Slot 3 is the same preset regardless of how many presets precede it.
    longer = list(SyntheticSampler(space, count=10, seed=42).iter_presets())
    assert longer[3].params == first[3].params


def test_synthetic_resample_redraws_the_same_slot_differently():
    space = make_space()
    sampler = SyntheticSampler(space, count=3, seed=1)
    original = list(sampler.iter_presets())[0]
    redrawn = sampler.resample(original, attempt=1)
    assert redrawn.slot == original.slot
    assert redrawn.params != original.params
    # Resampling is itself deterministic.
    assert sampler.resample(original, attempt=1).params == redrawn.params


# -- HumanPresetSource ------------------------------------------------------------

def test_preset_source_projects_onto_subset_and_tags_provenance():
    space = make_space()
    full = {"CONT A": 0.4, "CAT B": 1.0, "CONT C": 0.5, "CAT D": 0.0, "DROPPED": 0.9}
    presets = [LoadedPreset(params=full, source_file="bank.syx", voice_index=2, voice_name="LEAD")]
    presets = list(HumanPresetSource(presets, space, partition="test").iter_presets())
    assert len(presets) == 1
    preset = presets[0]
    assert set(preset.params) == set(space.names)  # DROPPED is gone
    assert preset.method == METHOD_HUMAN
    assert preset.partition == "test"
    assert preset.source_file == "bank.syx"
    assert preset.voice_index == 2
    assert preset.voice_name == "LEAD"


def test_preset_source_rejects_preset_missing_subset_params():
    space = make_space()
    presets = [LoadedPreset(params={"CONT A": 0.1}, source_file="b.syx", voice_index=0, voice_name="X")]
    with pytest.raises(KeyError):
        list(HumanPresetSource(presets, space, partition="train").iter_presets())


def test_preset_source_cannot_resample():
    space = make_space()
    presets = [LoadedPreset(
        params={"CONT A": 0.4, "CAT B": 1.0, "CONT C": 0.5, "CAT D": 0.0},
        source_file="b.syx", voice_index=0, voice_name="X")]
    source = HumanPresetSource(presets, space, partition="train")
    preset = list(source.iter_presets())[0]
    assert source.resample(preset, attempt=1) is None


# -- HybridSource: blend -----------------------------------------------------

def test_hybrid_blend_respects_synthetic_ratio_approximately():
    space = make_space()
    humans = [human_preset(voice_index=i) for i in range(4)]
    source = HybridSource(
        HybridSource.BLEND, humans, space, count=400, seed=7, synthetic_ratio=0.75
    )
    presets = list(source.iter_presets())
    synthetic_fraction = np.mean([p.method == METHOD_SYNTHETIC for p in presets])
    assert 0.65 < synthetic_fraction < 0.85
    assert all(p.method in (METHOD_SYNTHETIC, METHOD_HUMAN) for p in presets)


def test_hybrid_blend_human_picks_cannot_resample_but_synthetic_can():
    space = make_space()
    source = HybridSource(HybridSource.BLEND, [human_preset()], space, count=20, seed=3, synthetic_ratio=0.5)
    for preset in source.iter_presets():
        result = source.resample(preset, attempt=1)
        if preset.method == METHOD_HUMAN:
            assert result is None
        else:
            assert result is not None and result.method == METHOD_SYNTHETIC


# -- HybridSource: augment ---------------------------------------------------

def test_hybrid_augment_perturbs_only_k_params_and_records_parent():
    space = make_space()
    parent = human_preset(source="cool.syx", voice_index=5)
    source = HybridSource(
        HybridSource.AUGMENT, [parent], space, count=10, seed=9,
        num_perturbed_params=1, jitter=0.05,
    )
    for preset in source.iter_presets():
        assert preset.method == METHOD_AUGMENT
        assert preset.parent_id == "cool.syx:5"
        assert preset.source_file == "cool.syx" and preset.voice_index == 5
        changed = [n for n in space.names if preset.params[n] != parent.params[n]]
        # Only continuous params are eligible by default; at most k change.
        assert len(changed) <= 1
        for name in changed:
            assert name in ("CONT A", "CONT C")


def test_hybrid_augment_keeps_continuous_perturbations_in_bounds():
    space = make_space()
    # Parent pinned at an upper bound so jitter must clip.
    parent = PresetRecord(
        params={"CONT A": 1.0, "CAT B": 0.0, "CONT C": 0.8, "CAT D": 0.0},
        method=METHOD_HUMAN, partition="train", source_file="b.syx", voice_index=0, voice_name="X",
    )
    source = HybridSource(
        HybridSource.AUGMENT, [parent], space, count=30, seed=2,
        num_perturbed_params=2, jitter=0.5,
    )
    for preset in source.iter_presets():
        assert 0.0 <= preset.params["CONT A"] <= 1.0
        assert 0.2 <= preset.params["CONT C"] <= 0.8


def test_hybrid_augment_can_flip_categoricals_when_enabled():
    space = make_space()
    parent = human_preset()
    source = HybridSource(
        HybridSource.AUGMENT, [parent], space, count=60, seed=11,
        num_perturbed_params=4, jitter=0.01, flip_categoricals=True,
    )
    flipped_a_categorical = False
    for preset in source.iter_presets():
        for name in ("CAT B", "CAT D"):
            if preset.params[name] != parent.params[name]:
                flipped_a_categorical = True
                spec = next(s for s in space.parameter_specs if s.name == name)
                assert preset.params[name] in spec.options
    assert flipped_a_categorical


def test_hybrid_rejects_unknown_mode_and_empty_humans():
    space = make_space()
    with pytest.raises(ValueError):
        HybridSource("interpolate", [human_preset()], space, count=1, seed=0)
    with pytest.raises(ValueError):
        HybridSource(HybridSource.BLEND, [], space, count=1, seed=0)
