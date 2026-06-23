"""Dataset construction (Layer 2): preset sources and the rendering builder.

A :class:`~dataset.preset_sources.PresetSource` decides which presets exist (synthetic,
human, or hybrid); the :class:`~dataset.builder.DatasetBuilder` renders them
into a WAV + metadata corpus. The synth-specific preset loader lives in
:mod:`dataset.dexed_preset_loader`.
"""
from .preset_sources import (
    PresetRecord,
    PresetSource,
    SyntheticPresetSource,
    HumanPresetSource,
    HybridPresetSource,
)
from .builder import DatasetBuilder, RenderSettings

__all__ = [
    "PresetRecord",
    "PresetSource",
    "SyntheticPresetSource",
    "HumanPresetSource",
    "HybridPresetSource",
    "DatasetBuilder",
    "RenderSettings",
]
