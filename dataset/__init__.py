"""Dataset construction (Layer 2): preset sources and the rendering builder.

A :class:`~dataset.sources.PresetSource` decides which presets exist (synthetic,
human, or hybrid); the :class:`~dataset.builder.DatasetBuilder` renders them
into a WAV + metadata corpus. The synth-specific preset loader lives in
:mod:`dataset.dexed_preset_loader`.
"""
from .sources import (
    PresetRecord,
    PresetSource,
    SyntheticSampler,
    HumanPresetSource,
    HybridSource,
)
from .builder import (
    DatasetBuilder,
    RenderSettings,
    RenderExecutor,
    SequentialExecutor,
)

__all__ = [
    "PresetRecord",
    "PresetSource",
    "SyntheticSampler",
    "HumanPresetSource",
    "HybridSource",
    "DatasetBuilder",
    "RenderSettings",
    "RenderExecutor",
    "SequentialExecutor",
]
