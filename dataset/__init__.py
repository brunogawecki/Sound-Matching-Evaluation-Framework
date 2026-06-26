"""Dataset construction (Layer 2): preset sources and the rendering builder.

A :class:`~dataset.preset_sources.PresetSource` decides which presets exist (synthetic,
human, or hybrid); the :class:`~dataset.builder.DatasetBuilder` renders them
into a WAV + metadata corpus. The synth-specific preset loader lives in
:mod:`dataset.dexed_preset_loader`.

These names are exposed lazily (PEP 562): importing them pulls in the
synth/render stack (and, when rendering, dawdreamer), so the corpus-*consumption*
path (:mod:`dataset.torch_dataset`, used for training on a machine without the VST)
can import without dragging the generation stack along.
"""
from typing import TYPE_CHECKING

__all__ = [
    "PresetRecord",
    "PresetSource",
    "SyntheticPresetSource",
    "HumanPresetSource",
    "HybridPresetSource",
    "DatasetBuilder",
    "RenderSettings",
]

_PRESET_SOURCE_NAMES = frozenset({
    "PresetRecord",
    "PresetSource",
    "SyntheticPresetSource",
    "HumanPresetSource",
    "HybridPresetSource",
})
_BUILDER_NAMES = frozenset({"DatasetBuilder", "RenderSettings"})

if TYPE_CHECKING:
    from .preset_sources import (
        PresetRecord,
        PresetSource,
        SyntheticPresetSource,
        HumanPresetSource,
        HybridPresetSource,
    )
    from .builder import DatasetBuilder, RenderSettings


def __getattr__(name: str):
    if name in _PRESET_SOURCE_NAMES:
        from . import preset_sources
        return getattr(preset_sources, name)
    if name in _BUILDER_NAMES:
        from . import builder
        return getattr(builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
