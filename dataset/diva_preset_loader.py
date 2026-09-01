"""Diva preset loader for the Flow Synthesizer ``diva_raw`` collection.

The Diva-side half of the human-preset pipeline (the sibling of
:mod:`dataset.dexed_preset_loader` and :mod:`dataset.dexed_sqlite_preset_loader`). The source
is the 11,217-preset corpus published with Esling et al.'s Flow Synthesizer
(``diva_raw.zip``, https://nubo.ircam.fr/index.php/s/nL3NQomqxced6eJ), which stores one preset
per ``raw/<hash>_60_100.npz`` file. Each archive member holds three arrays; only ``param`` is
read here, a pickled ``Dict[str, float]`` giving all 281 of Diva's synthesis parameters
normalized to ``[0, 1]``.

Two format facts make the adapter almost trivial:

* The keys are Diva's own module-qualified parameter names, written ``'VCF1: Model'`` where
  this framework writes ``'VCF1.Model'`` (D-NAMING). Translating the separator maps all 281
  names onto ``synth/diva/parameters.py`` exactly, with nothing left over on either side, so
  the mapping is by name and no translation table is needed.
* The values are already in the plugin's own normalized ``[0, 1]`` scale, and every discrete
  parameter lands exactly on its option grid (verified over the whole corpus).

The ``_60_100`` suffix is the note and velocity the paper rendered at. It is ignored: only the
parameters are taken, and the audio is re-rendered under this framework's own contract (D3).

**The corpus does not vary every parameter.** It was generated from a fixed base patch with
only Flow Synthesizer's 64 continuous parameters set from the presets, so 179 of the 237
parameters in D-DIVA-SUBSET are constant across all 11,217 files. Use
:func:`dataset.preset_loader_common.constant_parameters` to measure this on whatever is
actually loaded before treating the result as a general-purpose Diva human corpus.
"""
from __future__ import annotations

import io
import os
import zipfile
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from synth.parameter_space import ParameterSpace
from .preset_loader_common import (
    LoadedPreset,
    PresetSplit,
    deduplicate_presets,
    split_presets,
)

# The corpus writes 'Module: Name'; this framework addresses parameters as 'Module.Name'.
_CORPUS_SEPARATOR = ": "
_FRAMEWORK_SEPARATOR = "."

def _to_framework_names(params: Dict[str, float]) -> Dict[str, float]:
    """Rewrite one preset's keys from the corpus' separator to this framework's."""
    return {
        name.replace(_CORPUS_SEPARATOR, _FRAMEWORK_SEPARATOR, 1): float(value)
        for name, value in params.items()
    }


def _read_param_dict(data: bytes) -> Dict[str, float]:
    """Unpack the ``param`` member of one ``.npz`` preset, ignoring ``audio`` and ``chars``."""
    with np.load(io.BytesIO(data), allow_pickle=True) as archive:
        params = archive["param"].item()
    if not isinstance(params, dict):
        raise ValueError(f"Expected 'param' to hold a dict, got {type(params).__name__}.")
    return _to_framework_names(params)


class DivaPresetLoader:
    """Load, deduplicate and split the Flow Synthesizer Diva presets.

    Args:
        parameter_space: the estimated subset; presets are projected onto it for
            deduplication (so dedup sees what actually gets rendered).
        test_fraction: share of surviving presets held out for the test set
            (default 0.0 -- all presets go to train).
        split_seed: seed for the preset-level train/test shuffle.
        dedup_threshold: max-norm distance between projected ML vectors below
            which two presets are duplicates.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        test_fraction: float = 0.0,
        split_seed: int = 0,
        dedup_threshold: float = 1e-3,
    ):
        if not 0.0 <= test_fraction <= 1.0:
            raise ValueError(f"test_fraction must be in [0, 1], got {test_fraction}.")
        self._parameter_space = parameter_space
        self._test_fraction = float(test_fraction)
        self._split_seed = int(split_seed)
        self._dedup_threshold = float(dedup_threshold)

    def load(
        self, source_path: str, limit: Optional[int] = None, show_progress: bool = False
    ) -> PresetSplit:
        """Load presets from the collection, deduplicate, and split into train/test.

        Args:
            source_path: either ``diva_raw.zip`` or a directory holding the extracted
                ``.npz`` presets (searched recursively). Reading the zip directly avoids
                unpacking 1.4 GB that is barely compressed.
            limit: cap on the number of raw presets read, in sorted-name order; ``None``
                loads all of them. Deduplication and the split then run over the capped
                set, so a capped run stays fast and self-consistent.
            show_progress: draw a tqdm bar for the (slow, O(n^2)) deduplication scan.
        """
        presets = self._load_presets(source_path, limit)
        kept = deduplicate_presets(
            presets, self._parameter_space, self._dedup_threshold, show_progress=show_progress
        )
        return split_presets(kept, self._test_fraction, self._split_seed)

    # -- loading -------------------------------------------------------------
    def _load_presets(self, source_path: str, limit: Optional[int]) -> List[LoadedPreset]:
        source_file = os.path.basename(os.path.normpath(source_path))
        presets: List[LoadedPreset] = []
        for voice_index, (member_name, data) in enumerate(_iter_members(source_path, limit)):
            params = _read_param_dict(data)
            self._check_subset_coverage(params, member_name)
            presets.append(
                LoadedPreset(
                    params=params,
                    source_file=source_file,
                    voice_index=voice_index,
                    voice_name=os.path.splitext(os.path.basename(member_name))[0],
                )
            )
        if not presets:
            raise RuntimeError(f"No .npz presets found under {source_path}.")
        return presets

    # -- name-based adapter (D-NAMING) ---------------------------------------
    def _check_subset_coverage(self, params: Dict[str, float], member_name: str) -> None:
        """Fail loudly if a preset does not name every estimated-subset parameter.

        Mirrors ``DexedSqlitePresetLoader._check_subset_coverage``: the corpus' names are
        Diva's own, so the mapping is by name, and a future renaming must raise rather than
        silently drop parameters. Checked per preset, because each ``.npz`` carries its own
        dict rather than sharing one declared parameter table.
        """
        missing = [name for name in self._parameter_space.names if name not in params]
        if missing:
            raise RuntimeError(
                f"Subset parameter names not present in {member_name}: {missing}. "
                "The collection's parameter naming may have changed."
            )


def _iter_members(source_path: str, limit: Optional[int]) -> Iterator[Tuple[str, bytes]]:
    """Yield ``(member_name, raw_bytes)`` for each ``.npz`` preset, in sorted-name order."""
    if zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path) as archive:
            names = _limited(sorted(n for n in archive.namelist() if n.endswith(".npz")), limit)
            for name in names:
                yield name, archive.read(name)
        return
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"{source_path} is neither a zip archive nor a directory.")
    paths: List[str] = []
    for directory, _subdirectories, filenames in os.walk(source_path):
        paths.extend(
            os.path.join(directory, filename)
            for filename in filenames
            if filename.endswith(".npz")
        )
    for path in _limited(sorted(paths), limit):
        with open(path, "rb") as preset_file:
            yield path, preset_file.read()


def _limited(names: Sequence[str], limit: Optional[int]) -> Sequence[str]:
    return names if limit is None else names[: int(limit)]
