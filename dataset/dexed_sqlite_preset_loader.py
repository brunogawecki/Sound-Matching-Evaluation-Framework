"""Dexed preset loader for the preset-gen-vae SQLite collection.

The second synth-specific half of the human-preset pipeline (the sibling of
:mod:`dataset.dexed_preset_loader`, which reads ``.syx`` cartridges). Here the
source is the ~30k-voice preset-gen-vae / Le Vaillant DX7 database
(``paper_repos/preset-gen-vae/synth/dexed_presets.sqlite``), which stores each
voice as a pickled 155-float vector normalized to ``[0, 1]``.

The database ships two tables:

* ``param`` -- ``index_param`` -> ``name``: the parameter names, in vector order.
* ``preset`` -- one row per voice, with ``pickled_params_np_array`` holding the
  155-float vector (``np.save``'d into a BLOB) plus ``name`` / ``labels``.

Crucially, the ``param`` names are Dexed's own plugin-reported names -- the same
names this framework addresses parameters by (D-NAMING). So the adapter maps a
voice onto :class:`LoadedPreset` params **by name** (never by index): it zips the
``param`` names with the vector, then asserts every estimated-subset name is
present. Operator ordering (this DB is OP1-first, ``.syx`` is OP6-first) is
therefore irrelevant -- the mapping is name-driven, not positional.

Loaded voices are deduplicated on their subset projection and split into a
seeded, voice-disjoint train/test partition, reusing the shared helpers in
:mod:`dataset.dexed_preset_loader`.
"""
from __future__ import annotations

import io
import os
import sqlite3
from typing import Dict, List, Optional

import numpy as np

from synth.parameter_space import ParameterSpace
from .dexed_preset_loader import (
    LoadedPreset,
    PresetSplit,
    deduplicate_presets,
    split_presets,
)

_EXPECTED_VECTOR_LENGTH = 155


def _unpickle_vector(blob: bytes) -> np.ndarray:
    """Decode one ``pickled_params_np_array`` BLOB back into its 1D float vector."""
    buffer = io.BytesIO(blob)
    buffer.seek(0)
    return np.load(buffer)


class DexedSqlitePresetLoader:
    """Load, deduplicate and split the preset-gen-vae SQLite voices into human presets.

    Args:
        parameter_space: the estimated subset; presets are projected onto it for
            deduplication (so dedup sees what actually gets rendered).
        test_fraction: share of surviving voices held out for the test set
            (default 0.0 -- all presets go to train; raise to hold out a
            seeded, disjoint test set).
        split_seed: seed for the voice-level train/test shuffle.
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
        self, db_path: str, limit: Optional[int] = None, show_progress: bool = False
    ) -> PresetSplit:
        """Load voices from the SQLite database, deduplicate, and split into train/test.

        Args:
            db_path: path to ``dexed_presets.sqlite``.
            limit: cap on the number of raw voices read (ordered by ``index_preset``);
                ``None`` loads all of them. Deduplication and the split then run over
                the capped set, so a capped run stays fast and self-consistent.
            show_progress: draw a tqdm bar for the (slow, O(n^2)) deduplication scan.
        """
        presets = self._load_presets_from_db(db_path, limit)
        kept = deduplicate_presets(
            presets, self._parameter_space, self._dedup_threshold, show_progress=show_progress
        )
        return split_presets(kept, self._test_fraction, self._split_seed)

    # -- loading -------------------------------------------------------------
    def _load_presets_from_db(self, db_path: str, limit: Optional[int]) -> List[LoadedPreset]:
        connection = sqlite3.connect(db_path)
        try:
            param_names = self._read_param_names(connection)
            self._check_subset_coverage(param_names)
            source_file = os.path.basename(db_path)
            presets: List[LoadedPreset] = []
            for index_preset, name, blob in self._read_preset_rows(connection, limit):
                vector = _unpickle_vector(blob)
                presets.append(
                    LoadedPreset(
                        params=self._voice_params(param_names, vector),
                        source_file=source_file,
                        voice_index=int(index_preset),
                        voice_name=(name or "").rstrip(),
                    )
                )
            return presets
        finally:
            connection.close()

    @staticmethod
    def _read_param_names(connection: sqlite3.Connection) -> List[str]:
        rows = connection.execute("SELECT name FROM param ORDER BY index_param").fetchall()
        return [row[0] for row in rows]

    @staticmethod
    def _read_preset_rows(connection: sqlite3.Connection, limit: Optional[int]):
        query = "SELECT index_preset, name, pickled_params_np_array FROM preset ORDER BY index_preset"
        if limit is not None:
            query += " LIMIT ?"
            return connection.execute(query, (int(limit),)).fetchall()
        return connection.execute(query).fetchall()

    # -- name-based adapter (D-NAMING) ---------------------------------------
    def _check_subset_coverage(self, param_names: List[str]) -> None:
        """Fail loudly if the database does not name every estimated-subset parameter.

        The database's parameter names are Dexed's plugin-reported names, so the
        mapping is by name; this guard mirrors ``subset.build_parameter_space`` and
        catches any future renaming rather than silently mapping the wrong values.
        """
        available = set(param_names)
        missing = [name for name in self._parameter_space.names if name not in available]
        if missing:
            raise RuntimeError(
                f"Subset parameter names not present in the preset database: {missing}. "
                "The database's parameter naming may have changed."
            )

    @staticmethod
    def _voice_params(param_names: List[str], vector: np.ndarray) -> Dict[str, float]:
        if vector.shape != (len(param_names),):
            raise ValueError(
                f"Preset vector has shape {vector.shape}; expected ({len(param_names)},) "
                "to match the param table."
            )
        return {name: float(value) for name, value in zip(param_names, vector)}
