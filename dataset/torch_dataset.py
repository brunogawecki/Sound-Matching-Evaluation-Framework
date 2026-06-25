"""PyTorch Dataset over a rendered corpus (Layer 2 -> Layer 3 seam).

Reads a corpus built by :class:`dataset.builder.DatasetBuilder` and emits
``(audio, target)`` training pairs:

- ``audio``  : the raw rendered waveform, a ``float32`` tensor of shape
  ``[num_samples]`` (88200 at the D3 contract). It is returned **as rendered**,
  with no feature extraction and no normalization: converting to a
  mel-spectrogram / STFT / hand-crafted features is each model's own job, because
  different model families want different representations (see
  ``docs/PROJECT_CONTEXT.md``). The Dataset stays representation-agnostic.
- ``target`` : the ML-side vector, a ``float32`` tensor of shape
  ``[ml_dimension]`` (continuous params in place, categoricals as one-hot blocks
  per D2), produced by :meth:`ParameterSpace.synth_dict_to_ml_vector`.

The Dataset takes a :class:`ParameterSpace` by injection. :meth:`load`
reconstructs that space from the corpus's own ``run_summary.json`` (written by the
builder), so training/evaluation need **no live synthesizer or VST** -- the corpus
is self-describing, which is what lets training run on an external cluster where
the plugin is unavailable.

This module is intentionally not re-exported from ``dataset/__init__`` so that
importing the corpus-generation path does not require ``torch``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset

from synth.parameter_space import ParameterSpace


class RenderedCorpusDataset(Dataset):
    """A built WAV + metadata corpus as ``(audio, target)`` training pairs.

    Args:
        corpus_dir: the corpus directory (the ``<output_root>/<run_name>/`` written
            by the builder), containing ``metadata.csv`` and ``audio/``.
        parameter_space: the space defining the ML-side target vector. Its
            ``names`` must match the parameter columns in ``metadata.csv``.

    Target-only consumers (e.g. the mean-parameter baseline, #7) use the
    :attr:`targets` matrix and never touch the audio.
    """

    def __init__(
        self,
        corpus_dir: Union[str, Path],
        parameter_space: ParameterSpace,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.parameter_space = parameter_space
        self.metadata = pd.read_csv(self.corpus_dir / "metadata.csv")

        parameter_names = parameter_space.names
        missing = [name for name in parameter_names if name not in self.metadata.columns]
        if missing:
            raise ValueError(
                f"metadata.csv at {self.corpus_dir} is missing parameter columns {missing}; "
                "it does not match the given ParameterSpace."
            )

        # Targets depend only on the (static) synth-side params, so build the full
        # (N, ml_dimension) matrix once here rather than per __getitem__.
        parameter_rows: list = self.metadata.loc[:, parameter_names].to_dict(orient="records")
        if parameter_rows:
            self._targets = np.stack(
                [parameter_space.synth_dict_to_ml_vector(row) for row in parameter_rows]
            ).astype(np.float32)
        else:
            self._targets = np.zeros((0, parameter_space.ml_dimension), dtype=np.float32)

    @classmethod
    def load(cls, corpus_dir: Union[str, Path]) -> "RenderedCorpusDataset":
        """Load a dataset from a corpus directory, space and all (no VST needed).

        Unlike ``__init__`` (which takes a ParameterSpace by injection), this reads
        everything from disk: it reconstructs the space from the corpus's own
        ``run_summary.json`` via :meth:`ParameterSpace.from_dict`.

        Raises:
            ValueError: if the summary predates the serialized space (rebuild the
                corpus with the current DatasetBuilder).
        """
        corpus_dir = Path(corpus_dir)
        with open(corpus_dir / "run_summary.json") as summary_file:
            summary = json.load(summary_file)
        if "parameter_space" not in summary:
            raise ValueError(
                f"{corpus_dir / 'run_summary.json'} has no 'parameter_space'. Rebuild this "
                "corpus with the current DatasetBuilder so it carries its parameter map."
            )
        parameter_space = ParameterSpace.from_dict(summary["parameter_space"])
        return cls(corpus_dir, parameter_space)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target = torch.from_numpy(self._targets[index])
        return self._read_audio(index), target

    # -- audio ---------------------------------------------------------------
    def _read_audio(self, index: int) -> torch.Tensor:
        """Lazily read one sample's WAV as a ``float32`` mono tensor ``[num_samples]``."""
        relative_path = self.metadata.iloc[index]["audio_path"]
        _, audio = wavfile.read(self.corpus_dir / relative_path)
        return torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))

    # -- target-only access (for the mean-parameter baseline, #7) ------------
    @property
    def targets(self) -> torch.Tensor:
        """The full ``(N, ml_dimension)`` target matrix as a ``float32`` tensor."""
        return torch.from_numpy(self._targets)
