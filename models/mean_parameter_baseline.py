"""The trivial mean-parameter baseline (the naive floor every family must beat).

Ignores the input audio entirely and always predicts the training set's mean
parameter vector. Built first (issue #7) to exercise the full
train -> predict -> re-render -> metric path before any real model exists, and to
de-risk the :class:`~models.base_model.BaseModel` contract.

The mean is taken over the **ML-side** target vectors: for a continuous parameter
this is the per-parameter mean; for a categorical one-hot block the mean is the
class frequency distribution, whose argmax (taken by
:meth:`ParameterSpace.ml_vector_to_synth_dict`) is the majority class. So a single
average-then-decode handles "mean for continuous, mode for categoricals" with no
special-casing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch

from models.base_model import BaseModel
from synth.parameter_space import ParameterSpace

if TYPE_CHECKING:
    from dataset.torch_dataset import RenderedCorpusDataset


class MeanParameterBaseline(BaseModel):
    """Predicts the training set's mean parameter vector, ignoring the audio."""

    def __init__(self) -> None:
        self._mean_vector: Optional[np.ndarray] = None
        self._parameter_space: Optional[ParameterSpace] = None

    def fit(
        self,
        train_dataset: "RenderedCorpusDataset",
        validation_dataset: Optional["RenderedCorpusDataset"] = None,
        config: Optional[Dict[str, object]] = None,
    ) -> None:
        """Average the training corpus's ML-side target matrix.

        Raises:
            ValueError: if the corpus is empty (no rows to average).
        """
        targets = train_dataset.targets.numpy()
        if targets.shape[0] == 0:
            raise ValueError("Cannot fit MeanParameterBaseline on an empty corpus.")
        self._parameter_space = train_dataset.parameter_space
        self._mean_vector = targets.mean(axis=0)

    def predict(self, audio: torch.Tensor) -> Dict[str, float]:
        """Return the fixed mean prediction as a synth-side dict (audio ignored)."""
        if self._mean_vector is None or self._parameter_space is None:
            raise RuntimeError("MeanParameterBaseline must be fit (or loaded) before predict.")
        return self._parameter_space.ml_vector_to_synth_dict(self._mean_vector)

    def save(self, path: Path) -> None:
        """Write the mean vector and its ParameterSpace so :meth:`load` is standalone."""
        if self._mean_vector is None or self._parameter_space is None:
            raise RuntimeError("MeanParameterBaseline must be fit before save.")
        path = Path(path)
        payload = {
            "mean_vector": self._mean_vector.tolist(),
            "parameter_space": self._parameter_space.to_dict(),
        }
        path.write_text(json.dumps(payload))

    def load(self, path: Path) -> None:
        """Restore the mean vector and ParameterSpace saved by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        self._mean_vector = np.asarray(payload["mean_vector"], dtype=np.float64)
        self._parameter_space = ParameterSpace.from_dict(payload["parameter_space"])
