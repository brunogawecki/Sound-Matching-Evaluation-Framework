"""Shared base for deep model families (issue #22).

:class:`BaseDeepModel` implements the parts of :class:`~models.base_model.BaseModel`
that are provably shared by every deep family **because of the ParameterSpace
contract** -- ``save``, ``load`` and ``predict`` -- against an injected network
(a plain ``nn.Module``). What is *not* shared, ``fit`` (each family trains
differently) and the network's architecture, stay abstract.

Crucially this base imports **no Lightning** and no VST: it depends only on
``torch`` and the pure :mod:`models.training.checkpoint` / ``synth.parameter_space``.
A family's ``fit`` uses the Lightning harness under ``models/training/``; the Mac
eval path only ever calls ``load``/``predict`` here, so Lightning never reaches the
Mac (D-FRAMEWORK / D-SELFDESC / D-EVAL).

The split between this and a family subclass:

- the subclass provides ``_build_network(architecture_hparams) -> nn.Module``
  (its architecture) so ``load`` can rebuild the network before loading weights, and a
  ``fit`` that trains a network and registers it via :meth:`_set_trained_network`;
- everything else (the checkpoint format, ``predict``'s decode path) lives here.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from models.base_model import BaseModel
from models.training.checkpoint import export_checkpoint, load_checkpoint
from synth.parameter_space import ParameterSpace


class BaseDeepModel(BaseModel):
    """``BaseModel`` with shared ``save``/``load``/``predict`` for deep families."""

    def __init__(self) -> None:
        self._network: Optional[nn.Module] = None
        self._architecture_hparams: Optional[Dict[str, Any]] = None
        self._parameter_space: Optional[ParameterSpace] = None

    # -- per-family hooks ----------------------------------------------------
    @abc.abstractmethod
    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        """Construct (untrained) the family's network from its hparams.

        Called by :meth:`load` to rebuild the network's structure before loading the
        saved ``state_dict``. Must be deterministic in ``architecture_hparams`` and
        depend only on ``torch`` (no VST, no Lightning).
        """

    # ``fit`` stays abstract (inherited from BaseModel) -- each family trains
    # differently. A family's fit trains a network then calls _set_trained_network.

    def _set_trained_network(
        self,
        network: nn.Module,
        architecture_hparams: Dict[str, Any],
        parameter_space: ParameterSpace,
    ) -> None:
        """Register a freshly-trained network so ``save``/``predict`` can use it.

        A family's ``fit`` calls this after ``trainer.fit`` with the trained network,
        the hparams that built it, and the corpus's ParameterSpace.
        """
        self._network = network
        self._architecture_hparams = dict(architecture_hparams)
        self._parameter_space = parameter_space

    # -- BaseModel contract --------------------------------------------------
    def save(self, path: Path) -> None:
        """Export the trained network + hparams + ParameterSpace (a torch artifact)."""
        if (
            self._network is None
            or self._architecture_hparams is None
            or self._parameter_space is None
        ):
            raise RuntimeError("Model must be fit (or loaded) before save.")
        export_checkpoint(
            self._network,
            self._architecture_hparams,
            self._parameter_space,
            Path(path),
        )

    def load(self, path: Path) -> None:
        """Restore a model from a checkpoint written by :meth:`save` (no VST/Lightning)."""
        payload = load_checkpoint(Path(path))
        architecture_hparams = payload["architecture_hparams"]
        network = self._build_network(architecture_hparams)
        network.load_state_dict(payload["state_dict"])
        network.eval()
        self._network = network
        self._architecture_hparams = dict(architecture_hparams)
        self._parameter_space = ParameterSpace.from_dict(payload["parameter_space"])

    def predict(self, audio: torch.Tensor) -> Dict[str, float]:
        """Predict a synth-side dict for one waveform ``[num_samples]``.

        Runs the network end-to-end (featurization lives inside the network's
        ``forward``), then decodes the ML-side output to a synth-side dict --
        ``ml_vector_to_synth_dict`` argmax-decodes categorical blocks and clips
        continuous values into bounds, so the result is always a valid wrapper input.
        """
        if self._network is None or self._parameter_space is None:
            raise RuntimeError("Model must be fit (or loaded) before predict.")
        self._network.eval()
        with torch.no_grad():
            prediction = self._network(audio.unsqueeze(0))
        vector = prediction.squeeze(0).cpu().numpy()
        return self._parameter_space.ml_vector_to_synth_dict(vector)
