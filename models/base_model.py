"""The Layer 3 model contract.

Every sound-matching model family (search-based, discriminative, generative,
neural proxy) implements :class:`BaseModel`. The contract is deliberately small
and framework-agnostic: it says *what* a model does (fit, predict, save, load),
not *how* it trains. Whether a concrete model runs a hand-written PyTorch loop or
a PyTorch-Lightning ``Trainer`` is its own private business -- that choice is
deferred to the first deep family and must never leak into this interface.

``predict`` returns a **synth-side dict** (parameter name -> normalized float in
``[0, 1]``), so its output is directly consumable by
``BaseSynthesizer.set_parameters`` at evaluation time with no glue code. A model
that works internally in the ML-side vector space decodes via
``ParameterSpace.ml_vector_to_synth_dict`` before returning.

Models depend only on ``torch`` / ``numpy`` and the pure
``synth.parameter_space`` -- never on a live VST, the renderers, or dawdreamer.
The training and evaluation path runs on an external GPU cluster where the plugin
is unavailable (D-SELFDESC); a corpus is self-describing and carries its own
``ParameterSpace``.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import torch

if TYPE_CHECKING:
    from dataset.torch_dataset import RenderedCorpusDataset


class BaseModel(abc.ABC):
    """Abstract base class every sound-matching model family implements."""

    @abc.abstractmethod
    def fit(
        self,
        train_dataset: "RenderedCorpusDataset",
        validation_dataset: Optional["RenderedCorpusDataset"] = None,
        config: Optional[Dict[str, object]] = None,
    ) -> None:
        """Fit the model to a training corpus.

        For search-based families (e.g. the genetic algorithm) this may be a
        no-op; for deep models it is the training loop. ``validation_dataset``
        and ``config`` are optional so trivial models can ignore them.
        """

    @abc.abstractmethod
    def predict(self, audio: torch.Tensor) -> Dict[str, float]:
        """Predict synth-side parameters for one target waveform.

        Args:
            audio: a single rendered waveform tensor, shape ``[num_samples]``
                (as emitted by ``RenderedCorpusDataset.__getitem__``).

        Returns:
            A synth-side dict mapping every estimated parameter name to a
            normalized float in ``[0, 1]``, ready for
            ``BaseSynthesizer.set_parameters``.
        """

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model so :meth:`load` can fully restore it."""

    @abc.abstractmethod
    def load(self, path: Path) -> None:
        """Restore a model saved by :meth:`save` (no dataset or VST needed)."""
