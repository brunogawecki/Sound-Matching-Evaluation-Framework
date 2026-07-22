"""Shared base for deep model families: the Lightning ``fit`` skeleton plus
save/load/predict over an injected network.

A subclass provides three hooks: ``_build_architecture_hparams`` (constructor knobs +
corpus-derived values), ``_build_network`` (so ``fit`` and ``load`` can rebuild the
network from those hparams), and ``_build_lightning_module`` (the family's training
recipe). The template ``fit`` here does the rest -- seeding, data module, trainer,
best-checkpoint reload, :meth:`_set_trained_network` -- and so do the checkpoint
format and ``predict``'s decode path. Lightning is imported lazily inside ``fit``
(D-FRAMEWORK), so the eval path depends only on ``torch`` and the pure
:mod:`models.training.checkpoint` / ``synth.parameter_space``.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch import nn

from models.base_model import BaseModel
from models.training.checkpoint import (
    export_checkpoint,
    load_checkpoint,
    network_state_dict_from_lightning_checkpoint,
)
from models.training.config import TrainingConfig
from models.training.loss import ParameterLoss
from synth.parameter_space import ParameterSpace

if TYPE_CHECKING:
    import lightning.pytorch as pl

    from dataset.torch_dataset import RenderedCorpusDataset


class BaseDeepModel(BaseModel):
    """``BaseModel`` with shared ``save``/``load``/``predict`` for deep families."""

    def __init__(self, default_root_dir: str = "lightning_logs") -> None:
        self._default_root_dir = default_root_dir
        self._network: Optional[nn.Module] = None
        self._architecture_hparams: Optional[Dict[str, Any]] = None
        self._parameter_space: Optional[ParameterSpace] = None

    # -- per-family hooks ----------------------------------------------------
    @abc.abstractmethod
    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        """Construct (untrained) the family's network from its hparams.

        Called by :meth:`fit` and :meth:`load` to build the network's structure
        (before training / before loading the saved ``state_dict``). Must be
        deterministic in ``architecture_hparams`` and depend only on ``torch``
        (no VST, no Lightning).
        """

    @abc.abstractmethod
    def _build_architecture_hparams(
        self, train_dataset: "RenderedCorpusDataset", parameter_space: ParameterSpace
    ) -> Dict[str, Any]:
        """The hparams dict :meth:`_build_network` consumes.

        Called once at the start of :meth:`fit`: constructor knobs plus anything
        derived from the corpus (``ml_dimension``, render length, ...), so ``load``
        can rebuild the exact network offline from the checkpoint alone.
        """

    @abc.abstractmethod
    def _build_lightning_module(
        self,
        network: nn.Module,
        parameter_loss: ParameterLoss,
        training_config: TrainingConfig,
    ) -> "pl.LightningModule":
        """The family's training-time LightningModule wrapping ``network``.

        Import the module class lazily inside the method body so the eval path
        stays free of training dependencies (D-FRAMEWORK).
        """

    def fit(
        self,
        train_dataset: "RenderedCorpusDataset",
        validation_dataset: Optional["RenderedCorpusDataset"] = None,
        config: Optional[Dict[str, object]] = None,
    ) -> None:
        """The shared training skeleton: hooks -> trainer -> best checkpoint -> register."""
        # Lazy: the training-only Lightning stack (D-FRAMEWORK) stays off the eval path.
        import lightning.pytorch as pl

        from models.training.data_module import CorpusDataModule
        from models.training.trainer_factory import build_trainer

        training_config = TrainingConfig.from_dict(config)
        pl.seed_everything(training_config.seed, workers=True)

        parameter_space = train_dataset.parameter_space
        architecture_hparams = self._build_architecture_hparams(train_dataset, parameter_space)
        network = self._build_network(architecture_hparams)
        self._warm_start_network(network, architecture_hparams)
        parameter_loss = ParameterLoss(parameter_space, training_config.loss)
        lightning_module = self._build_lightning_module(network, parameter_loss, training_config)
        data_module = CorpusDataModule(
            train_dataset, validation_dataset, training_config.data, seed=training_config.seed
        )

        will_validate = data_module.will_validate
        monitor = "val_loss" if will_validate else "train_loss"
        trainer = build_trainer(
            training_config,
            default_root_dir=self._default_root_dir,
            monitor=monitor,
            run_validation=will_validate,
        )

        run_metadata = {
            "architecture": type(self).__name__,
            "dataset": train_dataset.corpus_dir.name,
        }
        for logger in trainer.loggers:
            logger.log_hyperparams({**run_metadata, **training_config.to_dict()})
        trainer.fit(lightning_module, datamodule=data_module)

        # Load the best checkpoint's weights, then register the trained network.
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            network.load_state_dict(network_state_dict_from_lightning_checkpoint(best_path))
        self._set_trained_network(network, architecture_hparams, parameter_space)

    def _warm_start_network(
        self, network: nn.Module, architecture_hparams: Dict[str, Any]
    ) -> None:
        """Optionally initialize the freshly-built network before training (default: no-op).

        Called by :meth:`fit` between building the untrained network and the trainer run.
        Families that train in stages override this to load a compatible ``state_dict`` --
        e.g. SynthRL-i warm-starting from a SynthRL-p checkpoint.
        """

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

    # -- optional per-family checkpoint payload ------------------------------
    def _extra_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        """Family-specific data to persist alongside the network (default: none).

        Override to stash extra tensors/plain-data a family needs at ``load`` time
        beyond the network weights -- e.g. ``IS2``'s cached ITF training pool.
        """
        return None

    def _restore_extra_checkpoint_state(self, extra_state: Optional[Dict[str, Any]]) -> None:
        """Restore whatever :meth:`_extra_checkpoint_state` wrote (default: no-op)."""

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
            extra_state=self._extra_checkpoint_state(),
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
        self._restore_extra_checkpoint_state(payload.get("extra_state"))

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
        audio = audio.to(next(self._network.parameters()).device)
        with torch.no_grad():
            prediction = self._network(audio.unsqueeze(0))
        vector = prediction.squeeze(0).cpu().numpy()
        return self._parameter_space.ml_vector_to_synth_dict(vector)
