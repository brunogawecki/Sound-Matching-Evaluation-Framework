"""Test-only tiny deep model that drives the training harness end-to-end.

Not collected as a test (no ``test_`` prefix). ``TinyNetwork`` featurizes inside
``forward`` (waveform -> pooled bins -> linear -> ML-side vector) so the harness can
be exercised on CPU with no GPU and no VST, standing in for a real deep family.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.training.checkpoint import network_state_dict_from_lightning_checkpoint
from models.training.config import TrainingConfig
from models.training.data_module import CorpusDataModule
from models.training.lightning_module import LightningRegressor
from models.training.loss import ParameterLoss
from models.training.trainer_factory import build_trainer


class TinyNetwork(nn.Module):
    """Raw audio ``[batch, num_samples]`` -> pooled features -> ML-side vector."""

    def __init__(self, ml_dimension: int, num_features: int = 8) -> None:
        super().__init__()
        self.num_features = num_features
        self.linear = nn.Linear(num_features, ml_dimension)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Featurization lives in the network: pool |waveform| into num_features bins.
        pooled = F.adaptive_avg_pool1d(audio.abs().unsqueeze(1), self.num_features)
        return self.linear(pooled.squeeze(1))


class TinyDeepModel(BaseDeepModel):
    """A minimal :class:`BaseDeepModel` family wiring the harness components."""

    def __init__(self, num_features: int = 8, default_root_dir: str = "lightning_logs") -> None:
        super().__init__()
        self._num_features = num_features
        self._default_root_dir = default_root_dir

    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        return TinyNetwork(
            ml_dimension=architecture_hparams["ml_dimension"],
            num_features=architecture_hparams["num_features"],
        )

    def fit(
        self,
        train_dataset: RenderedCorpusDataset,
        validation_dataset: Optional[RenderedCorpusDataset] = None,
        config: Optional[Dict[str, object]] = None,
    ) -> None:
        training_config = TrainingConfig.from_dict(config)
        pl.seed_everything(training_config.seed, workers=True)

        parameter_space = train_dataset.parameter_space
        architecture_hparams = {
            "ml_dimension": parameter_space.ml_dimension,
            "num_features": self._num_features,
        }
        network = self._build_network(architecture_hparams)
        parameter_loss = ParameterLoss(parameter_space, training_config.loss)
        lightning_regressor = LightningRegressor(network, parameter_loss, training_config.optimizer)
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
        trainer.fit(lightning_regressor, datamodule=data_module)

        # Export step: load the best .ckpt's network weights, then register the network.
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            network.load_state_dict(network_state_dict_from_lightning_checkpoint(best_path))
        self._set_trained_network(network, architecture_hparams, parameter_space)
