"""Test-only tiny deep model that drives the training harness end-to-end.

Not collected as a test (no ``test_`` prefix). ``TinyNetwork`` featurizes inside
``forward`` (waveform -> pooled bins -> linear -> ML-side vector) so the harness can
be exercised on CPU with no GPU and no VST, standing in for a real deep family.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.training.config import TrainingConfig
from models.training.loss import ParameterLoss
from synth.parameter_space import ParameterSpace


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
    """A minimal :class:`BaseDeepModel` family wiring the harness components.

    ``fit`` is the inherited template; only the three ``_build_*`` hooks are supplied.
    """

    def __init__(self, num_features: int = 8, default_root_dir: str = "lightning_logs") -> None:
        super().__init__(default_root_dir=default_root_dir)
        self._num_features = num_features

    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        return TinyNetwork(
            ml_dimension=architecture_hparams["ml_dimension"],
            num_features=architecture_hparams["num_features"],
        )

    def _build_architecture_hparams(
        self, train_dataset: RenderedCorpusDataset, parameter_space: ParameterSpace
    ) -> Dict[str, Any]:
        return {
            "ml_dimension": parameter_space.ml_dimension,
            "num_features": self._num_features,
        }

    def _build_lightning_module(
        self, network: nn.Module, parameter_loss: ParameterLoss, training_config: TrainingConfig
    ):
        from models.training.lightning_module import LightningRegressor

        return LightningRegressor(network, parameter_loss, training_config.optimizer)
