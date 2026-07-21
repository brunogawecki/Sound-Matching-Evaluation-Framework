"""LightningModule that trains the InverSynth II proxy regressor (``IS2xITF``, Stage 2).

Objective: ``parameter_loss + audio_loss_weight * audio_loss`` -- the paper's parameters loss
plus its synthesizer-proxy audio loss (its Eq. 4, ``lambda`` = ``audio_loss_weight``). The
parameter term reuses :class:`ParameterLoss` unchanged; the audio term is the MAE between the
proxy's reconstructed spectrogram and the encoder's mel-dB input. The single optimizer trains the
encoder and the proxy jointly, as the paper does. Extends :class:`LightningRegressor` -- only the
loss step differs; step routing, optimizer, and ``ParameterLoss`` logging are inherited.

Training-only (D-FRAMEWORK): imported lazily by :class:`IS2xITF`, so the eval path needs no
Lightning. The proxy exists solely for this gradient; ``predict`` re-renders with the real Dexed.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from models.inversynth2.network import IS2NetworkOutput
from models.training.config import LossConfig, OptimizerConfig
from models.training.lightning_module import LightningRegressor
from models.training.loss import ParameterLoss


class LightningIS2Regressor(LightningRegressor):
    """Wraps an :class:`IS2Network` + :class:`ParameterLoss` into a trainable module.

    ``network`` exposes ``forward(audio) -> [batch, ml_dimension]`` (encoder only, eval) and
    ``forward_training(audio) -> IS2NetworkOutput`` (prediction + proxy + target spectrogram).
    ``loss_config.audio_loss_weight`` scales the proxy audio loss (the paper's ``lambda``).
    """

    def __init__(
        self,
        network: nn.Module,
        parameter_loss: ParameterLoss,
        optimizer_config: OptimizerConfig,
        loss_config: LossConfig,
    ) -> None:
        super().__init__(network, parameter_loss, optimizer_config)
        self._audio_loss_weight = float(loss_config.audio_loss_weight)

    def _shared_step(self, batch: List[torch.Tensor], stage: str) -> torch.Tensor:
        audio, targets = batch
        output: IS2NetworkOutput = self.network.forward_training(audio)
        parameter_losses = self.parameter_loss(output.prediction, targets)
        audio_loss = F.l1_loss(output.proxy_spectrogram, output.target_spectrogram)
        total = parameter_losses["loss"] + self._audio_loss_weight * audio_loss

        log = dict(on_step=False, on_epoch=True, batch_size=audio.shape[0])
        self.log(f"{stage}_loss", total, prog_bar=True, **log)
        self.log(f"{stage}_parameter_loss", parameter_losses["loss"], prog_bar=True, **log)
        self.log(f"{stage}_audio_loss", audio_loss, prog_bar=True, **log)
        self._log_parameter_losses(output.prediction, targets, parameter_losses, stage, log)
        return total
