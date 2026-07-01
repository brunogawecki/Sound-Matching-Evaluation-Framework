"""``pl.Trainer`` factory wiring callbacks, logger, and SLURM survival (issue #22).

Centralizes the harness's Trainer policy so every family builds the same Trainer
from its :class:`~models.training.config.TrainingConfig`:

- **SLURM survival** -- when running under SLURM, a
  ``SLURMEnvironment(auto_requeue=True)`` plugin so the cluster's 24 h SIGTERM
  checkpoints and requeues (D-FRAMEWORK). Off SLURM (e.g. the local Mac) it is
  omitted so the same factory runs anywhere.
- **Checkpointing** -- ``ModelCheckpoint(monitor=..., save_top_k=1, mode="min",
  save_last=True)``. ``last.ckpt`` is the requeue resume point; ``best`` is what the
  family exports into the clean inference artifact afterwards.
- **Logging** -- ``CSVLogger`` (no-internet-friendly on compute nodes) +
  ``LearningRateMonitor``.
- **Precision / scale** -- ``precision`` / ``accelerator`` / ``devices`` /
  ``strategy`` straight from config (``bf16-mixed`` + ``ddp`` on A100; 32-bit single
  device locally).

Imports Lightning: a training-only module.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment

from models.training.config import TrainingConfig


def build_trainer(
    training_config: TrainingConfig,
    default_root_dir: Union[str, Path] = "lightning_logs",
    monitor: str = "val_loss",
) -> pl.Trainer:
    """Build a configured :class:`~lightning.pytorch.Trainer`.

    Args:
        training_config: the resolved config (its ``trainer`` sub-config drives
            precision/scale/duration; ``seed`` is applied by the caller via
            ``pl.seed_everything`` before ``fit``).
        default_root_dir: where checkpoints and CSV logs are written.
        monitor: the logged metric ``ModelCheckpoint``/``EarlyStopping`` track.
            Defaults to ``"val_loss"``; callers training without validation should
            pass ``"train_loss"``.
    """
    trainer_config = training_config.trainer
    default_root_dir = Path(default_root_dir)

    callbacks: List[Callback] = [
        ModelCheckpoint(
            monitor=monitor,
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch:02d}-{" + monitor + ":.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if trainer_config.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode="min",
                patience=trainer_config.early_stopping_patience,
            )
        )

    plugins: List[object] = []
    if SLURMEnvironment.detect():
        plugins.append(SLURMEnvironment(auto_requeue=True))

    # CSVLogger only, by D-FRAMEWORK (no-internet-friendly on the hgx compute nodes).
    # TODO(later): make the logger config-driven (a TrainerConfig knob, e.g.
    #   "csv" | "tensorboard" | "wandb", or a list). TensorBoardLogger is offline-safe
    #   (add `tensorboard` to requirements-cluster.txt); WandbLogger only once outbound
    #   network from the hgx nodes is confirmed. Lightning accepts a list of loggers, so
    #   this is an additive change isolated to this factory.
    logger = CSVLogger(save_dir=str(default_root_dir))

    return pl.Trainer(
        max_epochs=trainer_config.max_epochs,
        precision=trainer_config.precision,
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        strategy=trainer_config.strategy,
        deterministic=trainer_config.deterministic,
        gradient_clip_val=trainer_config.gradient_clip_val,
        log_every_n_steps=trainer_config.log_every_n_steps,
        default_root_dir=str(default_root_dir),
        callbacks=callbacks,
        logger=logger,
        plugins=plugins or None,
    )
