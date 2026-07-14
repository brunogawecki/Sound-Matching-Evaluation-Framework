"""``pl.Trainer`` factory wiring callbacks, logger, and SLURM survival.

Builds a Trainer from a :class:`~models.training.config.TrainingConfig`: a
``ModelCheckpoint`` (best + last) and ``LearningRateMonitor``, optional
``EarlyStopping``, a ``CSVLogger`` (plus an opt-in ``WandbLogger``),
precision/scale straight from config, and a ``SLURMEnvironment(auto_requeue=True)``
plugin when running under SLURM.
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
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment

from models.training.config import TrainingConfig


def build_trainer(
    training_config: TrainingConfig,
    default_root_dir: Union[str, Path] = "lightning_logs",
    monitor: str = "val_loss",
    run_validation: bool = True,
) -> pl.Trainer:
    """Build a configured :class:`~lightning.pytorch.Trainer`.

    Args:
        training_config: the resolved config (its ``trainer`` sub-config drives
            precision/scale/duration; ``seed`` is applied by the caller via
            ``pl.seed_everything`` before ``fit``).
        default_root_dir: where checkpoints and CSV logs are written.
        monitor: the logged metric ``ModelCheckpoint``/``EarlyStopping`` track.
        run_validation: whether a validation loop runs. Pass both ``monitor`` and
            this from the DataModule's ``will_validate``: ``True`` -> ``"val_loss"``,
            ``False`` -> ``"train_loss"`` and the validation loop is disabled.
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

    # CSVLogger always runs; WandbLogger is opt-in via logger.wandb (config.py).
    # name="" stops CSVLogger nesting a second `lightning_logs/` under save_dir.
    loggers: List[Logger] = [CSVLogger(save_dir=str(default_root_dir), name="")]
    logger_config = training_config.logger
    if logger_config.wandb:
        loggers.append(
            WandbLogger(
                project=logger_config.project,
                entity=logger_config.entity,
                name=logger_config.run_name,
                save_dir=str(default_root_dir),
            )
        )

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
        logger=loggers,
        plugins=plugins or None,
        # No validation loop when the DataModule has no validation source.
        limit_val_batches=1.0 if run_validation else 0,
    )
