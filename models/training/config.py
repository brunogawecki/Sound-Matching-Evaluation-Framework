"""Typed training configuration.

A nested ``dict`` (or ``training_config.yaml``) parsed into a frozen
:class:`TrainingConfig` of frozen sub-configs, the single source of truth for every
training knob. Unknown keys are rejected at parse time so a typo fails loudly.
``yaml`` is imported lazily so the eval path needs no ``pyyaml``; :meth:`to_dict`
round-trips the resolved config for reproducibility.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


def _reject_unknown_keys(cls: type, data: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``data`` unchanged, or raise if it carries keys ``cls`` does not define."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__} got unknown config key(s) {sorted(unknown)}; "
            f"valid keys are {sorted(known)}."
        )
    return data


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer + (optional) LR-scheduler settings.

    Defaults follow the discriminative-regressor lineage: AdamW at ``3e-4`` with
    no weight decay. ``scheduler`` is ``None`` (constant LR) unless set to a
    supported name (currently ``"cosine"``).
    """
    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    scheduler: Optional[str] = None
    scheduler_max_epochs: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OptimizerConfig":
        if data is None:
            return cls()
        return cls(**_reject_unknown_keys(cls, data))


@dataclass(frozen=True)
class LossConfig:
    """Controls-loss weighting, plus the VAE reconstruction/latent terms.

    ``categorical_loss_weight`` defaults to ``0.2`` (preset-gen-vae's
    ``categorical_loss_factor``). The ``beta*``/``reconstruction_loss``/``normalize_latent_loss``
    fields are read only by VAE families; ``audio_loss_weight`` only by the InverSynth II proxy
    families (``IS2xITF`` / ``IS2``); plain regressors ignore them all.
    """
    continuous_loss: str = "mse"  # "mse" | "mae"
    categorical_loss_weight: float = 0.2
    reconstruction_loss: str = "mse"
    beta: float = 0.2  # KL weight reached after warmup
    beta_start_value: float = 0.1
    beta_warmup_epochs: int = 25
    normalize_latent_loss: bool = True
    audio_loss_weight: float = 1.0  # InverSynth II proxy audio loss weight (paper's lambda)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LossConfig":
        if data is None:
            return cls()
        return cls(**_reject_unknown_keys(cls, data))


@dataclass(frozen=True)
class RLConfig:
    """Reinforcement-learning stage knobs, read only by the RL family (``SynthRLi``).

    Defaults track the SynthRL repo's ``config/stage2.yaml`` + ``utils/buffer.py``:
    per-target PER buffer capacity (``per_capacity`` = 5), one experience sampled per
    target per update (``buffer.sample`` draws 1), the number of gradient-free render
    passes that fill the buffers before training (``finetune.py`` runs ``per_capacity``
    of them; ``0`` skips the pre-fill), the parameter-loss -> RL curriculum ramp length
    in epochs (``rl_coef`` ramps over epochs 199->299, i.e. 100 epochs; ``0`` disables the
    ramp for short runs), the render-worker count for the in-loop reward (``None`` =
    ``os.cpu_count()``; the paper fixes ``synth_render_workers`` = 16), the render engine,
    and the three reward-distance weights (paper Eq. 5 / repo ``model/loss.py``: w1/w2/w3).
    Non-RL families ignore them all.
    """
    buffer_capacity: int = 5
    samples_per_target: int = 1
    prefill_epochs: int = 5
    ramp_epochs: int = 100
    num_render_workers: Optional[int] = None
    renderer: str = "dawdreamer"
    reward_spectrogram_weight: float = 0.27
    reward_spectral_convergence_weight: float = 0.7
    reward_mfcc_weight: float = 0.03

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RLConfig":
        if data is None:
            return cls()
        return cls(**_reject_unknown_keys(cls, data))


@dataclass(frozen=True)
class DataConfig:
    """DataLoader + train/val split settings.

    ``val_fraction`` carves a seeded sample-level validation split from the train
    corpus **only when no explicit validation corpus is given** to ``fit``. The
    held-out human test set is never used for training-time validation.
    """
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    val_fraction: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DataConfig":
        if data is None:
            return cls()
        return cls(**_reject_unknown_keys(cls, data))


@dataclass(frozen=True)
class LoggerConfig:
    """Experiment-tracking settings.

    ``CSVLogger`` is always attached; ``wandb=True`` also attaches a ``WandbLogger``.
    ``entity`` is the wandb namespace (``None`` uses the API key's default); the key
    and online/offline mode come from the environment (``WANDB_API_KEY`` /
    ``WANDB_MODE``). Blank ``run_name`` lets wandb auto-name the run.
    """
    wandb: bool = False
    project: str = "Sound-Matching-Evaluation-Framework"
    entity: Optional[str] = None
    run_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LoggerConfig":
        if data is None:
            return cls()
        return cls(**_reject_unknown_keys(cls, data))


@dataclass(frozen=True)
class TrainerConfig:
    """``pl.Trainer`` + callback settings (precision / scale / SLURM survival).

    Cluster defaults to ``bf16-mixed`` on A100 with ``ddp``; locally one device at
    32-bit is the sensible fallback (set ``precision="32-true"``, ``strategy="auto"``,
    ``devices=1``). ``deterministic`` is on by default for reproducibility.
    """
    max_epochs: int = 100
    precision: str = "bf16-mixed"
    accelerator: str = "auto"
    devices: Union[int, str] = "auto"
    strategy: str = "auto"
    deterministic: bool = True
    gradient_clip_val: Optional[float] = None
    early_stopping_patience: Optional[int] = None
    log_every_n_steps: int = 50

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrainerConfig":
        if data is None:
            return cls()
        return cls(**_reject_unknown_keys(cls, data))


@dataclass(frozen=True)
class TrainingConfig:
    """The complete, resolved training configuration.

    Built from a plain ``dict`` (``from_dict``) or a YAML file (``from_yaml``).
    Round-trips through :meth:`to_dict` so the exact settings of a run can be
    echoed next to its checkpoint.
    """
    seed: int = 0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrainingConfig":
        """Build a config from a nested ``dict`` (the shape ``fit`` receives).

        Top-level keys: ``seed``, ``optimizer``, ``loss``, ``data``, ``trainer``,
        ``logger``, ``rl``. Unknown keys at any level raise ``ValueError``.
        """
        data = dict(data or {})
        _reject_unknown_keys(cls, data)
        return cls(
            seed=int(data.get("seed", 0)),
            optimizer=OptimizerConfig.from_dict(data.get("optimizer")),
            loss=LossConfig.from_dict(data.get("loss")),
            data=DataConfig.from_dict(data.get("data")),
            trainer=TrainerConfig.from_dict(data.get("trainer")),
            logger=LoggerConfig.from_dict(data.get("logger")),
            rl=RLConfig.from_dict(data.get("rl")),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load and parse a ``training_config.yaml`` (lazy ``yaml`` import)."""
        import yaml  # lazy: only the cluster/training path needs pyyaml

        with open(path) as config_file:
            data = yaml.safe_load(config_file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a YAML mapping, got {type(data).__name__}.")
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """The fully-resolved config as a nested JSON/YAML-safe ``dict``."""
        return dataclasses.asdict(self)
