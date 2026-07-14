import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.training.config import (
    DataConfig,
    LoggerConfig,
    LossConfig,
    OptimizerConfig,
    TrainerConfig,
    TrainingConfig,
)


# -- defaults ----------------------------------------------------------------

def test_defaults_match_the_decided_values():
    config = TrainingConfig.from_dict({})
    assert config.seed == 0
    assert config.optimizer.name == "adamw"
    assert config.optimizer.learning_rate == pytest.approx(3e-4)
    assert config.loss.continuous_loss == "mse"
    assert config.loss.categorical_loss_weight == pytest.approx(0.2)  # preset-gen-vae
    assert config.trainer.precision == "bf16-mixed"
    assert config.trainer.deterministic is True
    assert config.data.val_fraction is None
    assert config.logger.wandb is False  # tracking is opt-in
    assert config.logger.project == "Sound-Matching-Evaluation-Framework"


def test_from_dict_none_is_all_defaults():
    assert TrainingConfig.from_dict(None).to_dict() == TrainingConfig.from_dict({}).to_dict()


# -- nested parsing ----------------------------------------------------------

def test_nested_keys_are_parsed_into_sub_configs():
    config = TrainingConfig.from_dict(
        {
            "seed": 7,
            "optimizer": {"learning_rate": 1e-3, "weight_decay": 0.01},
            "loss": {"continuous_loss": "mae", "categorical_loss_weight": 0.5},
            "data": {"batch_size": 32, "val_fraction": 0.1},
            "trainer": {"max_epochs": 5, "precision": "32-true", "devices": 1},
            "logger": {"wandb": True, "project": "p", "entity": "e"},
        }
    )
    assert config.seed == 7
    assert isinstance(config.optimizer, OptimizerConfig)
    assert config.optimizer.learning_rate == pytest.approx(1e-3)
    assert config.loss.continuous_loss == "mae"
    assert config.data.batch_size == 32 and config.data.val_fraction == pytest.approx(0.1)
    assert config.trainer.max_epochs == 5 and config.trainer.devices == 1
    assert config.logger.wandb is True and config.logger.project == "p"
    assert config.logger.entity == "e"


# -- round trip --------------------------------------------------------------

def test_to_dict_round_trips_through_from_dict():
    original = TrainingConfig.from_dict(
        {"seed": 3, "optimizer": {"name": "adam"}, "trainer": {"max_epochs": 9}}
    )
    assert TrainingConfig.from_dict(original.to_dict()).to_dict() == original.to_dict()


# -- strictness --------------------------------------------------------------

def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        TrainingConfig.from_dict({"learning_rate": 1e-3})  # belongs under 'optimizer'


def test_unknown_nested_key_is_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        OptimizerConfig.from_dict({"lr": 1e-3})  # wrong name; should be learning_rate


@pytest.mark.parametrize("sub_config", [DataConfig, LossConfig, TrainerConfig, LoggerConfig])
def test_each_sub_config_rejects_typos(sub_config):
    with pytest.raises(ValueError, match="unknown config key"):
        sub_config.from_dict({"definitely_not_a_real_knob": 1})


# -- yaml --------------------------------------------------------------------

def test_from_yaml_parses_a_file(tmp_path):
    pytest.importorskip("yaml")
    config_path = tmp_path / "training_config.yaml"
    config_path.write_text(
        "seed: 11\n"
        "optimizer:\n"
        "  learning_rate: 0.001\n"
        "loss:\n"
        "  categorical_loss_weight: 0.3\n"
    )
    config = TrainingConfig.from_yaml(config_path)
    assert config.seed == 11
    assert config.optimizer.learning_rate == pytest.approx(1e-3)
    assert config.loss.categorical_loss_weight == pytest.approx(0.3)


def test_from_yaml_rejects_non_mapping(tmp_path):
    pytest.importorskip("yaml")
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        TrainingConfig.from_yaml(config_path)


# -- logger wiring in build_trainer ------------------------------------------

def _logger_class_names(trainer):
    return {type(logger).__name__ for logger in trainer.loggers}


def test_build_trainer_attaches_csv_logger_only_by_default(tmp_path):
    pytest.importorskip("lightning")
    from models.training.trainer_factory import build_trainer

    trainer = build_trainer(TrainingConfig.from_dict({}), default_root_dir=tmp_path)
    assert _logger_class_names(trainer) == {"CSVLogger"}


def test_build_trainer_csv_logger_does_not_nest_a_second_lightning_logs(tmp_path):
    """The CSV logger writes straight under default_root_dir (name="").

    Its default name would append another ``lightning_logs/`` inside the root,
    which under a per-job root (``lightning_logs/<job id>/``) reads as
    ``lightning_logs/<job id>/lightning_logs/version_0``.
    """
    pytest.importorskip("lightning")
    from models.training.trainer_factory import build_trainer

    trainer = build_trainer(TrainingConfig.from_dict({}), default_root_dir=tmp_path)
    log_dir = Path(trainer.loggers[0].log_dir)
    assert log_dir.parent == tmp_path
    assert "lightning_logs" not in log_dir.relative_to(tmp_path).parts


def test_build_trainer_adds_wandb_logger_when_enabled(tmp_path, monkeypatch):
    pytest.importorskip("lightning")
    pytest.importorskip("wandb")  # constructing WandbLogger needs the package
    from models.training.trainer_factory import build_trainer

    # Offline so no network / API key is needed; wandb.init is deferred until the
    # experiment is accessed, so building the trainer stays offline regardless.
    monkeypatch.setenv("WANDB_MODE", "offline")
    trainer = build_trainer(
        TrainingConfig.from_dict({"logger": {"wandb": True, "project": "unit-test"}}),
        default_root_dir=tmp_path,
    )
    assert _logger_class_names(trainer) == {"CSVLogger", "WandbLogger"}
