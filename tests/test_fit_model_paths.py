"""Tests for scripts/fit_model.py's run-scoping helper.

``--run-id`` (the SLURM job id, passed by cluster/train.sbatch) is what keeps two
runs of the same model family from overwriting each other's checkpoint, and what
lets the dashboard pull exactly one job's artifacts.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import config  # noqa: E402
from fit_model import resolve_run_paths  # noqa: E402


def test_run_id_scopes_checkpoint_and_logs_under_the_job():
    checkpoint_path, log_root = resolve_run_paths("PresetGenVAEFlowRegressor", "982169")
    base_dir = Path(config.BASE_DIR)
    assert checkpoint_path == base_dir / "checkpoints" / "982169" / "presetgen_vae_flow.pt"
    assert log_root == base_dir / "lightning_logs" / "982169"


def test_without_a_run_id_the_layout_is_flat():
    checkpoint_path, log_root = resolve_run_paths("PresetGenVAEFlowRegressor", None)
    base_dir = Path(config.BASE_DIR)
    assert checkpoint_path == base_dir / "checkpoints" / "presetgen_vae_flow.pt"
    assert log_root == base_dir / "lightning_logs"


def test_two_jobs_of_one_family_do_not_collide():
    first, _ = resolve_run_paths("PresetGenVAEFlowRegressor", "982169")
    second, _ = resolve_run_paths("PresetGenVAEFlowRegressor", "982170")
    assert first != second
    assert first.name == second.name  # same registry filename, different job dir


def test_checkpoint_filename_still_comes_from_the_registry():
    checkpoint_path, _ = resolve_run_paths("MeanParameterBaseline", "111")
    assert checkpoint_path.name == "mean_parameter_baseline.json"
