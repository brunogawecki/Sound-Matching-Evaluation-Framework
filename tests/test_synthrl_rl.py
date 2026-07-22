"""Tests for the SynthRL-i in-domain RL stage (Step 6).

Covers the plan's L1/L2 checks without a VST: a finite REINFORCE single-step gradient,
reward climbing when overfitting a tiny corpus, the warm-start-from-SynthRL-p hook, and
registry + end-to-end evaluability. The RL loop renders inside training, so a picklable-
free fake backend renders the same ``AMP*sin`` waveform the corpus's ``FakeSynth`` does,
giving the policy a real (tiny) reward signal. Skips when torch/lightning/librosa absent.
"""
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("librosa")

import evaluation.evaluator as evaluator_module
from dataset.builder import DatasetBuilder, RenderSettings
from dataset.preset_sources import SyntheticPresetSource
from dataset.torch_dataset import RenderedCorpusDataset
from evaluation.evaluator import Evaluator
from models.registry import MODEL_REGISTRY
from models.synthrl import SynthRLi, SynthRLp
from models.training.checkpoint import load_checkpoint
from synth.parameter_space import ParameterSpace, ParameterSpecification

SAMPLE_RATE = 16000
DURATION_SEC = 1.0
EXPECTED_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

TINY_KWARGS = dict(
    num_bins=8, n_fft=512, hop_length=128, win_length=512, n_mels=64, mel_fmax=8000.0,
    d_model=32, num_conv_layers=3, num_encoder_layers=2, num_decoder_layers=2,
    num_heads=4, feedforward_dim=64, dropout=0.0,
)


def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.8),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 0.5, 1.0], default=0.0),
    ])


def _sine(amp: float) -> np.ndarray:
    time = np.arange(EXPECTED_SAMPLES) / SAMPLE_RATE
    return (amp * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)


class FakeSynth:
    renderer_name = "fake"

    def __init__(self, space: ParameterSpace, sample_rate: int = SAMPLE_RATE):
        self._space = space
        self._sample_rate = sample_rate
        self._state: Dict[str, float] = {s.name: s.default for s in space.parameter_specs}

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def parameter_space(self) -> ParameterSpace:
        return self._space

    def get_parameter_defaults(self) -> Dict[str, float]:
        return {s.name: s.default for s in self._space.parameter_specs}

    def set_parameters(self, params: Dict[str, float]) -> None:
        self._state.update(params)

    def get_parameters(self) -> Dict[str, float]:
        return dict(self._state)

    def render_audio(self, midi_note, velocity, duration_sec, note_duration_sec=None) -> np.ndarray:
        return _sine(float(self._state["AMP"]))


class _FakeRenderBackend:
    """Stands in for the parallel fresh-process backend: renders ``AMP*sin`` in-process."""

    def render(self, params: Dict[str, float]) -> np.ndarray:
        return _sine(float(params["AMP"]))

    def render_batch(self, patches: List[Dict[str, float]]) -> List[np.ndarray]:
        return [self.render(patch) for patch in patches]

    def close(self) -> None:
        pass


class _FakeEvalBackend:
    """Stands in for FreshProcessRenderBackend at eval time (Evaluator re-render)."""

    def __init__(self, settings, renderer="dawdreamer"):
        pass

    def render(self, params):
        return _sine(float(params["AMP"]))

    def close(self):
        pass


def build_corpus(tmp_path, run_name, count, seed) -> RenderedCorpusDataset:
    synth = FakeSynth(make_space())
    settings = RenderSettings(
        midi_note=60, velocity=100, duration_sec=DURATION_SEC, note_duration_sec=DURATION_SEC
    )
    source = SyntheticPresetSource(
        make_space(), count=count, seed=seed, sampling_ranges={"AMP": (0.7, 1.0)}
    )
    DatasetBuilder(synth, render_settings=settings).build(source, run_name=run_name, output_root=tmp_path)
    return RenderedCorpusDataset.load(tmp_path / run_name)


def rl_config(max_epochs=12, seed=0):
    return {
        "seed": seed,
        "optimizer": {"learning_rate": 5e-3},
        "data": {"batch_size": 8, "val_fraction": 0.25},
        "trainer": {
            "max_epochs": max_epochs, "precision": "32-true", "accelerator": "cpu",
            "devices": 1, "log_every_n_steps": 1,
        },
        "rl": {"buffer_capacity": 6, "samples_per_target": 4, "ramp_epochs": 0},
    }


def make_synthrli(**overrides):
    kwargs = dict(backend_factory=_FakeRenderBackend, **TINY_KWARGS)
    kwargs.update(overrides)
    return SynthRLi(**kwargs)


def logged_metric(log_dir, name) -> list:
    metrics_files = list(log_dir.rglob("metrics.csv"))
    assert metrics_files, "CSVLogger should have written a metrics.csv"
    return pd.read_csv(metrics_files[0])[name].dropna().tolist()


def test_registry_entry_constructs_synthrli():
    registration = MODEL_REGISTRY["SynthRLi"]
    assert registration.model_class is SynthRLi
    assert registration.default_checkpoint_filename.endswith(".pt")


def test_reinforce_single_step_gradient_is_finite(tmp_path):
    from models.synthrl.lightning_module import SynthRLReinforceRegressor

    train_dataset = build_corpus(tmp_path, "train", count=8, seed=0)
    space = train_dataset.parameter_space
    model = make_synthrli(default_root_dir=str(tmp_path / "logs"))
    hparams = model._build_architecture_hparams(train_dataset, space)
    network = model._build_network(hparams)

    module = SynthRLReinforceRegressor(
        network, model._training_representation, training_config_optimizer(),
        render_settings=model._training_render_settings, sample_rate=SAMPLE_RATE,
        buffer_capacity=6, samples_per_target=4, ramp_epochs=0,
        backend_factory=_FakeRenderBackend,
    )
    module._backend = _FakeRenderBackend()

    audio = torch.stack([train_dataset[i][0] for i in range(4)])
    targets = torch.stack([train_dataset[i][1] for i in range(4)])
    loss = module.training_step([audio, targets], 0)
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [p.grad for p in network.parameters() if p.grad is not None]
    assert gradients, "REINFORCE should produce gradients on the policy network"
    assert all(torch.isfinite(g).all() for g in gradients)


def training_config_optimizer():
    from models.training.config import OptimizerConfig
    return OptimizerConfig(learning_rate=5e-3)


def test_reward_climbs_when_overfitting_a_tiny_corpus(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=8, seed=0)
    log_dir = tmp_path / "logs"
    model = make_synthrli(default_root_dir=str(log_dir))
    model.fit(train_dataset, config=rl_config(max_epochs=15))

    rewards = logged_metric(log_dir, "train_reward")
    assert len(rewards) >= 2
    # Later training reward beats the start (mean of the last third vs the first third).
    third = max(1, len(rewards) // 3)
    assert np.mean(rewards[-third:]) > np.mean(rewards[:third])


def test_warm_start_loads_the_synthrlp_checkpoint(tmp_path):
    train_dataset = build_corpus(tmp_path, "train", count=8, seed=0)
    space = train_dataset.parameter_space

    pretrained = SynthRLp(default_root_dir=str(tmp_path / "p_logs"), **TINY_KWARGS)
    pretrained.fit(train_dataset, config=rl_config(max_epochs=2))
    checkpoint_path = tmp_path / "synthrl_p.pt"
    pretrained.save(checkpoint_path)

    # Build a fresh SynthRL-i network and warm-start it from the -p checkpoint.
    warm = make_synthrli(init_from_checkpoint=str(checkpoint_path))
    hparams = warm._build_architecture_hparams(train_dataset, space)
    network = warm._build_network(hparams)
    warm._warm_start_network(network, hparams)

    saved_state = load_checkpoint(checkpoint_path)["state_dict"]
    for name, tensor in network.state_dict().items():
        assert torch.allclose(tensor, saved_state[name]), f"{name} not warm-started"


def test_fit_then_evaluate_through_the_evaluator(tmp_path, monkeypatch):
    train_dataset = build_corpus(tmp_path, "train", count=8, seed=0)
    model = make_synthrli(default_root_dir=str(tmp_path / "logs"))
    model.fit(train_dataset, config=rl_config(max_epochs=4))

    monkeypatch.setattr(evaluator_module, "FreshProcessRenderBackend", _FakeEvalBackend)
    corpus = build_corpus(tmp_path, "eval", count=4, seed=1)
    result = Evaluator(corpus).evaluate(model, out_dir=tmp_path / "results")

    assert len(result.per_sample_metrics) == 4
    assert result.summary["model_class"] == "SynthRLi"
