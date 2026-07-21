"""Tests for the flow-matching CNF family (Stage 1: the CNF (MLP) model).

Plugin-independent: the flow-matching math (rectified path, OT pairing, RK4, CFG
dropout), the AST encoder / network shapes, seeded-predict determinism, and an
end-to-end fit -> save -> load -> predict smoke test on a tiny synthetic corpus
(mirroring ``tests/test_sound2synth.py``). Skips cleanly when ``torch``/``lightning``
are absent.
"""
import math
import os
import sys
from typing import Dict

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from models.flow_matching import FlowMatchingMLP, FlowMatchingParam2Tok
from models.flow_matching.encoder import AudioSpectrogramTransformer
from models.flow_matching.flow_matching import (
    optimal_transport_pairing,
    rectified_path_sample,
    rectified_target_velocity,
    rk4_sample,
)
from models.flow_matching.network import FlowMatchingNetwork
from models.flow_matching.vector_field import (
    ConditionalResidualMLPField,
    DiffusionTransformerBlock,
    EquivariantTransformerField,
    Param2TokProjection,
)
from synth.parameter_space import ParameterSpace, ParameterSpecification

SAMPLE_RATE = 8000
DURATION_SEC = 0.5
NUM_AUDIO_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

# Small transformer/mel settings so the CPU tests stay fast; the architecture wiring
# (patching, per-layer conditioning, CFG, RK4) is what is under test, not capacity.
TINY_KWARGS = dict(
    n_mels=32,
    window_duration_ms=25.0,
    hop_duration_ms=10.0,
    encoder_d_model=32,
    encoder_num_heads=4,
    encoder_num_layers=2,
    patch_size=16,
    patch_stride=10,
    field_d_model=32,
    field_num_layers=2,
    field_num_heads=4,
    num_parameter_tokens=8,
    time_encoding_dimension=16,
    sample_steps=8,
)

TINY_NETWORK_KWARGS = dict(
    num_audio_samples=NUM_AUDIO_SAMPLES,
    sample_rate=SAMPLE_RATE,
    num_conditioning_outputs=2,  # = field_num_layers, as the family derives it
    **TINY_KWARGS,
)


def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.8),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 0.5, 1.0], default=0.0),
    ])


# -- flow-matching math -------------------------------------------------------------


def test_rectified_path_interpolates_between_noise_and_data():
    x0 = torch.randn(4, 7)
    x1 = torch.randn(4, 7)
    assert torch.allclose(rectified_path_sample(x0, x1, torch.zeros(4, 1)), x0)
    assert torch.allclose(rectified_path_sample(x0, x1, torch.ones(4, 1)), x1)
    midpoint = rectified_path_sample(x0, x1, torch.full((4, 1), 0.5))
    assert torch.allclose(midpoint, (x0 + x1) / 2)
    assert torch.allclose(rectified_target_velocity(x0, x1), x1 - x0)


def test_rk4_sample_integrates_a_known_linear_ode():
    class DecayField(torch.nn.Module):
        # dx/dt = -x, so x(1) = x(0) * e^-1 regardless of conditioning.
        def forward(self, x, t, conditioning=None):
            return -x

    noise = torch.randn(5, 3)
    sample = rk4_sample(DecayField(), noise, conditioning=None, num_steps=32, cfg_strength=2.0)
    assert torch.allclose(sample, noise * math.exp(-1.0), atol=1e-5)


def test_optimal_transport_pairing_permutes_noise_and_reduces_cost():
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(16, 6, generator=generator)
    targets = torch.randn(16, 6, generator=generator)
    paired = optimal_transport_pairing(noise, targets)
    # A permutation of the original rows (each noise row used exactly once)...
    matches = (paired[:, None, :] == noise[None, :, :]).all(dim=-1)
    assert torch.equal(matches.sum(dim=1), torch.ones(16, dtype=torch.long))
    assert torch.equal(matches.sum(dim=0), torch.ones(16, dtype=torch.long))
    # ...whose row-wise coupling cost does not exceed the unpaired one.
    unpaired_cost = (noise - targets).norm(dim=-1).sum()
    paired_cost = (paired - targets).norm(dim=-1).sum()
    assert paired_cost <= unpaired_cost + 1e-6


def test_cfg_dropout_replaces_conditioning_at_rate_one_and_keeps_it_at_zero():
    field = ConditionalResidualMLPField(
        num_params=4, d_model=16, time_encoding_dimension=8, conditioning_dim=6, num_layers=2
    )
    per_layer = torch.randn(5, 2, 6)
    assert torch.equal(field.apply_dropout(per_layer, rate=0.0), per_layer)
    dropped = field.apply_dropout(per_layer, rate=1.0)
    assert torch.allclose(dropped, field.cfg_dropout_token.expand(5, 2, 6))
    single = torch.randn(5, 6)
    dropped_single = field.apply_dropout(single, rate=1.0)
    assert torch.allclose(dropped_single, field.cfg_dropout_token[0].expand(5, 6))


# -- vector field + encoder shapes --------------------------------------------------


def test_mlp_field_maps_state_to_velocity_with_per_layer_conditioning():
    field = ConditionalResidualMLPField(
        num_params=9, d_model=16, time_encoding_dimension=8, conditioning_dim=6, num_layers=3
    )
    x = torch.randn(4, 9)
    t = torch.rand(4, 1)
    per_layer_conditioning = torch.randn(4, 3, 6)
    assert field(x, t, per_layer_conditioning).shape == (4, 9)
    assert field(x, t, None).shape == (4, 9)  # unconditional (CFG) branch
    assert field.penalty() == 0.0


def test_param2tok_projection_maps_params_to_tokens_and_back():
    projection = Param2TokProjection(d_model=16, d_token=16, num_params=9, num_tokens=5)
    x = torch.randn(4, 9)
    tokens = projection.param_to_token(x)
    assert tokens.shape == (4, 5, 16)
    assert projection.token_to_param(tokens).shape == (4, 9)
    penalty = projection.penalty()
    assert penalty.ndim == 0 and torch.isfinite(penalty) and penalty > 0.0


def test_dit_block_is_permutation_equivariant_over_its_tokens():
    """No positional encoding -> permuting the tokens permutes the output the same way.

    This is the property the whole Param2Tok argument rests on, so it is asserted on the
    block directly rather than inferred from the field's output.
    """
    block = DiffusionTransformerBlock(
        d_model=16, conditioning_dim=16, num_heads=4, feedforward_dimension=16
    )
    block.eval()
    tokens = torch.randn(2, 6, 16)
    conditioning = torch.randn(2, 16)
    permutation = torch.randperm(6)
    with torch.no_grad():
        straight = block(tokens, conditioning)[:, permutation]
        permuted = block(tokens[:, permutation], conditioning)
    assert torch.allclose(straight, permuted, atol=1e-5)


def test_equivariant_field_maps_state_to_velocity_and_penalizes_the_assignment():
    field = EquivariantTransformerField(
        num_params=9,
        d_model=16,
        time_encoding_dimension=8,
        conditioning_dim=6,
        num_layers=3,
        num_heads=4,
        num_tokens=5,
        projection_penalty=0.01,
    )
    x = torch.randn(4, 9)
    t = torch.rand(4, 1)
    per_layer_conditioning = torch.randn(4, 3, 6)
    assert field(x, t, per_layer_conditioning).shape == (4, 9)
    assert field(x, t, None).shape == (4, 9)  # unconditional (CFG) branch
    # penalty() returns the *weighted* L1, so the training step can add it unscaled.
    assert torch.allclose(field.penalty(), 0.01 * field.projection.assignment.abs().mean())


def test_ast_encoder_emits_one_conditioning_vector_per_output_token():
    encoder = AudioSpectrogramTransformer(
        d_model=32,
        num_heads=4,
        num_layers=2,
        num_conditioning_outputs=3,
        patch_size=16,
        patch_stride=10,
        input_channels=1,
        spectrogram_shape=(32, 51),
    )
    spectrogram = torch.randn(2, 1, 32, 51)
    assert encoder(spectrogram).shape == (2, 3, 32)


# -- network ------------------------------------------------------------------------


def test_network_featurize_matches_the_render_contract_shape():
    network = FlowMatchingNetwork(ml_dimension=4, **TINY_NETWORK_KWARGS)
    features = network.featurize(torch.randn(2, NUM_AUDIO_SAMPLES))
    hop_length = int(TINY_KWARGS["hop_duration_ms"] / 1000.0 * SAMPLE_RATE)
    assert features.shape == (2, 1, TINY_KWARGS["n_mels"], 1 + NUM_AUDIO_SAMPLES // hop_length)


def test_network_sample_is_deterministic_given_a_seeded_generator():
    network = FlowMatchingNetwork(ml_dimension=4, **TINY_NETWORK_KWARGS)
    network.eval()
    audio = torch.randn(2, NUM_AUDIO_SAMPLES)
    with torch.no_grad():
        first = network.sample(audio, generator=torch.Generator().manual_seed(7))
        second = network.sample(audio, generator=torch.Generator().manual_seed(7))
        different_seed = network.sample(audio, generator=torch.Generator().manual_seed(8))
    assert first.shape == (2, 4)
    assert torch.equal(first, second)
    assert not torch.allclose(first, different_seed)


def test_build_network_is_deterministic_in_hparams():
    model = FlowMatchingMLP(**TINY_KWARGS)
    hparams = {
        "ml_dimension": 4,
        "mel_mean_db": -40.0,
        "mel_std_db": 20.0,
        "vector_field_architecture": "mlp",
        **TINY_NETWORK_KWARGS,
    }
    first = model._build_network(hparams)
    second = model._build_network(hparams)
    first_shapes = {name: tuple(p.shape) for name, p in first.state_dict().items()}
    second_shapes = {name: tuple(p.shape) for name, p in second.state_dict().items()}
    assert first_shapes == second_shapes


def test_param2tok_network_samples_deterministically():
    network = FlowMatchingNetwork(
        ml_dimension=4, vector_field_architecture="param2tok", **TINY_NETWORK_KWARGS
    )
    network.eval()
    audio = torch.randn(2, NUM_AUDIO_SAMPLES)
    with torch.no_grad():
        first = network.sample(audio, generator=torch.Generator().manual_seed(7))
        second = network.sample(audio, generator=torch.Generator().manual_seed(7))
    assert first.shape == (2, 4)
    assert torch.equal(first, second)


@pytest.mark.parametrize("model_class", [FlowMatchingMLP, FlowMatchingParam2Tok])
def test_predict_is_seeded_and_decodes_to_a_valid_synth_dict(model_class):
    space = make_space()
    model = model_class(**TINY_KWARGS)
    network = FlowMatchingNetwork(
        ml_dimension=space.ml_dimension,
        vector_field_architecture=model_class._vector_field_architecture,
        **TINY_NETWORK_KWARGS,
    )
    model._set_trained_network(network, {"ml_dimension": space.ml_dimension}, space)

    audio = torch.randn(NUM_AUDIO_SAMPLES)
    first = model.predict(audio)
    second = model.predict(audio)
    assert first == second  # per-call seeded generator -> reproducible sample
    assert set(first) == set(space.names)
    assert first["CAT"] in (0.0, 0.5, 1.0)
    assert 0.0 <= first["AMP"] <= 1.0


# -- end-to-end fit -> save -> load -> predict --------------------------------------


class FakeSynth:
    """A no-VST sine synth (mirrors tests/test_sound2synth.py)."""

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
        samples = int(duration_sec * self._sample_rate)
        time = np.arange(samples) / self._sample_rate
        return float(self._state["AMP"]) * np.sin(2.0 * np.pi * 220.0 * time)


def build_corpus(tmp_path, run_name, count, seed):
    from dataset.builder import DatasetBuilder, RenderSettings
    from dataset.preset_sources import SyntheticPresetSource
    from dataset.torch_dataset import RenderedCorpusDataset

    synth = FakeSynth(make_space())
    settings = RenderSettings(
        midi_note=60, velocity=100, duration_sec=DURATION_SEC, note_duration_sec=DURATION_SEC
    )
    source = SyntheticPresetSource(
        make_space(), count=count, seed=seed, sampling_ranges={"AMP": (0.7, 1.0)}
    )
    DatasetBuilder(synth, render_settings=settings).build(
        source, run_name=run_name, output_root=tmp_path
    )
    return RenderedCorpusDataset.load(tmp_path / run_name)


@pytest.mark.parametrize("model_class", [FlowMatchingMLP, FlowMatchingParam2Tok])
def test_fit_export_load_predict_end_to_end(tmp_path, model_class):
    pytest.importorskip("lightning")  # training-only dependency; skip locally if absent
    import pandas as pd

    train_dataset = build_corpus(tmp_path, "train", count=16, seed=0)
    log_dir = tmp_path / "logs"

    model = model_class(
        default_root_dir=str(log_dir), validation_sample_steps=4, **TINY_KWARGS
    )
    model.fit(
        train_dataset,
        config={
            "seed": 0,
            "optimizer": {"learning_rate": 1e-3},
            "data": {"batch_size": 4, "val_fraction": 0.25},
            "trainer": {
                "max_epochs": 2,
                "precision": "32-true",
                "accelerator": "cpu",
                "devices": 1,
                "log_every_n_steps": 1,
            },
        },
    )

    # The flow objective and the sampling-based val monitor were both logged and finite.
    metrics_files = list(log_dir.rglob("metrics.csv"))
    assert metrics_files, "CSVLogger should have written a metrics.csv"
    metrics = pd.read_csv(metrics_files[0])
    assert np.isfinite(metrics["train_loss"].dropna()).all()
    assert np.isfinite(metrics["val_loss"].dropna()).all()

    checkpoint_path = tmp_path / "flow_matching.pt"
    model.save(checkpoint_path)
    assert checkpoint_path.exists()

    # Fresh instance loads with no dataset, no VST, no Lightning -- then predicts.
    reloaded = model_class(**TINY_KWARGS)
    reloaded.load(checkpoint_path)

    audio, _ = train_dataset[0]
    prediction = reloaded.predict(audio)
    space = train_dataset.parameter_space
    assert set(prediction) == set(space.names)
    assert prediction["CAT"] in (0.0, 0.5, 1.0)
    assert 0.0 <= prediction["AMP"] <= 1.0
    # The reloaded network reproduces the trained model's seeded prediction exactly.
    assert prediction == model.predict(audio)
