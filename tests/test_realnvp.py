"""Tests for the plain-torch RealNVP port (``models/presetgen_vae/realnvp.py``, issue #35).

Shape/finiteness/determinism checks plus the ``CustomRealNVP`` layer schedule the port
must reproduce: alternating checkerboard masks, and no flow batch-norm or dropout on the
two last coupling layers.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from models.presetgen_vae.realnvp import AffineCouplingLayer, FlowBatchNorm, RealNVP


def build_flow(features: int = 7, coupling_layers: int = 4, dropout: float = 0.5) -> RealNVP:
    return RealNVP(
        features=features, hidden_features=16, coupling_layers=coupling_layers, dropout=dropout
    )


def test_forward_preserves_shape_for_odd_feature_count():
    flow = build_flow(features=7)
    flow.eval()
    inputs = torch.randn(3, 7)
    assert flow(inputs).shape == (3, 7)


def test_forward_with_log_determinant_shapes_and_finiteness():
    flow = build_flow()
    flow.train()
    inputs = torch.randn(4, 7)
    outputs, log_abs_determinant = flow.forward_with_log_determinant(inputs)
    assert outputs.shape == (4, 7)
    assert log_abs_determinant.shape == (4,)
    assert torch.isfinite(outputs).all() and torch.isfinite(log_abs_determinant).all()


def test_eval_forward_is_deterministic():
    flow = build_flow()
    flow.eval()
    inputs = torch.randn(2, 7)
    assert torch.allclose(flow(inputs), flow(inputs))


def test_no_batch_norm_or_dropout_on_last_two_coupling_layers():
    flow = build_flow(features=6, coupling_layers=4, dropout=0.5)
    coupling_layers = [m for m in flow.layers if isinstance(m, AffineCouplingLayer)]
    batch_norm_layers = [m for m in flow.layers if isinstance(m, FlowBatchNorm)]
    assert len(coupling_layers) == 4
    assert len(batch_norm_layers) == 2  # between layers, but not after the last two couplings
    assert isinstance(flow.layers[-1], AffineCouplingLayer)
    assert isinstance(flow.layers[-2], AffineCouplingLayer)
    # Dropout is active in the early conditioners and disabled in the two last ones.
    assert coupling_layers[0].conditioner.blocks[0].dropout.p == 0.5
    assert coupling_layers[-2].conditioner.blocks[0].dropout.p == 0.0
    assert coupling_layers[-1].conditioner.blocks[0].dropout.p == 0.0


def test_alternating_masks_swap_identity_and_transform_halves():
    flow = build_flow(features=5, coupling_layers=2)
    first, second = [m for m in flow.layers if isinstance(m, AffineCouplingLayer)]
    assert set(first.transform_indices.tolist()) == set(second.identity_indices.tolist())
    assert set(first.identity_indices.tolist()) == set(second.transform_indices.tolist())


def test_features_below_two_raise():
    with pytest.raises(ValueError, match="features >= 2"):
        RealNVP(features=1, hidden_features=8, coupling_layers=2)


def test_gradients_flow_through_outputs_and_log_determinant():
    flow = build_flow()
    flow.train()
    inputs = torch.randn(4, 7, requires_grad=True)
    outputs, log_abs_determinant = flow.forward_with_log_determinant(inputs)
    (outputs.sum() + log_abs_determinant.sum()).backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
