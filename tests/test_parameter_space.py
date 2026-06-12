import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from synth.parameter_space import ParameterSpecification, ParameterSpace


# ---------------------------------------------------------------------------
# Pure-Python tests (no plugin required)
# ---------------------------------------------------------------------------

def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="CONT A", kind="continuous", default=0.25),
        ParameterSpecification(name="CAT B", kind="categorical", options=[0.0, 0.5, 1.0], default=0.5),
        ParameterSpecification(name="CONT C", kind="continuous", bounds=(0.2, 0.8), default=0.5),
        ParameterSpecification(name="CAT D", kind="categorical", options=[0.0, 1.0], default=0.0),
    ])


def test_dims():
    space = make_space()
    assert space.synth_dimension == 4
    assert space.ml_dimension == 1 + 3 + 1 + 2


def test_loss_slices_partition_the_vector_exactly():
    space = make_space()
    slices = space.loss_slices
    assert [name for _, _, name in slices] == space.names
    assert [kind for _, kind, _ in slices] == [
        "continuous", "categorical", "continuous", "categorical"
    ]
    covered = []
    for vector_slice, _, _ in slices:
        covered.extend(range(vector_slice.start, vector_slice.stop))
    assert covered == list(range(space.ml_dimension))


def test_roundtrip_is_exact_for_grid_valid_dicts():
    space = make_space()
    rng = np.random.default_rng(0)
    for _ in range(50):
        params = space.sample_uniform(rng)
        recovered = space.ml_vector_to_synth_dict(space.synth_dict_to_ml_vector(params))
        assert recovered == params


def test_one_hot_encoding_layout():
    space = make_space()
    vector = space.synth_dict_to_ml_vector(
        {"CONT A": 0.7, "CAT B": 0.5, "CONT C": 0.3, "CAT D": 1.0}
    )
    assert vector.tolist() == [0.7, 0.0, 1.0, 0.0, 0.3, 0.0, 1.0]


def test_categorical_encode_snaps_to_nearest_option():
    space = make_space()
    vector = space.synth_dict_to_ml_vector(
        {"CONT A": 0.0, "CAT B": 0.51, "CONT C": 0.5, "CAT D": 0.4}
    )
    decoded = space.ml_vector_to_synth_dict(vector)
    assert decoded["CAT B"] == 0.5
    assert decoded["CAT D"] == 0.0


def test_decode_argmax_accepts_non_one_hot_blocks():
    space = make_space()
    vector = np.array([0.5, 0.1, 0.2, 0.9, 0.5, 0.8, 0.3])
    decoded = space.ml_vector_to_synth_dict(vector)
    assert decoded["CAT B"] == 1.0
    assert decoded["CAT D"] == 0.0


def test_decode_clips_continuous_to_bounds():
    space = make_space()
    vector = np.array([1.5, 1.0, 0.0, 0.0, -0.3, 1.0, 0.0])
    decoded = space.ml_vector_to_synth_dict(vector)
    assert decoded["CONT A"] == 1.0
    assert decoded["CONT C"] == 0.2


def test_encode_rejects_missing_and_extra_keys():
    space = make_space()
    with pytest.raises(KeyError):
        space.synth_dict_to_ml_vector({"CONT A": 0.1})
    full = space.sample_uniform(np.random.default_rng(1))
    with pytest.raises(KeyError):
        space.synth_dict_to_ml_vector({**full, "NOT A PARAM": 0.0})


def test_decode_rejects_wrong_vector_shape():
    space = make_space()
    with pytest.raises(ValueError):
        space.ml_vector_to_synth_dict(np.zeros(space.ml_dimension + 1))


def test_sample_uniform_is_deterministic_and_valid():
    space = make_space()
    a = space.sample_uniform(np.random.default_rng(123))
    b = space.sample_uniform(np.random.default_rng(123))
    assert a == b
    assert set(a.keys()) == set(space.names)
    assert a["CAT B"] in (0.0, 0.5, 1.0)
    assert a["CAT D"] in (0.0, 1.0)
    assert 0.0 <= a["CONT A"] <= 1.0
    assert 0.2 <= a["CONT C"] <= 0.8


def test_spec_validation():
    with pytest.raises(ValueError):
        ParameterSpecification(name="X", kind="categorical")  # no options
    with pytest.raises(ValueError):
        ParameterSpecification(name="X", kind="continuous", options=[0.0, 1.0])
    with pytest.raises(ValueError):
        ParameterSpecification(name="X", kind="continuous", bounds=(0.8, 0.2))
    with pytest.raises(ValueError):
        ParameterSpecification(name="X", kind="ordinal")  # unknown kind
    with pytest.raises(ValueError):
        ParameterSpace([
            ParameterSpecification(name="X", kind="continuous"),
            ParameterSpecification(name="X", kind="continuous"),
        ])


# ---------------------------------------------------------------------------
# Integration with the live plugin (provisional Dexed subset)
# ---------------------------------------------------------------------------

PLUGIN_PATH = os.path.expanduser(config.DEXED_PATH)

needs_plugin = pytest.mark.skipif(
    not os.path.exists(PLUGIN_PATH),
    reason=f"Dexed plugin not found at {PLUGIN_PATH}",
)


@pytest.fixture(scope="module")
def synth():
    from synth.dexed import DexedWrapper
    return DexedWrapper(
        plugin_path=PLUGIN_PATH,
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE,
    )


@needs_plugin
def test_dexed_parameter_space_dimensions(synth):
    space = synth.parameter_space
    assert space.synth_dimension == 29
    # ALGORITHM 32 + LFO WAVE 6 + 6x F COARSE 32 + 21 continuous
    assert space.ml_dimension == 251
    assert synth.parameter_space is space  # cached


@needs_plugin
def test_dexed_parameter_space_defaults_are_grid_valid(synth):
    space = synth.parameter_space
    defaults = {parameter_spec.name: parameter_spec.default for parameter_spec in space.parameter_specs}
    recovered = space.ml_vector_to_synth_dict(space.synth_dict_to_ml_vector(defaults))
    assert recovered == pytest.approx(defaults)


@needs_plugin
def test_dexed_sampled_subset_roundtrips_through_the_synth(synth):
    space = synth.parameter_space
    params = space.sample_uniform(np.random.default_rng(42))
    synth.set_parameters(params)
    readback = synth.get_parameters()
    for name, value in params.items():
        assert readback[name] == pytest.approx(value, abs=1e-6), name
