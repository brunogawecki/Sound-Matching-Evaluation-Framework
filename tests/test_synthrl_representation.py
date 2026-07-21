import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.parameter_space import ParameterSpecification, ParameterSpace
from models.synthrl.representation import SynthRLRepresentation


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


def test_class_counts_bins_continuous_keeps_categorical_native():
    representation = SynthRLRepresentation(make_space(), num_bins=25)
    # continuous -> num_bins, categorical -> option cardinality, in subset order.
    assert representation.class_counts == [25, 3, 25, 2]
    assert representation.total_class_dimension == 25 + 3 + 25 + 2


def test_class_slices_partition_the_flat_vector_exactly():
    representation = SynthRLRepresentation(make_space(), num_bins=8)
    covered = []
    for block_slice in representation.class_slices:
        covered.extend(range(block_slice.start, block_slice.stop))
    assert covered == list(range(representation.total_class_dimension))


def test_roundtrip_decodes_within_one_bin_for_continuous_exact_for_categorical():
    space = make_space()
    num_bins = 25
    representation = SynthRLRepresentation(space, num_bins=num_bins)
    rng = np.random.default_rng(0)
    spec_by_name = {spec.name: spec for spec in space.parameter_specs}
    for _ in range(200):
        params = space.sample_uniform(rng)
        recovered = representation.class_indices_to_synth_dict(
            representation.synth_dict_to_class_indices(params)
        )
        for name, value in params.items():
            spec = spec_by_name[name]
            if spec.kind == "categorical":
                assert recovered[name] == value
            else:
                low, high = spec.bounds
                half_bin = (high - low) / (2.0 * num_bins)
                assert abs(recovered[name] - value) <= half_bin + 1e-12


def test_bin_edges_map_to_first_and_last_bin():
    space = ParameterSpace([ParameterSpecification(name="CONT", kind="continuous", bounds=(0.2, 0.8))])
    representation = SynthRLRepresentation(space, num_bins=10)
    assert representation.synth_dict_to_class_indices({"CONT": 0.2})[0] == 0
    assert representation.synth_dict_to_class_indices({"CONT": 0.8})[0] == 9


def test_class_logits_decode_via_per_head_argmax():
    representation = SynthRLRepresentation(make_space(), num_bins=8)
    indices = np.array([3, 2, 5, 1])
    smooth = representation.smoothed_target_vector(indices)
    # Argmax of the (peaked) smoothed target recovers the source indices.
    assert representation.class_logits_to_class_indices(smooth).tolist() == indices.tolist()


def test_smoothed_targets_sum_to_one_per_head():
    representation = SynthRLRepresentation(make_space(), num_bins=8, label_smoothing_sigma=1.0)
    target = representation.smoothed_target_vector(np.array([3, 2, 5, 1]))
    for block_slice in representation.class_slices:
        assert abs(float(target[block_slice].sum()) - 1.0) < 1e-9


def test_categorical_heads_are_one_hot_continuous_heads_are_smoothed():
    space = make_space()
    representation = SynthRLRepresentation(space, num_bins=8, label_smoothing_sigma=1.0)
    target = representation.smoothed_target_vector(np.array([3, 2, 5, 1]))
    slices = representation.class_slices
    # CAT B (position 1) is one-hot; only the target class carries mass.
    cat_block = target[slices[1]]
    assert np.count_nonzero(cat_block) == 1
    assert cat_block[2] == 1.0
    # CONT A (position 0) is Gaussian-smoothed; neighbours of bin 3 carry mass.
    cont_block = target[slices[0]]
    assert cont_block[3] == cont_block.max()
    assert cont_block[2] > 0.0 and cont_block[4] > 0.0


def test_zero_sigma_makes_continuous_heads_one_hot():
    representation = SynthRLRepresentation(make_space(), num_bins=8, label_smoothing_sigma=0.0)
    target = representation.smoothed_target_vector(np.array([3, 2, 5, 1]))
    cont_block = target[representation.class_slices[0]]
    assert np.count_nonzero(cont_block) == 1
    assert cont_block[3] == 1.0
