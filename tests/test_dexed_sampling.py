import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.dexed.synth import _AUDIBLE_SAMPLING_RANGES


# ---------------------------------------------------------------------------
# Pure-Python: no plugin required. _AUDIBLE_SAMPLING_RANGES is the declarative
# range-override map (surfaced by DexedWrapper.audible_sampling_ranges) that makes
# synthetic Dexed draws audible by construction: OP1 (a carrier in all 32 DX7
# algorithms) is sampled from calibrated high ranges, while timbre/temporal
# parameters stay free. Calibrated to the built-in Dexed presets (D-AUDIBLE).
# ---------------------------------------------------------------------------

def test_constrains_op1_loudness_params():
    assert set(_AUDIBLE_SAMPLING_RANGES) == {
        "OP1 OUTPUT LEVEL",
        "OP1 EG LEVEL 1",   # attack peak
        "OP1 EG RATE 1",    # attack not glacial
    }


def test_ranges_are_valid_high_subranges_of_the_unit_interval():
    for parameter_name, (min_val, max_val) in _AUDIBLE_SAMPLING_RANGES.items():
        assert 0.0 <= min_val < max_val <= 1.0, parameter_name
