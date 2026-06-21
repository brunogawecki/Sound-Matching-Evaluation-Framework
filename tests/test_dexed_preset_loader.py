import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.parameter_space import ParameterSpecification, ParameterSpace
from dataset.dexed_preset_loader import DexedPresetLoader


# ---------------------------------------------------------------------------
# Pure-Python: synthetic cartridges built byte-by-byte, no VST required.
# A small ParameterSpace over real Dexed names (a subset of unpack_voice's
# output) is enough to exercise projection / dedup / split.
# ---------------------------------------------------------------------------

def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="OP1 OUTPUT LEVEL", kind="continuous", default=0.0),
        ParameterSpecification(name="OP1 F FINE", kind="continuous", default=0.0),
        ParameterSpecification(
            name="ALGORITHM", kind="categorical",
            options=[n / 31 for n in range(32)], default=0.0,
        ),
    ])


def op_offset(op: int) -> int:
    return (6 - op) * 17


def voice_bytes(output_level: int = 0, algorithm: int = 0) -> bytes:
    voice = bytearray(128)
    voice[op_offset(1) + 14] = output_level  # OP1 OUTPUT LEVEL (0..99)
    voice[110] = algorithm                   # ALGORITHM (0..31)
    return bytes(voice)


def cartridge(voices: list) -> bytes:
    """Wrap 32 packed voices in a valid 32-voice bulk-dump envelope."""
    assert len(voices) == 32
    body = b"".join(voices)
    header = bytes([0xF0, 0x43, 0x00, 0x09, 0x20, 0x00])
    checksum = (128 - (sum(body) & 0x7F)) & 0x7F
    return header + body + bytes([checksum, 0xF7])


@pytest.fixture
def write_cart(tmp_path):
    counter = {"n": 0}

    def _write(voices: list, name: str = None) -> str:
        counter["n"] += 1
        path = tmp_path / (name or f"cart_{counter['n']}.syx")
        path.write_bytes(cartridge(voices))
        return str(path)

    return _write


def test_loads_32_voices_per_cartridge(write_cart):
    space = make_space()
    loader = DexedPresetLoader(space, test_fraction=0.0)
    voices = [voice_bytes(output_level=i + 1) for i in range(32)]  # all distinct
    split = loader.load([write_cart(voices)])
    assert len(split.train) == 32
    assert len(split.test) == 0


def test_loads_voices_from_multiple_cartridges(write_cart):
    space = make_space()
    loader = DexedPresetLoader(space, test_fraction=0.0)
    bank_a = [voice_bytes(output_level=i + 1) for i in range(32)]
    bank_b = [voice_bytes(output_level=i + 1, algorithm=1) for i in range(32)]
    split = loader.load([write_cart(bank_a), write_cart(bank_b)])
    assert len(split.train) == 64


def test_dedup_collapses_identical_voices(write_cart):
    space = make_space()
    loader = DexedPresetLoader(space, test_fraction=0.0)
    voices = [voice_bytes(output_level=5)] * 32  # all identical
    split = loader.load([write_cart(voices)])
    assert len(split.train) == 1


def test_dedup_collapses_voices_differing_only_in_dropped_params(write_cart):
    # Two voices identical on the subset (OUTPUT LEVEL, F FINE, ALGORITHM) but
    # differing in a dropped param (OP1 BREAK POINT) must collapse: they render
    # identically under the fixed contract.
    space = make_space()
    base = voice_bytes(output_level=10)
    twin = bytearray(voice_bytes(output_level=10))
    twin[op_offset(1) + 8] = 50  # OP1 BREAK POINT -- a dropped (non-subset) param
    voices = [base, bytes(twin)] + [voice_bytes(output_level=20 + i) for i in range(30)]
    loader = DexedPresetLoader(space, test_fraction=0.0)
    split = loader.load([write_cart(voices)])
    assert len(split.train) == 31  # base and twin collapsed to one


def test_distinct_algorithms_are_not_deduplicated(write_cart):
    space = make_space()
    loader = DexedPresetLoader(space, test_fraction=0.0)
    voices = [voice_bytes(algorithm=i, output_level=10) for i in range(32)]
    split = loader.load([write_cart(voices)])
    assert len(split.train) == 32  # one-hot ALGORITHM blocks differ


def test_split_is_disjoint_deterministic_and_correctly_sized(write_cart):
    space = make_space()
    voices = [voice_bytes(output_level=i + 1) for i in range(32)]
    path = write_cart(voices, name="bank.syx")
    split = DexedPresetLoader(space, test_fraction=0.25, split_seed=123).load([path])
    assert len(split.test) == 8  # round(32 * 0.25)
    assert len(split.train) == 24

    train_ids = {(p.source_file, p.voice_index) for p in split.train}
    test_ids = {(p.source_file, p.voice_index) for p in split.test}
    assert train_ids.isdisjoint(test_ids)

    again = DexedPresetLoader(space, test_fraction=0.25, split_seed=123).load([path])
    assert [p.voice_index for p in again.test] == [p.voice_index for p in split.test]


def test_provenance_records_source_and_voice_name(write_cart):
    space = make_space()
    voices = [voice_bytes(output_level=i + 1) for i in range(32)]
    named = bytearray(voices[0])
    named[118:128] = b"PIANO 1   "  # name in the trailing 10 bytes
    voices[0] = bytes(named)
    path = write_cart(voices, name="factory.syx")
    split = DexedPresetLoader(space, test_fraction=0.0).load([path])
    voice_zero = next(p for p in split.train if p.voice_index == 0)
    assert voice_zero.source_file == "factory.syx"
    assert voice_zero.voice_name == "PIANO 1"  # trailing spaces stripped
