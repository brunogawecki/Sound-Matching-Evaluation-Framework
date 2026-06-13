import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from synth.dexed import DexedWrapper
from synth.dexed.cartridge import (
    unpack_voice,
    voice_names,
    voice_parameters,
    validate_cartridge,
)

PLUGIN_PATH = os.path.expanduser(config.DEXED_PATH)
CARTRIDGE_PATH = os.path.expanduser(
    os.getenv(
        "DEXED_TEST_CARTRIDGE",
        "~/Library/Application Support/DigitalSuburban/Dexed/Cartridges/Dexed_01.syx",
    )
)

requires_plugin = pytest.mark.skipif(
    not os.path.exists(PLUGIN_PATH),
    reason=f"Dexed plugin not found at {PLUGIN_PATH}",
)
requires_cartridge = pytest.mark.skipif(
    not os.path.exists(CARTRIDGE_PATH),
    reason=f"No test cartridge at {CARTRIDGE_PATH} (set DEXED_TEST_CARTRIDGE)",
)


def make_wrapper() -> DexedWrapper:
    return DexedWrapper(PLUGIN_PATH, sample_rate=config.SAMPLE_RATE, buffer_size=config.BUFFER_SIZE)


def fake_cartridge(voices: bytes) -> bytes:
    """Wrap 4096 bytes of voice data in a valid 32-voice bulk-dump envelope."""
    assert len(voices) == 4096
    header = bytes([0xF0, 0x43, 0x00, 0x09, 0x20, 0x00])
    checksum = (128 - (sum(voices) & 0x7F)) & 0x7F
    return header + voices + bytes([checksum, 0xF7])


# A packed operator is 17 bytes; operators are stored OP6..OP1, so OP1 is the
# last block (offset 85) and OP6 is the first (offset 0).
def op_offset(op: int) -> int:
    return (6 - op) * 17


# ---------------------------------------------------------------------------
# unpack_voice: known bytes -> normalized parameter values (pure, no VST)
# ---------------------------------------------------------------------------

def test_unpack_all_zero_voice():
    params = unpack_voice(bytes(128))
    assert params["OP1 OUTPUT LEVEL"] == 0.0
    assert params["OP1 MODE"] == 0.0
    assert params["OP1 F COARSE"] == 0.0
    assert params["OP1 OSC DETUNE"] == 0.0  # detune 0 of 0..14
    assert params["ALGORITHM"] == 0.0
    assert params["FEEDBACK"] == 0.0
    # Operators have no on/off in the DX7 voice format; they default to on.
    assert all(params[f"OP{op} SWITCH"] == 1.0 for op in range(1, 7))


def test_unpack_master_tune_is_not_a_voice_parameter():
    assert "MASTER TUNE ADJ" not in unpack_voice(bytes(128))


def test_unpack_operator_order_op1_is_last_block():
    voice = bytearray(128)
    voice[op_offset(1) + 14] = 99  # OP1 output level (full)
    voice[op_offset(6) + 14] = 50  # OP6 output level
    params = unpack_voice(bytes(voice))
    assert params["OP1 OUTPUT LEVEL"] == 1.0
    assert params["OP6 OUTPUT LEVEL"] == 50 / 99


def test_unpack_coarse_and_mode_share_a_byte():
    voice = bytearray(128)
    voice[op_offset(1) + 15] = (10 << 1) | 1  # F COARSE=10, MODE=1
    params = unpack_voice(bytes(voice))
    assert params["OP1 F COARSE"] == 10 / 31
    assert params["OP1 MODE"] == 1.0


def test_unpack_detune_and_rate_scaling_share_a_byte():
    voice = bytearray(128)
    voice[op_offset(1) + 12] = (14 << 3) | 7  # DETUNE=14, RATE SCALING=7
    params = unpack_voice(bytes(voice))
    assert params["OP1 OSC DETUNE"] == 1.0
    assert params["OP1 RATE SCALING"] == 1.0


def test_unpack_key_velocity_and_amp_mod_share_a_byte():
    voice = bytearray(128)
    voice[op_offset(1) + 13] = (7 << 2) | 3  # KEY VELOCITY=7, A MOD SENS=3
    params = unpack_voice(bytes(voice))
    assert params["OP1 KEY VELOCITY"] == 1.0
    assert params["OP1 A MOD SENS."] == 1.0


def test_unpack_keyboard_scaling_curves_share_a_byte():
    voice = bytearray(128)
    voice[op_offset(1) + 11] = (3 << 2) | 2  # R CURVE=3, L CURVE=2
    params = unpack_voice(bytes(voice))
    assert params["OP1 L KEY SCALE"] == 2 / 3
    assert params["OP1 R KEY SCALE"] == 1.0


def test_unpack_algorithm_feedback_and_oscsync():
    voice = bytearray(128)
    voice[110] = 31                  # ALGORITHM (0..31)
    voice[111] = (1 << 3) | 7        # OSC KEY SYNC=1, FEEDBACK=7
    params = unpack_voice(bytes(voice))
    assert params["ALGORITHM"] == 1.0
    assert params["FEEDBACK"] == 1.0
    assert params["OSC KEY SYNC"] == 1.0


def test_unpack_lfo_byte_packs_sync_wave_and_pms():
    voice = bytearray(128)
    voice[116] = (7 << 4) | (5 << 1) | 1  # P MODE SENS=7, LFO WAVE=5, LFO KEY SYNC=1
    params = unpack_voice(bytes(voice))
    assert params["LFO KEY SYNC"] == 1.0
    assert params["LFO WAVE"] == 1.0      # 5 of 0..5
    assert params["P MODE SENS."] == 1.0  # 7 of 0..7


def test_unpack_transpose():
    voice = bytearray(128)
    voice[117] = 48
    assert unpack_voice(bytes(voice))["TRANSPOSE"] == 1.0


# ---------------------------------------------------------------------------
# Cartridge envelope: names, validation, voice selection (pure)
# ---------------------------------------------------------------------------

def test_voice_names_reads_the_trailing_ten_bytes():
    voices = bytearray(4096)
    name = b"TESTVOICE!"
    voices[118:128] = name  # name of voice 0
    names = voice_names(fake_cartridge(bytes(voices)))
    assert len(names) == 32
    assert names[0] == "TESTVOICE!"


def test_validate_rejects_wrong_size():
    with pytest.raises(ValueError):
        validate_cartridge(b"\xf0\x43\x00\x09\xf7")


def test_validate_rejects_bad_format_byte():
    bad = bytearray(fake_cartridge(bytes(4096)))
    bad[3] = 0x00  # not a 32-voice bulk dump
    with pytest.raises(ValueError):
        validate_cartridge(bytes(bad))


@pytest.mark.parametrize("voice_index", [-1, 32, 100])
def test_voice_parameters_rejects_out_of_range(voice_index):
    with pytest.raises(ValueError):
        voice_parameters(fake_cartridge(bytes(4096)), voice_index)


# ---------------------------------------------------------------------------
# Real cartridge parsing validated against Dexed (requires VST + cartridge)
# ---------------------------------------------------------------------------

@requires_plugin
@requires_cartridge
def test_parsed_first_voice_matches_dexed_defaults():
    """Dexed's init state is voice 0 of its factory cartridge (Dexed_01); the
    parsed voice-0 parameters must reproduce the wrapper's defaults exactly."""
    synth = make_wrapper()
    defaults = synth.get_parameter_defaults()
    parsed = voice_parameters(open(CARTRIDGE_PATH, "rb").read(), 0)
    for name, value in parsed.items():
        assert value == pytest.approx(defaults[name], abs=1e-6), name


# ---------------------------------------------------------------------------
# render_cartridge_voice (requires VST + cartridge)
# ---------------------------------------------------------------------------

@requires_plugin
@requires_cartridge
def test_render_cartridge_voice_is_mono_correct_length_and_audible():
    synth = make_wrapper()
    audio = synth.render_cartridge_voice(
        CARTRIDGE_PATH, voice_index=0, midi_note=60, velocity=100, duration_sec=2.0
    )
    assert audio.ndim == 1
    assert len(audio) == int(2.0 * synth.sample_rate)
    assert np.sqrt(np.mean(audio**2)) > 1e-4


@requires_plugin
@requires_cartridge
def test_different_voices_render_differently():
    voice_a = make_wrapper().render_cartridge_voice(
        CARTRIDGE_PATH, voice_index=0, midi_note=60, velocity=100, duration_sec=2.0
    )
    voice_b = make_wrapper().render_cartridge_voice(
        CARTRIDGE_PATH, voice_index=2, midi_note=60, velocity=100, duration_sec=2.0
    )
    assert not np.array_equal(voice_a, voice_b)


@requires_plugin
@requires_cartridge
def test_render_cartridge_voice_rejects_out_of_range_index():
    synth = make_wrapper()
    with pytest.raises(ValueError):
        synth.render_cartridge_voice(
            CARTRIDGE_PATH, voice_index=99, midi_note=60, velocity=100, duration_sec=1.0
        )
