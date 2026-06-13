"""Parse DX7 .syx cartridges into Dexed parameter dictionaries.

A DX7 ``.syx`` cartridge is a MIDI System Exclusive 32-voice bulk dump: a 6-byte
header, 4096 bytes of packed voice data (32 voices x 128 bytes), a checksum byte,
and an ``0xF7`` terminator. Each packed voice unpacks to the DX7 synthesis
parameters, which this module maps onto Dexed's plugin-reported parameter names
(matching synth/dexed/synth.py) with values normalized to [0, 1] the same way
Dexed itself normalizes them (raw field value / field maximum; categoricals as
index / (cardinality - 1)).

This is the only reliable way to load a specific cartridge voice into Dexed under
DawDreamer: its SysEx and MIDI Program Change are ignored in offline rendering, so
the voice must be applied as parameters (see docs/DECISIONS.md / project memory).

Operators are stored in the packed voice in reverse order (OP6 first, OP1 last).
"""

from typing import Dict, List

BULK_DUMP_SIZE = 4104
BULK_DUMP_FORMAT = 0x09  # SysEx format number for a 32-voice bulk dump
NUM_VOICES = 32
_HEADER_SIZE = 6
_PACKED_VOICE_SIZE = 128
_OPERATOR_SIZE = 17
_NAME_LENGTH = 10


def validate_cartridge(data: bytes) -> None:
    """Raise ValueError unless ``data`` is a DX7 32-voice bulk dump."""
    if (
        len(data) != BULK_DUMP_SIZE
        or data[0] != 0xF0
        or data[1] != 0x43
        or data[3] != BULK_DUMP_FORMAT
        or data[-1] != 0xF7
    ):
        raise ValueError(
            "Not a DX7 32-voice bulk dump: expected a 4104-byte SysEx starting "
            "with F0 43 0n 09 and ending with F7."
        )


def _packed_voice(data: bytes, voice_index: int) -> bytes:
    start = _HEADER_SIZE + voice_index * _PACKED_VOICE_SIZE
    return data[start:start + _PACKED_VOICE_SIZE]


def voice_names(data: bytes) -> List[str]:
    """The 32 patch names (last 10 bytes of each packed voice)."""
    validate_cartridge(data)
    names = []
    for voice_index in range(NUM_VOICES):
        voice = _packed_voice(data, voice_index)
        raw = bytes(voice[_PACKED_VOICE_SIZE - _NAME_LENGTH:_PACKED_VOICE_SIZE])
        names.append(raw.decode("ascii", errors="replace"))
    return names


def unpack_voice(packed_voice: bytes) -> Dict[str, float]:
    """Unpack one 128-byte packed DX7 voice into normalized Dexed parameters.

    Excludes ``MASTER TUNE ADJ``, which is a Dexed global, not part of a DX7 voice.
    """
    if len(packed_voice) != _PACKED_VOICE_SIZE:
        raise ValueError(f"A packed voice must be {_PACKED_VOICE_SIZE} bytes.")

    params: Dict[str, float] = {}

    for op in range(1, 7):
        block = packed_voice[(6 - op) * _OPERATOR_SIZE:(6 - op) * _OPERATOR_SIZE + _OPERATOR_SIZE]
        prefix = f"OP{op} "
        for i in range(4):
            params[f"{prefix}EG RATE {i + 1}"] = block[i] / 99
        for i in range(4):
            params[f"{prefix}EG LEVEL {i + 1}"] = block[4 + i] / 99
        params[f"{prefix}BREAK POINT"] = block[8] / 99
        params[f"{prefix}L SCALE DEPTH"] = block[9] / 99
        params[f"{prefix}R SCALE DEPTH"] = block[10] / 99
        params[f"{prefix}L KEY SCALE"] = (block[11] & 0x03) / 3          # left curve, 0..3
        params[f"{prefix}R KEY SCALE"] = ((block[11] >> 2) & 0x03) / 3   # right curve, 0..3
        params[f"{prefix}RATE SCALING"] = (block[12] & 0x07) / 7         # 0..7
        params[f"{prefix}OSC DETUNE"] = ((block[12] >> 3) & 0x0F) / 14   # 0..14 (7 = no detune)
        params[f"{prefix}A MOD SENS."] = (block[13] & 0x03) / 3          # 0..3
        params[f"{prefix}KEY VELOCITY"] = ((block[13] >> 2) & 0x07) / 7  # 0..7
        params[f"{prefix}OUTPUT LEVEL"] = block[14] / 99
        params[f"{prefix}MODE"] = float(block[15] & 0x01)                # 0 ratio / 1 fixed
        params[f"{prefix}F COARSE"] = ((block[15] >> 1) & 0x1F) / 31     # 0..31
        params[f"{prefix}F FINE"] = block[16] / 99
        params[f"{prefix}SWITCH"] = 1.0  # DX7 voices have no per-op on/off; default on

    for i in range(4):
        params[f"PITCH EG RATE {i + 1}"] = packed_voice[102 + i] / 99
    for i in range(4):
        params[f"PITCH EG LEVEL {i + 1}"] = packed_voice[106 + i] / 99
    params["ALGORITHM"] = packed_voice[110] / 31                         # 0..31
    params["FEEDBACK"] = (packed_voice[111] & 0x07) / 7                  # 0..7
    params["OSC KEY SYNC"] = float((packed_voice[111] >> 3) & 0x01)
    params["LFO SPEED"] = packed_voice[112] / 99
    params["LFO DELAY"] = packed_voice[113] / 99
    params["LFO PM DEPTH"] = packed_voice[114] / 99
    params["LFO AM DEPTH"] = packed_voice[115] / 99
    params["LFO KEY SYNC"] = float(packed_voice[116] & 0x01)
    params["LFO WAVE"] = ((packed_voice[116] >> 1) & 0x07) / 5           # 0..5
    params["P MODE SENS."] = ((packed_voice[116] >> 4) & 0x07) / 7       # 0..7
    params["TRANSPOSE"] = packed_voice[117] / 48                         # 0..48

    return params


def voice_parameters(data: bytes, voice_index: int) -> Dict[str, float]:
    """Validate a cartridge and unpack one voice into normalized parameters."""
    validate_cartridge(data)
    if not 0 <= voice_index < NUM_VOICES:
        raise ValueError(f"voice_index must be in [0, {NUM_VOICES - 1}], got {voice_index}.")
    return unpack_voice(_packed_voice(data, voice_index))
