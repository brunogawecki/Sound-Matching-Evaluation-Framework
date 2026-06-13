"""Quickly render one voice of a DX7 .syx cartridge to a WAV file.

Examples:
    python scripts/render_preset.py --list          # show the 32 voice names
    python scripts/render_preset.py 8               # render voice 8 -> dataset/audio/preset.wav
    python scripts/render_preset.py 8 --note 48 --out laurie.wav
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, synth) import when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from synth.dexed import DexedWrapper
from synth.dexed.cartridge import voice_names


@contextlib.contextmanager
def suppressed_stderr():
    """Silence the benign JUCE 'invalid URI' notice the VST3 host writes to the
    OS stderr file descriptor while Dexed loads. Only fd 2 is redirected, so real
    Python exceptions and tracebacks are unaffected."""
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)

# Default cartridge: the standard Dexed factory install location. Override with --syx.
DEFAULT_SYX = os.path.expanduser(
    "~/Library/Application Support/DigitalSuburban/Dexed/Cartridges/Dexed_01.syx"
)
# 44100 matches a typical DAW export, for a fair A/B against Ableton.
DEFAULT_SAMPLE_RATE = 44100


def resolve_voice(token: str, names: list) -> int:
    """Resolve a voice index from either a numeric index or a (partial) name.

    A plain integer is treated as the index. Otherwise the token is matched
    case-insensitively against the voice names: an exact match wins; failing
    that, a unique substring match is used.

    Raises:
        ValueError: If the index is out of range, the name matches nothing, or
            a substring matches more than one voice.
    """
    if token.lstrip("-").isdigit():
        index = int(token)
        if not 0 <= index < len(names):
            raise ValueError(f"voice index {index} out of range [0, {len(names) - 1}].")
        return index

    needle = token.strip().casefold()
    exact = [i for i, name in enumerate(names) if name.strip().casefold() == needle]
    if len(exact) == 1:
        return exact[0]

    matches = [i for i, name in enumerate(names) if needle in name.casefold()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"no voice matching {token!r}.")
    listed = ", ".join(f"{i} ({names[i].strip()})" for i in matches)
    raise ValueError(f"{token!r} is ambiguous; matches: {listed}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("voice", nargs="?", help="Voice to render: index (0-31) or (partial) name.")
    parser.add_argument("--syx", default=DEFAULT_SYX, help="Path to the .syx cartridge.")
    parser.add_argument("--list", action="store_true", help="List the 32 voice names and exit.")
    parser.add_argument("--note", type=int, default=config.MIDI_NOTE, help="MIDI note (default 60).")
    parser.add_argument("--velocity", type=int, default=config.VELOCITY, help="MIDI velocity (default 100).")
    parser.add_argument("--duration", type=float, default=config.DURATION_SEC, help="Total seconds.")
    parser.add_argument("--note-duration", type=float, default=config.NOTE_DURATION_SEC, help="Note-on to note-off seconds.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Render sample rate.")
    parser.add_argument("--out", default=None, help="Output WAV path (default dataset/audio/preset.wav).")
    args = parser.parse_args()

    syx_path = os.path.expanduser(args.syx)
    if not os.path.exists(syx_path):
        print(f"Cartridge not found: {syx_path}  (use --syx to point elsewhere)")
        sys.exit(1)

    with open(syx_path, "rb") as syx_file:
        cartridge_bytes = syx_file.read()
    try:
        names = voice_names(cartridge_bytes)
    except ValueError as error:
        print(f"Not a usable cartridge: {error}")
        sys.exit(1)

    if args.list:
        for index, name in enumerate(names):
            print(f"{index:2d}: {name.strip()}")
        return

    if args.voice is None:
        parser.error("voice is required (or use --list). Run with -h for help.")

    try:
        voice_index = resolve_voice(args.voice, names)
    except ValueError as error:
        print(f"Could not select voice: {error}")
        sys.exit(1)

    plugin_path = os.path.expanduser(config.DEXED_PATH)
    if not os.path.exists(plugin_path):
        print(f"Dexed plugin not found at {plugin_path}. Update DEXED_PATH in .env.")
        sys.exit(1)

    with suppressed_stderr():
        synth = DexedWrapper(plugin_path, sample_rate=args.sample_rate, buffer_size=config.BUFFER_SIZE)
    audio = synth.render_cartridge_voice(
        syx_path,
        voice_index=voice_index,
        midi_note=args.note,
        velocity=args.velocity,
        duration_sec=args.duration,
        note_duration_sec=args.note_duration,
    )

    if args.out is not None:
        output_path = args.out
    else:
        name = names[voice_index].strip().replace(" ", "_").replace("/", "-")
        output_path = str(config.AUDIO_OUT_DIR / f"preset_{voice_index:02d}_{name}.wav")

    # Raw float32 so the level matches the DAW for a fair comparison.
    wavfile.write(output_path, args.sample_rate, audio.astype(np.float32))
    print(f"voice {voice_index} '{names[voice_index].strip()}' "
          f"RMS={np.sqrt(np.mean(audio ** 2)):.4f}  ->  {output_path}")


if __name__ == "__main__":
    main()
