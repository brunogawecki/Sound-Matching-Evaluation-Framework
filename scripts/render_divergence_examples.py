"""
Render cross-method-divergent patches through all three benchmark arms so they can be
listened to side by side.

The cross-method divergence (D-REPRO / cross-engine tail) is *context-dependent*: a patch
renders differently depending on what was rendered before it. The three in-process arms
(reuse, reload, pedalboard) therefore replay the full patch sequence up to the selected index
and capture those patches in their original position — rendering them in isolation would not
reproduce the divergence.

The fourth arm, **subprocess**, is the clean reference: each selected patch is rendered at
position 0 of a *fresh OS process* (spawn, one process per patch), so it carries no accumulated
context. This is the context-independent render the in-process arms diverge from — on the
sensitive patches you should hear the reuse/reload/pedalboard captures differ audibly from the
subprocess one, while a second subprocess render would be bit-identical to it (D-REPRO).

Random mode (default):
    Replays the seed-0 random-subset sequence. SELECTED indices were chosen from
    host_agreement_3way_seed0.csv as the most cross-engine-divergent patches.
    Output: dataset/audio/divergence_examples/patch{idx:04d}_{arm}.wav

Cartridge mode (--cartridges):
    Loads real DX7 cartridge voices in the same order as the benchmark, replays the sequence
    to the last selected index, and captures selected patches from
    host_agreement_3way_cartridges.csv (the p90 and p95 range by LSD). Prints the voice name
    next to each file so you know what you're listening to.
    Output: dataset/audio/cartridge_divergence_examples/patch{idx:04d}_{arm}.wav

Output: 16-bit PCM WAVs. Each patch's three renders are normalized by their shared peak so
relative loudness between methods is preserved.
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from synth.dexed import DexedWrapper
from scripts.benchmark_renderers import (
    DEFAULT_CARTRIDGES_DIR,
    build_cartridge_patches,
    build_patches,
    suppressed_stderr,
    _make_wrapper,
    _render,
    _render_patch_in_fresh_process,
)

# Random-mode defaults: most divergent patches from host_agreement_3way_seed0.csv
_RANDOM_SELECTED = [241, 1382, 1697, 2841]
_RANDOM_SEED = 0

# Cartridge-mode defaults: 4 from the p90 range (7-12 dB LSD) and 4 from the p95 range
# (12-25 dB LSD), picked to span sound types visible in the 1056-voice cartridge run.
_CARTRIDGE_SELECTED = [
    # p90 range  (~8-12 dB) — pads, mallet sounds with sensitive LFO tails
    200,   # SynprezFM_06:08 L.A.Pad 4B  (11.82 dB)
    390,   # SynprezFM_12:06 EMULATOR    (11.98 dB)
    781,   # SynprezFM_24:13 SWISSCLOCK  (11.48 dB)
    939,   # SynprezFM_29:11 Pylos Synh  (11.60 dB)
    # p95 range (>12 dB)  — aggressive LFO / sample-and-hold voices
    8,     # Dexed_01:08  SAW EM UP      (19.41 dB)
    66,    # SynprezFM_02:02 SCHLBELL    (22.92 dB)
    131,   # SynprezFM_04:03 S-H ZIBBLE  (23.92 dB)
    593,   # SynprezFM_18:17 COMPUTER 1  (23.53 dB)
]


def render_arm_sequence(arm, patches, selected, max_index):
    """Replay patches[0..max_index] through one arm, capturing the selected indices.

    arm: 'reuse' (persistent dawdreamer), 'reload' (fresh dawdreamer per render),
    'pedalboard' (persistent pedalboard), or 'subprocess' (each selected patch rendered at
    position 0 of its own fresh OS process — the context-independent reference, so it needs no
    sequence replay).
    """
    captured = {}
    if arm == "subprocess":
        selected_sorted = sorted(selected)
        # spawn + maxtasksperchild=1 -> a clean OS heap per patch, the only context that resets
        # the hidden voice state (used as the clean reference, not an engine-level fix).
        # Only the selected patches are rendered; context replay is unnecessary.
        context = mp.get_context("spawn")
        with context.Pool(processes=1, maxtasksperchild=1) as pool:
            results = pool.map(
                _render_patch_in_fresh_process, [patches[i] for i in selected_sorted]
            )
        for index, audio in zip(selected_sorted, results):
            captured[index] = np.asarray(audio)
        return captured
    if arm in ("reuse", "pedalboard"):
        renderer = "dawdreamer" if arm == "reuse" else "pedalboard"
        with suppressed_stderr():
            wrapper = _make_wrapper(renderer)
        for i in range(max_index + 1):
            wrapper.set_parameters(patches[i])
            audio = _render(wrapper)
            if i in selected:
                captured[i] = np.asarray(audio)
    elif arm == "reload":
        wrapper = None
        for i in range(max_index + 1):
            wrapper = None  # drop previous engine before rebuilding (faithful teardown)
            with suppressed_stderr():
                wrapper = _make_wrapper("dawdreamer")
            wrapper.set_parameters(patches[i])
            audio = _render(wrapper)
            if i in selected:
                captured[i] = np.asarray(audio)
    else:
        raise ValueError(arm)
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cartridges", nargs="?", const=DEFAULT_CARTRIDGES_DIR, default=None,
        help="Render real cartridge voices instead of random patches. "
             "Bare --cartridges uses the standard Dexed install location.",
    )
    args = parser.parse_args()

    plugin_path = os.path.expanduser(config.DEXED_PATH)

    if args.cartridges is not None:
        if not os.path.exists(args.cartridges):
            print(f"Cartridge path not found: {args.cartridges}")
            sys.exit(1)
        print(f"Loading cartridge patches from {args.cartridges}")
        patches, labels = build_cartridge_patches(args.cartridges)
        selected_list = [i for i in _CARTRIDGE_SELECTED if i < len(patches)]
        out_dir = Path(config.BASE_DIR) / "dataset" / "audio" / "cartridge_divergence_examples"
    else:
        with suppressed_stderr():
            reference = DexedWrapper(plugin_path=plugin_path, sample_rate=config.SAMPLE_RATE,
                                     buffer_size=config.BUFFER_SIZE)
        max_needed = max(_RANDOM_SELECTED) + 1
        patches = build_patches(reference, max_needed, _RANDOM_SEED)
        del reference
        labels = None
        selected_list = _RANDOM_SELECTED
        out_dir = Path(config.BASE_DIR) / "dataset" / "audio" / "divergence_examples"

    selected = set(selected_list)
    max_index = max(selected_list)

    arms = ("reuse", "reload", "pedalboard", "subprocess")
    captured = {}
    for arm in arms:
        print(f"Rendering arm '{arm}' through patches 0..{max_index} "
              f"(capturing {sorted(selected_list)})...")
        captured[arm] = render_arm_sequence(arm, patches, selected, max_index)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting WAVs to {out_dir}")
    for idx in sorted(selected_list):
        renders = {arm: captured[arm][idx] for arm in arms if idx in captured[arm]}
        if not renders:
            print(f"  patch {idx:4d}: no renders captured (skipped)")
            continue
        label = f"  {labels[idx]}" if labels is not None else ""
        for arm, audio in renders.items():
            peak = float(np.max(np.abs(audio)))
            scale = 0.7 / peak if peak > 0 else 1.0
            pcm = np.clip(audio * scale, -1.0, 1.0)
            pcm = (pcm * 32767.0).astype(np.int16)
            path = out_dir / f"patch{idx:04d}_{arm}.wav"
            wavfile.write(str(path), config.SAMPLE_RATE, pcm)
        peaks_str = ", ".join(
            f"{arm}={float(np.max(np.abs(renders[arm]))):.3f}" for arm in renders
        )
        print(f"  [{idx:4d}]{label}  peak ({peaks_str})")

    print(f"\nListen to e.g. patch{sorted(selected_list)[0]:04d}_reuse.wav "
          f"vs patch{sorted(selected_list)[0]:04d}_pedalboard.wav")


if __name__ == "__main__":
    main()
