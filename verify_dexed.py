import os
import sys
import numpy as np
from scipy.io import wavfile
from synth.dexed import DexedWrapper
import config

# A random patch can legitimately be near-silent (e.g. all carriers at low output),
# so try a few seeds before concluding something is wrong.
MAX_ATTEMPTS = 5
MIN_AMPLITUDE = 1e-3


def verify_dexed() -> None:
    plugin_path = os.path.expanduser(config.DEXED_PATH)

    if not os.path.exists(plugin_path):
        print(f"Could not find Dexed plugin at: {plugin_path}")
        print("Please update DEXED_PATH in your .env file.")
        sys.exit(1)

    print(f"--- Verifying DexedWrapper with {plugin_path} ---")

    # 1. Initialize Wrapper using config defaults
    synth = DexedWrapper(
        plugin_path=plugin_path,
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE
    )
    print(f"Successfully initialized Dexed at {synth.sample_rate}Hz")
    print(f"Exposed parameters: {len(synth.parameter_names)}")

    for seed in range(MAX_ATTEMPTS):
        # 2. Randomize parameters reproducibly
        random_params = synth.randomize_parameters(np.random.default_rng(seed))

        # 3. Set parameters
        synth.set_parameters(random_params)

        # 4. Render audio using config defaults
        print(f"Rendering {config.DURATION_SEC}s (note held {config.NOTE_DURATION_SEC}s, seed {seed})...")
        audio = synth.render_audio(
            midi_note=config.MIDI_NOTE,
            velocity=config.VELOCITY,
            duration_sec=config.DURATION_SEC,
            note_duration_sec=config.NOTE_DURATION_SEC,
        )

        max_amp = np.max(np.abs(audio))
        if max_amp >= MIN_AMPLITUDE:
            break
        print(f"  Patch was near-silent (max amp {max_amp:.6f}), retrying with next seed...")
    else:
        print(f"All {MAX_ATTEMPTS} random patches rendered near-silent audio.")
        sys.exit(1)

    # 5. Save to the configured audio directory
    output_file = config.AUDIO_OUT_DIR / "dexed_verification.wav"
    wavfile.write(output_file, synth.sample_rate, (audio * 32767).astype(np.int16))

    print("\nSuccess!")
    print(f"Audio shape: {audio.shape}")
    print(f"Max amplitude: {max_amp:.4f}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    verify_dexed()
