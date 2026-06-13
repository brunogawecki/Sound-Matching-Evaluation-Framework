# Sound Matching Evaluation Framework

## Project Overview
This repository contains the codebase for a Master's Thesis focused on a **Sound Matching Evaluation Framework**. The primary objective is to conduct a standardized comparative benchmark of machine learning frameworks (Generative, Discriminative, Evolutionary) to predict synthesizer parameters from target audio waveforms. 

The framework is designed to programmatically generate audio datasets using headless VST synthesizers (currently supporting Dexed/DX7, with plans for Diva, Surge XT, and TAL-NoiseMaker) and subsequently evaluate ML models against these rendered sounds.

## Key Technologies
*   **Python 3.x**
*   **VST Hosting / Rendering:** `dawdreamer`
*   **Data Processing & Science:** `numpy`, `scipy`, `pandas`
*   **Configuration Management:** `python-dotenv`
*   **Progress Tracking:** `tqdm`
*   **Testing:** `pytest`

## Source of Truth for Decisions
Design decisions and terminology live in `docs/`, not in code comments:
*   `docs/DECISIONS.md`: locked and open design decisions (parameter addressing, categorical
    encoding, render contract, etc.). **Do not re-litigate LOCKED decisions; read this before
    making architectural changes.**
*   `docs/CONTEXT.md`: the canonical glossary (synth-side dict, ML-side vector, ParameterSpace,
    render context, etc.).

## Architecture & Code Structure
The framework is built with modularity and clear abstractions in mind.

*   `synth/`: Contains the synthesizer wrapper modules.
    *   `base_synth.py`: Defines the `BaseSynthesizer` abstract class. **All new synthesizer plugins must implement this interface.** It enforces methods for getting/setting parameters, rendering mono audio, defining parameter bounds/categorical mappings, and exposing a `parameter_space`.
    *   `dexed_synth.py`: A concrete implementation of the Dexed (DX7) synthesizer using `dawdreamer`. Addresses all parameters **by name** (builds a name→index map from the live plugin), excludes the VST-level extras and MIDI-CC passthroughs, and caches the freshly-loaded patch as the parameter defaults.
    *   `parameter_space.py`: The `ParameterSpecification` / `ParameterSpace` abstraction (Layer 2). Owns the ordered parameter subset and the two-way conversion between the **synth-side dict** (names → normalized floats) and the **ML-side vector** (continuous floats + one-hot categorical blocks), plus `loss_slices` for routing losses/metrics and `sample_uniform`.
    *   `dexed_subset.py`: The **provisional** Dexed parameter subset used during development (D1 deferred — see `docs/DECISIONS.md`). Builds a `ParameterSpace` from the live wrapper. Not the final dataset subset.
*   `config.py`: The centralized configuration module. It loads environment variables (from `.env`) to manage local VST paths, audio settings, and export directories.
*   `.env` (not committed): Stores local, machine-specific paths to VST plugins (`.vst3` or `.component`) and rendering defaults.
*   `docs/`: Decision records and glossary (`DECISIONS.md`, `CONTEXT.md`) — the source of truth for design intent.
*   `tests/`: `pytest` suite. Plugin-dependent tests skip automatically when the VST is absent.
*   `dataset/`: The target directory for generated audio (`.wav`) and metadata (`.csv`).
*   `paper_repos/`: Contains legacy/reference code from related papers (e.g., `InverSynth2`, `preset-gen-vae`).

## Development Conventions & Guidelines
When contributing to or expanding this framework, please adhere to the following rules:

1.  **Strict Abstraction:** Any new synthesizer added to the framework must inherit from `BaseSynthesizer` and implement all abstract methods. The underlying engine (e.g., DawDreamer, Pedalboard) should be completely hidden from the rest of the application.
2.  **Name-based addressing (D-NAMING):** Above the wrapper, parameters are referred to **only by their plugin-reported name** (e.g. `'ALGORITHM'`, `'OP1 OUTPUT LEVEL'`), never by numeric index. Each wrapper resolves names to indices internally. This is a LOCKED decision (see `docs/DECISIONS.md`).
3.  **Type Hinting:** Always use Python type hints (e.g., `Dict[str, Union[float, int]]`, `np.ndarray`) for function signatures and returns.
4.  **Descriptive Names (no abbreviations):** Spell identifiers out in full — `ml_dimension`, not `dim_ml`; `synth_dimension`, not `dim_synth`. Do not invent shortened variable, function, or attribute names. The only permitted abbreviations are established, universally-recognized conventions already used in the ecosystem (`np` for numpy, `rng` for a `np.random.Generator`).
5.  **Parameter Normalization:** The `BaseSynthesizer` contract expects subclasses to handle the translation between "real-world" VST parameters and normalized values. For example, DawDreamer expects parameters normalized between `[0.0, 1.0]`.
6.  **Audio Format:** The `render_audio` method must always return a **1D (mono) numpy array**.
7.  **Configuration over Hardcoding:** **Never hardcode paths to local VST plugins or directories.** Always define them as environment variables in `.env` and load them via `config.py`.
8.  **Data Generation Strategy (Hybrid):** For initial data generation and debugging, use a hybrid approach: store parameter metadata in a Pandas DataFrame (exported to CSV) and save the corresponding rendered audio as individual WAV files in the `dataset/audio/` directory.

## Building and Running

### Setup
1.  Ensure you have a working Python virtual environment (e.g., `venv/`).
2.  Install dependencies: `pip install -r requirements.txt`
3.  Create a `.env` file at the root of the project (copy settings from `config.py` defaults if needed) and update the `DEXED_PATH` (and others) to point to your local VST3/Component plugins.

### Verification
To test the audio rendering pipeline and ensure your local VST is correctly linked:
```bash
python scripts/verify_dexed.py
```
This script initializes the Dexed wrapper, randomizes its parameters with a seeded RNG, renders
`DURATION_SEC` of audio (note held for `NOTE_DURATION_SEC`, then a release tail) per the render
settings in `config.py`, retries across seeds if a patch is near-silent, and saves the result to
`dataset/audio/dexed_verification.wav`. It exits non-zero on failure.

### Tests
```bash
pytest
```
The suite covers the wrapper (name-based addressing, categorical mappings, render contract) and
the `ParameterSpace` conversions. Tests that require the Dexed VST skip automatically when the plugin
is not found at `DEXED_PATH`.
