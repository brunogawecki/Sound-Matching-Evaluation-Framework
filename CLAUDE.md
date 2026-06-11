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

## Architecture & Code Structure
The framework is built with modularity and clear abstractions in mind.

*   `synth/`: Contains the synthesizer wrapper modules.
    *   `base_synth.py`: Defines the `BaseSynthesizer` abstract class. **All new synthesizer plugins must implement this interface.** It enforces methods for getting/setting parameters, rendering mono audio, and defining parameter bounds/categorical mappings.
    *   `dexed_synth.py`: A concrete implementation of the Dexed (DX7) synthesizer using `dawdreamer`.
*   `config.py`: The centralized configuration module. It loads environment variables (from `.env`) to manage local VST paths, audio settings, and export directories.
*   `.env` (not committed): Stores local, machine-specific paths to VST plugins (`.vst3` or `.component`) and rendering defaults.
*   `dataset/`: The target directory for generated audio (`.wav`) and metadata (`.csv`).
*   `paper_repos/`: Contains legacy/reference code from related papers (e.g., `InverSynth2`, `preset-gen-vae`).

## Development Conventions & Guidelines
When contributing to or expanding this framework, please adhere to the following rules:

1.  **Strict Abstraction:** Any new synthesizer added to the framework must inherit from `BaseSynthesizer` and implement all abstract methods. The underlying engine (e.g., DawDreamer, Pedalboard) should be completely hidden from the rest of the application.
2.  **Type Hinting:** Always use Python type hints (e.g., `Dict[str, Union[float, int]]`, `np.ndarray`) for function signatures and returns.
3.  **Parameter Normalization:** The `BaseSynthesizer` contract expects subclasses to handle the translation between "real-world" VST parameters and normalized values. For example, DawDreamer expects parameters normalized between `[0.0, 1.0]`.
4.  **Audio Format:** The `render_audio` method must always return a **1D (mono) numpy array**.
5.  **Configuration over Hardcoding:** **Never hardcode paths to local VST plugins or directories.** Always define them as environment variables in `.env` and load them via `config.py`.
6.  **Data Generation Strategy (Hybrid):** For initial data generation and debugging, use a hybrid approach: store parameter metadata in a Pandas DataFrame (exported to CSV) and save the corresponding rendered audio as individual WAV files in the `dataset/audio/` directory.

## Building and Running

### Setup
1.  Ensure you have a working Python virtual environment (e.g., `venv/`).
2.  Install dependencies: `pip install -r requirements.txt`
3.  Create a `.env` file at the root of the project (copy settings from `config.py` defaults if needed) and update the `DEXED_PATH` (and others) to point to your local VST3/Component plugins.

### Verification
To test the audio rendering pipeline and ensure your local VST is correctly linked:
```bash
python verify_dexed.py
```
This script will initialize the Dexed wrapper, randomize its parameters, render a 2-second audio file, and save it to `dataset/audio/dexed_verification.wav`.
