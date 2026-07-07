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

## Work tracking — GitHub Issues vs `DECISIONS.md`
Build work is tracked as **GitHub Issues** on Project board
`https://github.com/users/brunogawecki/projects/2` (auto-add workflow surfaces every open issue).
The split is one-way and must be kept clean:
*   **`docs/DECISIONS.md` owns the *why*** — rationale, alternatives, locked/open status. Never copy
    this into an issue.
*   **Issues own the *do*** — concrete, PR-sized work; close via `fixes #N` commits. An issue that
    depends on an open decision **points** at it (e.g. "blocked by D1 — see `docs/DECISIONS.md`")
    rather than duplicating its content.
*   **Deciding an open decision is not a GitHub task** — it resolves in `docs/DECISIONS.md`; only the
    *work it unblocks* becomes issues.
*   Labels: `build` / `bug` (type), `layer-2-data` / `layer-3-models` / `layer-4-eval` (area).
    Milestones = Phases (`Phase 2`, `Phase 3`).

## Architecture & Code Structure
The framework is built with modularity and clear abstractions in mind.

*   `synth/`: Contains the synthesizer wrapper modules.
    *   `base_synth.py`: Defines the `BaseSynthesizer` abstract class. **All new synthesizer plugins must implement this interface.** It enforces methods for getting/setting parameters, rendering mono audio, defining parameter bounds/categorical mappings, and exposing a `parameter_space`.
    *   `dexed/synth.py`: A concrete implementation of the Dexed (DX7) synthesizer. Addresses all parameters **by name** (builds a name→index map from the live plugin), excludes the VST-level extras and MIDI-CC passthroughs, and caches the freshly-loaded patch as the parameter defaults. All logic is engine-agnostic and delegates rendering to a pluggable `Renderer` (`DexedWrapper(renderer="dawdreamer" | "pedalboard")`).
    *   `renderers/`: The `Renderer` abstract class (`base.py`) and its implementations (`DawDreamerRenderer` default, `PedalboardRenderer` secondary). This is the only engine-specific layer; selecting a renderer is a per-run choice and renderers are never mixed within a run (D-RENDERER / D-REPRO — see `docs/DECISIONS.md`).
    *   `parameter_space.py`: The `ParameterSpecification` / `ParameterSpace` abstraction (Layer 2). Owns the ordered parameter subset and the two-way conversion between the **synth-side dict** (names → normalized floats) and the **ML-side vector** (continuous floats + one-hot categorical blocks), plus `loss_slices` for routing losses/metrics and `sample_uniform`.
    *   `dexed/subset.py`: The **final** Dexed parameter subset (D1 LOCKED — see `docs/DECISIONS.md`): the 103 estimated parameters. Builds a `ParameterSpace` from the live wrapper.
*   `config.py`: The centralized configuration module. It loads environment variables (from `.env`) to manage local VST paths, audio settings, and export directories.
*   `.env` (not committed): Stores local, machine-specific paths to VST plugins (`.vst3` or `.component`) and rendering defaults.
*   `docs/`: Decision records and glossary (`DECISIONS.md`, `CONTEXT.md`) — the source of truth for design intent.
*   `tests/`: `pytest` suite. Plugin-dependent tests skip automatically when the VST is absent.
*   `dataset/`: The Layer 2 data package. `builder.py` (`DatasetBuilder`) renders a `PresetSource` (`preset_sources.py`: synthetic / human / hybrid) into a self-describing corpus; `dexed_preset_loader.py` loads + dedups + splits DX7 cartridge voices; `render_backends.py` holds `RenderSettings` + `FreshProcessRenderBackend` (fresh-process-per-render at pos 0 — D-REPRO); `torch_dataset.py` (`RenderedCorpusDataset`) consumes a corpus as `(audio, target)` pairs and rebuilds its `ParameterSpace` from `run_summary.json` with no live VST (D-SELFDESC). Generated corpora live under `dataset/<run_name>/` (gitignored).
*   `models/`: The Layer 3 model package. `base_model.py` defines the `BaseModel` ABC (`fit` / `predict` / `save` / `load`); `base_deep_model.py` (`BaseDeepModel`) adds the shared, Lightning-free `save`/`load`/`predict` for deep families; `mean_parameter_baseline.py` (`MeanParameterBaseline`) is the naive train-set-mean floor (issue #7); `sound2synth.py` (`Sound2SynthSpectrogramRegressor`, issue #19/#31) is the **first real deep family** — a VGG11-BN net over a log-power STFT emitting the ML-side vector through `ParameterSpace`. It is a deliberately *basic* first cut (single spectrogram branch + plain MLP head), **not** the paper's full multi-modal encoder / grouped-FC classifier, which remains future work. `registry.py` exposes `MODEL_REGISTRY` — name → (model class, default checkpoint filename) — the single source of truth `scripts/fit_model.py --model` and `scripts/evaluate.py --model` both read; a new family registers once and is trainable and evaluable everywhere. `models/training/` holds the PyTorch-Lightning training harness (issue #28, imported lazily so the eval path stays training-dependency-free — D-FRAMEWORK).
*   `evaluation/`: The Layer 4 evaluation package. `registry.py` exposes `METRIC_PANEL` — 13 per-sample `MetricSpecification`s across the parameter / magnitude / timbre / loudness / pitch axes (callables in `metrics/`; the embedding/perceptual axis is deferred — D-METRIC-PERCEPTUAL, issue #8). `evaluator.py` (`Evaluator`, issue #9) scores a fitted model on a corpus: per sample it predicts, re-renders the prediction fresh-process at pos 0, runs the panel, and writes `results/<corpus>/<model>/{per_sample.csv, eval_summary.json}` (render contract read from the corpus, never `config.py` — D-EVAL). Outputs under `results/` and `checkpoints/` are gitignored.
*   `paper_repos/`: Contains legacy/reference code from related papers (e.g., `InverSynth2`, `preset-gen-vae`).
*   `dashboard/`: A private, localhost **Streamlit** control panel over the pipeline (issue #12). Four pages — Build dataset → Fit model → Evaluate → Results — that **subprocess** the `scripts/*.py` commands and read their output files back for display; it never imports the pipeline library, so it cannot drift from the CLI. `script_specs.py` holds declarative `ArgSpec`/`ScriptSpec` tables mirroring each script's `argparse`; `forms.py` turns a spec into widgets + argv (`build_command` is the pure, tested core); `command_runner.py` streams stdout live (honoring `\r` for one-line `tqdm`); `discovery.py` scans `config.py` paths for dropdown options. The D1 subset is shown read-only. Run with `streamlit run dashboard/app.py`. See `docs/ARCHITECTURE.md`.

## Development Conventions & Guidelines
When contributing to or expanding this framework, please adhere to the following rules:

1.  **Strict Abstraction:** Any new synthesizer added to the framework must inherit from `BaseSynthesizer` and implement all abstract methods. The underlying engine (e.g., DawDreamer, Pedalboard) should be completely hidden from the rest of the application.
2.  **Name-based addressing (D-NAMING):** Above the wrapper, parameters are referred to **only by their plugin-reported name** (e.g. `'ALGORITHM'`, `'OP1 OUTPUT LEVEL'`), never by numeric index. Each wrapper resolves names to indices internally. This is a LOCKED decision (see `docs/DECISIONS.md`).
3.  **Type Hinting:** Always use Python type hints (e.g., `Dict[str, Union[float, int]]`, `np.ndarray`) for function signatures and returns.
4.  **Descriptive Names (no abbreviations):** Spell identifiers out in full — `ml_dimension`, not `dim_ml`; `synth_dimension`, not `dim_synth`. Do not invent shortened variable, function, or attribute names. The only permitted abbreviations are established, universally-recognized conventions already used in the ecosystem (`np` for numpy, `pd` for pandas, `df`/`df_*` for a pandas DataFrame, `rng` for a `np.random.Generator`).
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
