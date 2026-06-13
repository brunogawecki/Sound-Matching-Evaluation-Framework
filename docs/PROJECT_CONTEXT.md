# Sound Matching Evaluation Framework — Project Context

> **Audience**: This document is a handoff to Claude Code. It captures the thesis context, decisions made so far, the current state of the codebase, the target architecture, and the recommended next steps. Read this before touching any code.

---

## 1. Project overview

### Thesis

**Title**: *Synthesizer Inversion: Reconstructing Synthesizer Patches from Audio Waveforms*
**Institution**: Poznań University of Technology, Faculty of Computing and Telecommunication
**Supervisor**: prof. dr. inż. Ewa Łukasik
**Author**: Bruno Gawęcki

The thesis sits in the field of **automatic synthesizer programming**, focusing on **sound matching**: predicting the parameter settings of a target synthesizer such that its rendered audio matches a target waveform.

A peer-reviewed Systematic Literature Review on this topic was presented at the AES 160th Convention (Copenhagen, May 2026) and is published in the AES E-Library (id 23199). That SLR is the methodological foundation of this framework.

### What this framework is

A **comparative benchmark** of machine learning frameworks for synthesizer sound matching. The deliverable is not a new model — it is a standardised, reproducible evaluation harness that lets multiple model families be compared apples-to-apples on the same synthesizers, same datasets, and same perceptual metrics.

The motivation is directly drawn from the SLR's conclusion: the field is fragmented because every paper uses a different synth, different dataset construction, and a different metric panel, making cross-paper comparison nearly impossible.

### Technical challenges (from the SLR)

1. **Many-to-one mapping (symmetry)**: Multiple parameter configurations can produce perceptually identical audio (e.g., swapping two identical operators). Regression to the mean of valid solutions is a known failure mode.
2. **Non-differentiability of commercial synths**: Compiled VSTs cannot have gradients backpropagated through them. Solutions in the literature: parameter-based losses, neural proxies, gradient-free search, or custom differentiable synths.
3. **Data scarcity**: Human-curated preset datasets are small; synthetic random sampling does not reflect the distribution of musically realistic sounds.
4. **Perceptual metric misalignment**: Mathematical distance metrics on parameters or raw audio do not align with human auditory perception.

---

## 2. Scope and decisions already made

These are **locked**. Do not re-litigate them unless the user explicitly asks.

| Axis | Decision | Rationale |
|---|---|---|
| Synth scope | **Black-box commercial only** as primary scope; differentiable as a stretch goal | Practical relevance to musicians; biggest gap in the literature |
| Target synths | **Two synths of different types**: Dexed (FM, free DX7 emulation) + Surge XT (subtractive/hybrid) | Tests cross-architecture generalisation; both are open-source and DawDreamer-compatible |
| Model family count | **6+ families** | One representative per family from the SLR taxonomy |
| Primary evaluation axis | **Perceptual audio similarity** (not parameter accuracy) | Aligns with the SLR conclusion that perceptual metrics matter more than parameter MSE |
| Dataset strategy | **Hybrid**: large synthetic dataset (random sampling) for training; held-out human preset collection for the test set | Tests the distribution-shift problem head-on |
| Neural proxy ([5] InverSynth II style) | **Baseline only**, not the main contribution | Useful for enabling audio-loss training on the black-box models, but not the research story |

### Model families to implement

These are the six families from the SLR taxonomy. Pick **one strong representative per family**:

1. **Evolutionary / search** — genetic algorithm baseline (canonical reference: Horner et al. 1993 [22], Masuda quality-diversity [30])
2. **Discriminative MLP/CNN** — InverSynth-style CNN on log-STFT spectrograms [6]
3. **Transformer-based discriminative** — Audio Spectrogram Transformer [8] / Sound2Synth [11]
4. **Generative — VAE** — Le Vaillant et al. preset-gen-vae [47]
5. **Generative — Normalising flow / flow matching** — Esling et al. flow synthesizer [16] or Hayes et al. equivariant flow matching [21]
6. **Neural proxy / RL** — InverSynth II [5] or SynthRL [39]

Citations refer to the thesis bibliography (see the master's thesis PDF / AES SLR paper).

---

## 3. Open design decisions

These are **not yet decided**. The user should resolve them before any further significant code is written. Each has a known-good default if no strong preference emerges, but the user should consciously choose.

### D1. Parameter subset

**Question**: Which parameters does each model actually estimate? Lock the rest at defaults.

**Why it matters**: Dexed has 155 parameters; Surge XT has more. **No paper in the SLR estimates the full parameter space.** Sound2Synth [11], Esling et al. [16], and Le Vaillant et al. [47] all work on subsets. With 6 models being benchmarked, they MUST all target the same subset, or the comparison is invalid.

**Recommended default**: For Dexed, the 30-40 most audibly impactful parameters: algorithm, LFO settings, and per-operator output level + frequency ratio + envelope. For Surge XT, a comparably-sized subset of the main oscillator + filter + envelope parameters.

**Action**: Pick the subset; codify it in a Python file (e.g. `synth/dexed_subset.py`) that lists each estimated parameter with its kind, bounds, and default.

### D2. Categorical encoding

**Question**: How are categorical parameters represented in ML model inputs/outputs?

**Why it matters**: The current code (in `synth/dexed_synth.py::get_categorical_mappings`) stores categoricals as evenly-spaced floats in [0,1] — e.g., a 6-option LFO waveform becomes `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`. This is **correct for DawDreamer's input format** but **wrong as an ML training target**: regressing 0.4 vs 0.6 has no meaningful loss interpretation when the classes are unordered (sine vs square vs triangle).

**Recommended default**: ML models predict categoricals via **one-hot vectors with cross-entropy loss** (the approach in [47, 5, 13]). The framework needs a `ParameterSpace` abstraction that converts between two parameter dict formats:
- **Synth-side dict** — all values are floats in [0,1] (what DawDreamer wants)
- **ML-side vector** — continuous params as floats, categoricals as one-hot blocks

### D3. MIDI note and render duration

**Question**: Single fixed note + duration, or varied per sample?

**Why it matters**: Affects dataset size, generation time, and what the models actually learn.

**Recommended default**: Single fixed note (C4 = MIDI 60), velocity 100, **1-2 second duration**. This matches InverSynth [6], Sound2Synth [11], and AST [8]. The current `config.py` value of 4 seconds is unusually long and will quadruple dataset generation time without strong evidence of benefit.

### D4. Human preset source for the test set

**Question**: Where do the human-curated test presets come from, and how are they imported into DawDreamer's normalised-float parameter format?

**Why it matters**: The hybrid dataset strategy depends on this. Synthetic-only evaluation does not test generalisation to musically realistic sounds.

**Known sources**:
- **Dexed**: ~30k DX7 cartridge SysEx patches circulate online; the Dexed plugin ships with a starter set
- **Surge XT**: factory presets ship as `.fxp` files (~2000 patches)

The conversion from SysEx/`.fxp` to DawDreamer normalised floats is **non-trivial** and should be budgeted as a small subproject of its own.

---

## 4. Current codebase

### Tree

```
sound_matching_evaluation_framework/
├── config.py                  # Env-var-driven paths + audio defaults
├── requirements.txt           # numpy, scipy, dawdreamer, pandas, tqdm, python-dotenv
├── GEMINI.md                  # Original project notes (predecessor to this document)
├── .env                       # Local VST paths (not committed)
├── scripts/
│   ├── verify_dexed.py        # Smoke test: renders one random Dexed patch to WAV
│   └── render_preset.py       # Render one voice of a DX7 .syx cartridge to WAV
└── synth/
    ├── __init__.py
    ├── base_synth.py          # BaseSynthesizer abstract class
    └── dexed_synth.py         # DexedWrapper (DawDreamer)
```

### What works

- `BaseSynthesizer` defines a clean contract: `set_parameters`, `get_parameters`, `render_audio`, `get_parameter_bounds`, `get_categorical_mappings`, and a default `randomize_parameters` in the base class.
- `DexedWrapper` is a working concrete implementation using DawDreamer; `verify_dexed.py` exercises the full path (init → randomise → set → render → save WAV).
- `config.py` reads VST paths and audio defaults from `.env`, keeping the codebase portable across machines.
- The separation of `get_parameter_bounds()` (continuous) from `get_categorical_mappings()` (discrete) is the right architectural call — most naïve implementations conflate these and silently produce broken models.

### What is incomplete or known to be wrong

- **No Surge XT wrapper** yet (the path is in `config.py` but no `SurgeXTWrapper` class exists).
- **No `ParameterSpace` abstraction** — parameter metadata is locked inside the synth wrapper. Every ML model would currently have to reach into the synth to figure out which params are continuous vs categorical, which is wrong layering.
- **No dataset abstraction** — `dataset/audio/` exists but there is no `DatasetBuilder` or `SoundMatchingDataset` class.
- **No model abstraction** (`BaseModel`) and no model implementations.
- **No evaluator / metric panel**.
- **Categorical encoding is synth-friendly but ML-hostile** — see D2 above.
- **Render duration is 4 seconds** — likely too long (see D3).

### Code-level observations to validate during review

These are findings from a prior architectural review. Verify them and report on each:

1. `DexedWrapper.set_parameters` calls `int(param_name)` on every set, every render. For batched dataset generation this adds up. Cache the int conversions in `__init__`.
2. `DexedWrapper.get_parameters` reads all 155 parameters; once a subset is chosen (D1), only read the subset.
3. DawDreamer's render engine state (LFO phase, reverb tails) **can leak between renders**. Verify reproducibility with: "render same params twice, assert bit-identical audio." If it fails, the wrapper needs to reset state explicitly between renders.
4. `render_audio` does stereo-to-mono via simple averaging. Confirm this is acceptable for the targeted metrics, or document why.
5. The `randomize_parameters` default in `BaseSynthesizer` uses `np.random.uniform` and `np.random.choice` directly — no seedable `Generator`. For reproducible dataset generation, this should accept an `rng: np.random.Generator`.

---

## 5. Target architecture

The framework has four layers. The **Synth** layer is mostly built; the other three are todo.

```
┌──────────────────────────────────────────────────────┐
│  Layer 1 — Synth     [BUILT]                         │
│  BaseSynthesizer · DexedWrapper · SurgeXTWrapper(td) │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│  Layer 2 — Data      [TODO]                          │
│  ParameterSpace · DatasetBuilder · SoundMatchDataset     │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│  Layer 3 — Models    [TODO]                          │
│  BaseModel · GA · CNN · AST · VAE · Flow · Proxy     │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│  Layer 4 — Evaluation [TODO]                         │
│  BaseMetric · Evaluator · perceptual metric panel    │
└──────────────────────────────────────────────────────┘
```

The Evaluator re-renders predicted parameters through the **same** `BaseSynthesizer` instance used at dataset-generation time, then computes the metric panel on (target_audio, rendered_predicted_audio). The synth is therefore shared across the data-generation path and the evaluation path — this is a contract, not an optimisation.

### Layer 2 — Data (next to build)

**`ParameterSpace`** is the keystone abstraction. Everything depends on it. Build this first. Sketch:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ParameterSpecification:
    name: str                                              # e.g. "op1_output_level"
    kind: Literal["continuous", "categorical", "binary"]
    bounds: tuple[float, float] | None = None              # for continuous
    options: list[float] | None = None                     # for categorical (normalised values)
    default: float = 0.5
    label: str = ""                                        # human-readable
    synth_index: int | None = None                         # plugin-specific index (e.g. Dexed param idx)

class ParameterSpace:
    """Canonical, ordered parameter space for a chosen synth subset."""
    def __init__(self, parameter_specs: list[ParameterSpecification]): ...

    @property
    def dim_ml(self) -> int: ...           # one-hot expanded dim, ML-side
    @property
    def dim_synth(self) -> int: ...        # flat dim, synth-side

    # Two-way conversion
    def synth_dict_to_ml_vector(self, d: dict[str, float]) -> np.ndarray: ...
    def ml_vector_to_synth_dict(self, v: np.ndarray) -> dict[str, float]: ...

    # For loss/metric routing: which slices are continuous vs categorical
    @property
    def loss_slices(self) -> list[tuple[slice, str]]: ...

    # For dataset generation
    def sample_uniform(self, rng: np.random.Generator) -> dict[str, float]: ...
```

`BaseSynthesizer` should then expose a `parameter_space: ParameterSpace` property, and `dexed_subset.py` / `surge_subset.py` files define the actual parameter_specs (D1 made concrete).

**`DatasetBuilder`** — parallelises synthetic dataset generation. With 100k samples, parallelism matters. DawDreamer is not thread-safe within a process but works with `ProcessPoolExecutor` — one engine per worker. Use a parameter hash as the WAV filename for deduplication and reproducibility.

**`SoundMatchingDataset`** — a PyTorch `Dataset` over (audio_path, ml_target_vector) pairs, loading WAVs lazily. Audio loading and conversion to mel/STFT should be the model's job, not the dataset's — different models want different representations.

### Layer 3 — Models

**`BaseModel`** sketch:

```python
class BaseModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, train: SoundMatchingDataset, val: SoundMatchingDataset, cfg: dict) -> None: ...

    @abc.abstractmethod
    def predict(self, audio: np.ndarray) -> dict[str, float]:
        """Return a synth-side param dict (compatible with synth.set_parameters)."""

    @abc.abstractmethod
    def save(self, path: Path) -> None: ...
    @abc.abstractmethod
    def load(self, path: Path) -> None: ...
```

Critical: `predict` returns a **synth-side dict**, not an ML vector. The model internally converts via `ParameterSpace.ml_vector_to_synth_dict`. This means `predict` outputs are always directly usable by `synth.set_parameters` — no glue code at evaluation time.

For the GA baseline, `fit` is a no-op and `predict` runs the GA loop. For deep models, `fit` is the training loop.

### Layer 4 — Evaluation

**Metric panel** — at minimum, all of:

- **Parameter-side**: MAE on continuous params, accuracy on categoricals (for diagnostic; not the primary)
- **Audio-side, spectral**: log-spectral distance (LSD), spectral convergence (SC), multi-scale STFT MAE
- **Audio-side, perceptual proxy**: MFCC MAE, optionally a deep embedding distance (CLAP or Wav2CLIP)
- **Optional**: human listening study for the final results table

The Evaluator iterates over a test dataset, predicts params for each target, re-renders via the synth, and computes every metric. Output: a results table (pandas DataFrame → CSV/Markdown).

A reference panel that matches recent literature (see SLR Table 1): MAE, Mel MAE, SC, MFCC MAE, MSS, and either CLAP or human eval as the perceptual top-line.

---

## 6. Conventions and constraints

These are inherited from `GEMINI.md` and extended.

1. **Strict abstraction**: every new synth inherits from `BaseSynthesizer`; the underlying engine (DawDreamer, Pedalboard, etc.) is hidden from the rest of the codebase.
2. **Type hints**: always. Use `Dict[str, Union[float, int]]`, `np.ndarray`, etc.
3. **Parameter normalisation**: the synth wrapper handles the translation between ML-friendly and DawDreamer-normalised formats. ML code should never see DawDreamer's [0,1] floats for categoricals.
4. **Audio format**: `render_audio` returns 1-D mono `np.ndarray`. Torch tensor conversion happens in the dataset/model layer, not the synth.
5. **No hard-coded paths**: VST paths and machine-specific settings go in `.env`, read via `config.py`.
6. **Reproducibility**: every randomised operation (sampling, model init, train/val split) takes a `seed` or `np.random.Generator`. The framework must produce bit-identical results across runs with the same seed.
7. **Dataset storage**: WAV files + a pandas DataFrame (parquet preferred over CSV for large datasets) of parameter vectors and metadata (filename, MIDI note, velocity, render duration, synth name, generation method).

---

## 7. Recommended next-step sequence

In strict order. Don't skip ahead.

1. **Resolve the open design decisions** (D1–D4). Document the choices in a new file `DECISIONS.md` (or extend this file). Without these locked, every layer downstream will need rework.
2. **Build `ParameterSpace` + `dexed_subset.py`**. Add a unit test that round-trips: `synth_dict → ml_vector → synth_dict` returns the original. Add a second test: `synth.set_parameters(synth_dict)` followed by `synth.get_parameters()` returns the same dict.
3. **Build `SurgeXTWrapper`** *next*, in parallel with picking its parameter subset. Doing the second synth implementation early stress-tests whether `BaseSynthesizer` actually generalises. Any contract leaks surface now, before any model code depends on them.
4. **Build `DatasetBuilder`** with `ProcessPoolExecutor` parallelism. Verify it can produce, say, 10k Dexed samples without running out of memory or hitting DawDreamer crashes.
5. **Build `SoundMatchingDataset`** as a thin PyTorch wrapper.
6. **Build `BaseModel`** + one trivial model (e.g., a "predict the dataset mean" baseline) to exercise the full train → predict → re-render → metric path.
7. **Build the metric panel** with at least LSD, SC, MFCC MAE, parameter MAE.
8. **Then** start implementing real model families, one at a time, beginning with whichever the user wants to prioritise (typical pragmatic order: GA → CNN → AST → VAE → Flow → Proxy).

---

## 8. Tasks for Claude Code

This document is a handoff. The two tasks for this session:

### Task A — Code review of the existing codebase

Walk through every file in the repo and report on:
- Whether each abstraction matches the contract described in this document
- The specific code-level observations listed in §4 (validate each one — confirm or refute)
- Any code smells, type-hint gaps, missing error handling, or dawdreamer-specific footguns
- Whether `verify_dexed.py` actually runs cleanly on the user's machine, and if there are any environment / dependency issues
- Whether the existing categorical encoding (D2) needs to change *now* or can wait until `ParameterSpace` is built

Output: a structured review, file-by-file, with concrete actionable findings (not vague observations).

### Task B — Future plan

Based on the review and on §3 (open decisions) and §7 (recommended sequence), produce:

- A prioritised task list for the next 4–6 weeks of development
- For each task: scope, acceptance criteria, dependencies, and rough effort estimate (small / medium / large)
- Concrete code skeletons for the next abstraction to build (`ParameterSpace`), aligned with the user's existing code style
- A list of questions the user must answer to unblock further work (mapped to D1–D4 and anything new the review uncovers)

The plan should be actionable enough that a future session can be opened with "continue from step N" and the work proceeds without re-discussion of context.

---

## 9. Reference material

- **The thesis manuscript** (master's thesis PDF) — full SLR with chapter references and bibliography
- **The AES SLR paper** (AES E-Library id 23199, 160th Convention 2026) — peer-reviewed condensed version
- **DawDreamer** — Braun, *Bridging the Gap Between DAWs and Python Interfaces*, arXiv:2111.09931
- **Dexed** — open-source DX7 emulation: https://asb2m10.github.io/dexed/
- **Surge XT** — open-source synth: https://surge-synthesizer.github.io/
- **Key SLR papers per family** (citation numbers from the master's thesis bibliography):
  - GA: Horner et al. [22], Masuda quality-diversity [30]
  - CNN: InverSynth [6]
  - Transformer: Bruford et al. AST [8], Sound2Synth [11]
  - VAE: Le Vaillant et al. [47]
  - Flow / Flow matching: Esling et al. [16], Hayes et al. [21]
  - Proxy / RL: InverSynth II [5], SynthRL [39]

---

*End of context document. When ready, begin Task A.*
