# Design Decisions

Locked and open design decisions for the sound matching evaluation framework.
Decisions marked **LOCKED** are settled — do not re-litigate unless the user explicitly asks.
Decisions marked **OPEN** block the work listed under "Blocks".

Last updated: 2026-06-11 (grilling session with Claude Code).

---

## LOCKED

### D-NAMING — Parameters are addressed by name

All public APIs, subset definitions, and dataset metadata refer to synthesizer parameters
by their plugin-reported **name** (e.g. `'ALGORITHM'`, `'OP1 OUTPUT LEVEL'`), never by
numeric index. Each wrapper builds a name→index map from the live plugin
(`get_parameters_description()`) at `__init__` and caches resolved indices internally.

**Why**: the Dexed VST3 build inserts `MonoMode` at index 3, shifting every index by one
relative to the classic Dexed layout the original code assumed. Index-based addressing
caused two critical bugs (see `PROJECT_CONTEXT.md` review follow-up / git history).
Name-based addressing kills this class of bug and is self-documenting. The same
convention applies to the future Surge XT wrapper.

### D-EXCLUDED — VST-level extra parameters are invisible above the wrapper

The Dexed VST3 plugin exposes 2238 parameters. Only the **152 synthesis parameters**
(indices 4–155: `MASTER TUNE ADJ` … `OP6 SWITCH`) are exposed by `DexedWrapper`.
Permanently excluded and locked at plugin defaults:

- `Cutoff`, `Resonance`, `Output`, `MonoMode` (Dexed's VST-level extras, not DX7 synthesis params)
- `Bypass`, `Program` (host/plugin management params — randomizing these mutes output or loads a different patch)
- All 2080 `MIDI CC <ch>|<cc>` JUCE passthrough parameters

### D2 — Categorical encoding: one-hot + cross-entropy (ML-side)

ML models represent categorical parameters (e.g. 32-option `ALGORITHM`, 6-option
`LFO WAVE`) as **one-hot blocks** trained with cross-entropy loss; continuous parameters
as floats with MSE/MAE. The synth-side representation stays DawDreamer's normalized
floats in [0,1]. `ParameterSpace` (Layer 2) owns the two-way conversion. Matches
Le Vaillant et al. [47] and InverSynth II [5].

### D3 — Render settings: C4, 4 s render, 3 s note

- MIDI note **60** (C4), velocity **100**, single fixed note per sample
- Render duration **4.0 s**, note-off at **3.0 s** → 1 s of release tail is captured,
  so release-envelope parameters are audible and learnable
- User consciously kept 4 s (doc recommended 1–2 s); revisit only if generation time
  becomes a real bottleneck

### D-KIND — Parameter kind rule (continuous vs categorical)

A parameter is **categorical** (one-hot + cross-entropy ML-side, grid points synth-side)
when its classes are unordered (`ALGORITHM`, `LFO WAVE`, switches) **or** when it is an
ordered grid whose adjacent steps are perceptually discontinuous. A parameter is
**continuous** when its underlying scale is fine-grained and perceptually smooth
(0–99 levels/rates, `F FINE`).

Consequences (locked 2026-06-11):

- **`OP{i} F COARSE` is categorical with 32 options.** One step can double the operator
  frequency, so a small regression error hides a perceptually massive one; additionally
  Dexed quantizes F COARSE internally to 32 values while reading back the raw float, so
  off-grid synth-side values create an artificial many-to-one (many floats → one sound).
- **No separate "binary" kind** — 2-option parameters are categoricals with cardinality 2
  (uniform one-hot blocks, one loss-routing code path).
- **`ParameterSpecification` carries no plugin index** — the `synth_index` field in the original
  PROJECT_CONTEXT §5 sketch predates D-NAMING and is dropped; index resolution stays
  inside the wrapper.

### D-REPRO — Render reproducibility contract (REVISED 2026-06-11, Phase 1)

**The contract**: rendering the same synth-side dict at the same position of an
identical fresh process is bit-identical. **Context-independence is NOT achievable**:
the same dict rendered after different prior renders can differ *audibly* for sensitive
patches (worst observed: waveform rel. diff 1.4, spectral convergence 1.35, LSD ~9 dB,
concentrated in the note attack and decaying by ~2 s).

Empirical basis (deep investigation, Phase 1 session):

- Dexed keeps hidden engine state that survives — and is not reset by — parameter
  re-application, `load_graph` (prepareToPlay), `load_state`, processor rebuild,
  warm-up notes, or OSC/LFO KEY SYNC settings. Behavior is consistent with
  stale/uninitialized per-voice memory: two fresh instances match only when their
  allocation + render histories are identical (freeing an engine and creating a new
  one reuses dirty memory and diverges).
- The earlier finding that "re-applying parameters before render restores
  bit-identity" was **patch luck** (the tested random patches were insensitive);
  re-application is kept (it is still necessary for parameter correctness) but it
  does NOT guarantee bit-identity.
- The full render *sequence* of a process is deterministic: three identical fresh
  processes produced bit-identical hashes for both the first and second renders.
  Regression test: `test_renders_reproduce_across_identical_fresh_processes`;
  the unachievable context-independent contract is pinned as a strict xfail
  (`test_render_unaffected_by_previous_render_content`) so an upstream fix is noticed.

**Consequences (to honor in Phase 2/3)**:

- Dataset generation must render each sample in a deterministic context
  (fresh worker process with a fixed code path, or a fixed single-process sequence
  re-runnable from the same seed).
- The Evaluator must re-render predictions in the *same kind* of fresh context used
  for target generation, otherwise a perfect parameter prediction would not reproduce
  the target audio (error floor up to SC ≈ 1.35 on sensitive patches — would dominate
  the benchmark).

### D-ORDER — Dexed-only vertical slice first

Build the full pipeline (wrapper fixes → ParameterSpace → DatasetBuilder → PyTorch dataset →
BaseModel + trivial baseline → metric panel) on **Dexed only**, producing a first results
table. The Surge XT wrapper comes after, re-using the proven recipe. Rationale: fastest
end-to-end feedback; avoids a second subset decision while D1 is open.

---

## OPEN

### D1 — Final Dexed parameter subset (deferred by user, 2026-06-11)

Which of the 152 exposed parameters the models estimate; the rest are locked at defaults.
**Blocks**: generation of the real training dataset (Phase 2 output).
**Does not block**: ParameterSpace, DatasetBuilder, model/metric code (all subset-agnostic;
development uses a provisional subset in `synth/dexed_subset.py`, clearly marked).
**Recommendation on file**: ~35 params — `ALGORITHM`, `FEEDBACK`, key LFO params, and
per-operator `OUTPUT LEVEL` + `F COARSE` + reduced envelope (attack + release rates).

### D4 — Human preset source for the test set (deferred by user, 2026-06-11)

Where the human-curated Dexed test presets come from and when the importer is built.
**Blocks**: evaluation on musically realistic presets (the distribution-shift story).
**Recommendation on file**: DX7 SysEx cartridge collections (~30k patches, documented
128-byte packed format); build the SysEx→param-dict importer after Layers 2–4 exist.

### D-METRIC-SR — Sample rate vs. deep-embedding metrics (decide at Phase 3)

Datasets render at 22 050 Hz; CLAP-style embedding metrics expect 48 kHz input.
Decide at metric-panel time: resample for the embedding metric, or render at a higher rate.
