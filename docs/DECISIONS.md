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
floats in [0,1]. `ParamSpace` (Layer 2) owns the two-way conversion. Matches
Le Vaillant et al. [47] and InverSynth II [5].

### D3 — Render settings: C4, 4 s render, 3 s note

- MIDI note **60** (C4), velocity **100**, single fixed note per sample
- Render duration **4.0 s**, note-off at **3.0 s** → 1 s of release tail is captured,
  so release-envelope parameters are audible and learnable
- User consciously kept 4 s (doc recommended 1–2 s); revisit only if generation time
  becomes a real bottleneck

### D-REPRO — Render reproducibility contract

`render_audio` **re-applies the current parameter dict before every render**.

Empirical basis (tested 2026-06-11): `set_parameters → render` is bit-identical across
repeated cycles and across fresh engine instances; `render → render` without re-setting
is NOT (max diff ~0.028, engine state leak). A regression test enforces this contract.

### D-ORDER — Dexed-only vertical slice first

Build the full pipeline (wrapper fixes → ParamSpace → DatasetBuilder → PyTorch dataset →
BaseModel + trivial baseline → metric panel) on **Dexed only**, producing a first results
table. The Surge XT wrapper comes after, re-using the proven recipe. Rationale: fastest
end-to-end feedback; avoids a second subset decision while D1 is open.

---

## OPEN

### D1 — Final Dexed parameter subset (deferred by user, 2026-06-11)

Which of the 152 exposed parameters the models estimate; the rest are locked at defaults.
**Blocks**: generation of the real training dataset (Phase 2 output).
**Does not block**: ParamSpace, DatasetBuilder, model/metric code (all subset-agnostic;
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
