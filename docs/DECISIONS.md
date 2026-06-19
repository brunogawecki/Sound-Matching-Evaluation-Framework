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

- **In-process engine teardown is not enough — only a fresh OS process isolates the state.**
  Freeing the engine and rebuilding it in the *same* process reuses dirty heap memory and
  re-diverges. The 2026-06-17 reload-per-render benchmark (under D-RENDERER) demonstrated this
  interventionally: in-process reload-per-render does not collapse the divergent tail — it
  produces a *third, equally-divergent* realization rather than converging on
  context-independence. Genuine isolation requires a **fresh OS process** (clean heap). A fresh
  process is deterministic: the same patch rendered at the same sequence position of two
  independent fresh processes is bit-identical (the cross-process hash check in this study).
- Dataset generation must therefore render in **fresh worker processes** — each a clean OS
  process, e.g. `multiprocessing` with the **spawn** start method, never **fork** (fork copies
  the parent's already-dirty heap and defeats the isolation). Each worker renders its assigned
  patches deterministically; a fixed single-process sequence re-runnable from the same seed is
  the reproducible fallback.
- The Evaluator must re-render predictions in the **same kind of fresh process** used for
  target generation, at the same sequence position, otherwise a perfect parameter prediction
  would not reproduce the target audio (error floor up to SC ≈ 1.35 / LSD ≈ 9 dB on sensitive
  patches — would dominate the benchmark). Simplest honest contract: generate each target at
  position 0 of a fresh process and re-render each prediction the same way, so target and
  re-render share an identical clean context. `scripts/benchmark_renderers.py --subprocess`
  quantifies the collapse (a fresh-process arm whose two independent realizations agree to ~0
  where the in-process arms keep a full tail).

**Policy — accept and document, do not engine-fix (2026-06-17, user decision)**:

The hidden voice state is treated as a **characterized limitation of the Dexed engine, reported
in the thesis as a threat to validity — not fixed at the engine level** (no Dexed C++ fork, no
attempt to zero-initialize the per-voice memory). Rationale:

- It **does not bias the between-framework comparison** — the core thesis result — as long as
  evaluation is rendered consistently: the leak adds an *equal* noise floor to every model, so
  model *ranking* is unaffected. The only real hazard is an inconsistent generation-vs-evaluation
  render context, which the render discipline in **Consequences** above neutralizes (deterministic
  generation; fresh-process re-render at evaluation).
- The leak is concentrated in **LFO / sample-&-hold / noise** voices (see the cartridge entry
  under D-RENDERER). **D1** may additionally choose to lock those parameters in the final subset,
  which both shrinks the leak's footprint and is a defensible scope decision — the same move
  preset-gen-vae made with its `prevent_SH_LFO` constraint.

The thesis should therefore (a) describe the phenomenon and its mechanism, (b) state the render
discipline used to keep it from biasing results, and (c) cite the characterization data
(`figures/data/context_leakage_seed0.csv`, the D-RENDERER benchmark entries).

**Follow-up (resolved 2026-06-17)**: leakage was initially measured *within DawDreamer only*.
`scripts/measure_context_leakage.py --renderer pedalboard` confirmed **Pedalboard leaks at the same
magnitude** (within-engine p90 7.08 / p95 8.51 dB vs DawDreamer 6.88 / 8.52; ρ = 0.62, 89% top-decile
overlap with the cross-engine tail) — so the hidden state is in the **shared Dexed plugin binary, not
the host**, and switching renderers does not avoid it. See the D-RENDERER "Pedalboard leakage test"
entry.

### D-ORDER — Dexed-only vertical slice first

Build the full pipeline (wrapper fixes → ParameterSpace → DatasetBuilder → PyTorch dataset →
BaseModel + trivial baseline → metric panel) on **Dexed only**, producing a first results
table. The Surge XT wrapper comes after, re-using the proven recipe. Rationale: fastest
end-to-end feedback; avoids a second subset decision while D1 is open.

### D-RENDERER — Rendering library is pluggable; DawDreamer is the default

The VST-hosting engine sits behind a `Renderer` interface (`synth/renderers/base.py`)
beneath the synthesizer wrappers. `DexedWrapper(renderer=...)` selects it; the surface a
renderer implements is tiny (enumerate parameters, get/set one parameter by index in raw [0,1],
render one held MIDI note to a raw `(channels, samples)` buffer). All engine-agnostic logic
(name↔index map, exclusions, categoricals, `ParameterSpace`, mono conversion) stays in the
wrapper, so it works with any renderer unchanged.

- **`DawDreamerRenderer` is the default** and the engine all `D-REPRO` characterization was done
  on. **`PedalboardRenderer`** is a secondary option (pip-installable, no Faust/automation —
  none of which this framework needs). **RenderMan is not supported** (Python 2.7 / Boost / no
  Apple Silicon).
- **Renderers must never be mixed within a single dataset/eval run.** The render-reproducibility
  contract (`D-REPRO`) holds per engine, not across engines — a target generated with one engine
  and re-rendered with another would inject an error floor. The active renderer name is recorded
  in run metadata.
- Engine choice was de-risked empirically by `scripts/benchmark_renderers.py`, which compares
  total render time (primary) and cross-engine audio agreement (secondary) over seeded patches.

**Benchmark results (2026-06-15)** — append-only; the decision above is unchanged.

- **Config.** `scripts/benchmark_renderers.py`, N=3000 patches sampled uniformly over the
  provisional subset; **seed 0 canonical** (seeds 1–2 also run, for stability). Render settings from
  `config.py`: 22050 Hz, 4.0 s render, 3.0 s note (note 60, velocity 100), buffer 128. Machine:
  Apple M5 (Mac17,2), 10 cores, macOS (Darwin 25.5, arm64). Absolute speed is hardware-dependent;
  the cross-engine *ratio* is the portable figure.
- **Speed.** DawDreamer median **3.6 ms/render** (~262 renders/s); Pedalboard median
  **18.1 ms/render** (~24 renders/s) → DawDreamer is **~5× faster per render**, stable across seeds
  (median ratio 4.8–5.0×). The headline "total render time" ratio swung **6.4×–13.1×** across seeds
  0–2 and is **not** stable: DawDreamer's total stayed ~11.7 s while Pedalboard's wall-clock total
  varied (75–155 s) from an outlier tail — its *median* per-render held at 18.1 ms, so the swing is
  scheduler/thermal noise, not patch content. Use the **~5× median per-render ratio** as the
  portable speed result, not the total-time ratio.
- **Near-silent patches.** ~**13%** of uniform-subset patches were near-silent (amplitude
  < 1e-3) and excluded from the agreement table (seed 0: 399/3000 = 13.3%; seeds 1–2:
  13.3–14.5%). Relevant to **D1** dataset generation: uniform sampling over the subset yields
  substantial silence.
- **Agreement (canonical seed 0; 2601 patches compared).**

  | metric | mean | median | p90 | p95 |
  |---|---|---|---|---|
  | log-spectral distance (dB) | 1.24 | 0.0001 | 7.08 | 8.51 |
  | spectral convergence | 0.158 | 0.0000 | 0.996 | 1.224 |
  | normalized RMS difference | 0.217 | 0.0000 | 1.410 | 1.424 |

  Percentiles were stable across seeds 0–2 (LSD p90 7.1–7.4 / p95 8.5–8.9; SC p90 1.0–1.1 /
  p95 1.22–1.28; RMS p90 ~1.41 / p95 ~1.42); medians stayed ~0.
- **Interpretation (HYPOTHESIS, not a finding).** Agreement looks **bimodal**: near-identical for
  the median patch (LSD ~0.0001 dB) but with a divergent ~p90 tail whose magnitude is the **same
  order as the D-REPRO within-engine worst case** (LSD ~9 dB, SC ~1.35). This suggests the
  cross-host disagreement is mostly the **D-REPRO hidden-voice-state mechanism** showing up
  *between* engines, not the two hosts rendering the patch differently. **Testable**: do the
  high-divergence patches here coincide with the high-context-leakage patches from the D-REPRO study?
- **Confirmatory test (2026-06-16)** — append-only; the decision above is unchanged. The testable
  question was run (`scripts/measure_context_leakage.py`). For each non-silent seed-0 patch a
  *within-engine* context-leakage score was measured in one DawDreamer process as the LSD between
  the patch rendered after primer A vs after primer C (the A/C-primer method of the D-REPRO xfail
  test `test_render_unaffected_by_previous_render_content`), then correlated against that same
  patch's cross-engine LSD. Over the 2601 patches: **Spearman ρ = 0.62** (p ≈ 4e-276); the
  within-DawDreamer leakage tail has the **same magnitude** as the cross-engine tail (leakage
  p90 6.88 / p95 8.52 dB vs cross-engine p90 6.97 / p95 8.54 dB; both medians ~0); and the
  **top-decile patches coincide 90.8%** of the time (9.1× over the 10% chance rate). The patches
  that disagree most *between* engines are thus overwhelmingly the same patches that are most
  context-dependent *within* one engine, at the same magnitude — **strong evidence the cross-engine
  tail is the D-REPRO hidden-voice-state mechanism, not host-implementation difference.** Caveats:
  the evidence is correlational (coincidence, not isolated causation); the correlation is carried by
  the shared tail (both medians ~0, so the bulk is uninformative); and it bounds but does not
  zero out a possible small genuine host difference. Per-patch data:
  `figures/data/context_leakage_seed0.csv`.

**Reload-per-render test (2026-06-17)** — append-only; the decision above is unchanged. The
*interventional* counterpart to the (correlational) confirmatory test above. `scripts/benchmark_renderers.py`
was rewritten into a **3-arm** benchmark: **(1) dawdreamer-reuse** (one persistent instance, the
default), **(2) dawdreamer-reload** (a fresh `DexedWrapper` — engine rebuilt + plugin reloaded — per
render, in-process; faithful to preset-gen-vae's reload-per-render, `paper_repos/preset-gen-vae/data/dexeddataset.py:243`),
and **(3) pedalboard**. Same patch set as the 2026-06-15 benchmark (N=3000, seed 0 canonical;
22050 Hz, 4.0 s / 3.0 s note; Apple M5). Two questions: how costly is reload-per-render, and does it
neutralize the hidden voice state (the mitigation the paper used but never characterized)?

- **Speed.** Median per-render: reuse **3.4 ms**, reload **30.8 ms** (decomposed: **27.0 ms** plugin
  reload + 3.8 ms render — the render component matches reuse, so the cost is purely the reload),
  pedalboard 18.2 ms. **Reload-per-render is ~9× slower than reuse** (and the reload arm is the slowest
  of the three). Total wall-clock to render all 3000: reuse 10.3 s, reload 93.6 s, pedalboard 124.6 s.
- **Sanity.** The **reuse↔pedalboard** table reproduced the recorded 2026-06-15 numbers (LSD p90 7.14 /
  p95 8.58 vs recorded 7.08 / 8.51; 2601 patches, 399 near-silent skipped), confirming the rewrite did
  not change the measurement.
- **Agreement — the interventional result.** All three pairwise tails are the **same magnitude**
  (medians ~0; LSD p90 / p95): reuse↔pedalboard **7.14 / 8.58**, reload↔pedalboard **7.02 / 8.48**,
  reuse↔reload **7.07 / 8.60**. Reload↔pedalboard is statistically indistinguishable from
  reuse↔pedalboard (~1.5% smaller, within seed noise) — **in-process reload does NOT collapse the
  cross-engine tail.** And reuse↔reload carries a full tail of the same size, so reload is not a no-op
  either: it produces a *third, equally-divergent* realization of the sensitive patches rather than
  converging on context-independence. This is exactly what D-REPRO predicted — freeing an engine and
  rebuilding it **in-process reuses dirty heap memory and diverges** — and it shows the paper's
  reload-per-render mitigation (which targeted gross hanging notes, never the subtle state) does **not**
  escape the hidden voice state on DawDreamer. Genuine isolation would require a **fresh OS process**
  per render (what preset-gen-vae's `multiprocessing.Pool` incidentally provided), consistent with the
  D-REPRO consequence that dataset generation render in fresh worker processes. Per-patch data (9
  metric columns, all three pairs): `figures/data/host_agreement_3way_seed0.csv`.

**Human-preset cartridge benchmark (2026-06-17)** — append-only; the decision above is unchanged.
The reload-per-render test above used seeded random patches; this run used **all 1056 voices from
the 33 real DX7 cartridges** in the standard Dexed install directory (`Dexed_01.syx` + 32
SynprezFM banks), via `scripts/benchmark_renderers.py --cartridges`. Same 3-arm setup, same render
settings (22050 Hz, 4.0 s / 3.0 s note; Apple M5).

- **Near-silence: 0/1056.** No near-silent patches — real human presets are all audible, in
  contrast to **13% silence** for uniform random subset sampling (see 2026-06-15 entry above).
  This is relevant to **D1**: the random-subset silence rate will inflate apparent dataset size.
- **Speed.** Consistent with the seeded runs: reuse **4.2 ms** / reload **30.8 ms** (26.5 ms
  reload + 4.5 ms render) / pedalboard **18.6 ms**; reload **7.4× slower** than reuse, reuse
  **4.5× faster** than pedalboard. Total wall-clock: reuse 4.4 s, reload 32.6 s, pedalboard 19.8 s.
- **Agreement.** Same bimodal structure, all three tails the same magnitude (LSD p90 / p95):
  reuse↔pedalboard **8.86 / 10.59**, reload↔pedalboard **8.93 / 11.10**, reuse↔reload **8.87 /
  11.07**. In-process reload does not collapse the cross-engine tail on the real-preset population
  either — conclusion generalizes from random patches to musically realistic ones.
- **Most-divergent presets.** The top divergers are overwhelmingly **LFO / sample-&-hold / noise**
  voices — exactly the patch class predicted by the hidden per-voice LFO/S&H state mechanism:
  `SynprezFM_21:02 CIGALES` (69.68 dB), `SynprezFM_13:21 CROSSING` (32.63),
  `SynprezFM_04:03 S-H ZIBBLE` (23.92), `SynprezFM_18:17 COMPUTER 1` (23.53),
  `SynprezFM_02:02 SCHLBELL` (22.92). Most musical pads/basses are bit-identical (median ≈ 0).
- Per-patch data (1056 rows, `patch_label` column): `figures/data/host_agreement_3way_cartridges.csv`.

**Pedalboard leakage test (2026-06-17)** — append-only; the decision above is unchanged. Resolves the
D-REPRO open follow-up: *does Pedalboard exhibit the same within-engine context leakage as DawDreamer,
or is it a clean anchor?* All prior leakage evidence (the 2026-06-16 confirmatory test) was measured
*within DawDreamer only*. `scripts/measure_context_leakage.py --renderer pedalboard` reruns the exact
A/C-primer probe — render each patch after primer A vs after primer C in one **persistent Pedalboard**
instance, LSD between the two — over the same seed-0 / N=3000 patches and primers, then correlates
against the same cross-engine LSD column (`figures/data/host_agreement_seed0.csv`).

- **Pedalboard leaks at the same magnitude as DawDreamer.** Within-Pedalboard context-leakage LSD
  (n=2601 non-silent): median **0.0000**, **p90 7.08 / p95 8.51 dB** — statistically the same as the
  DawDreamer baseline (median 0.0000, p90 6.88 / p95 8.52; `figures/data/context_leakage_seed0.csv`).
- **And it predicts the cross-engine tail just as strongly.** Spearman **ρ = 0.620** (p ≈ 6e-276)
  between within-Pedalboard leakage and the DawDreamer↔Pedalboard cross-engine LSD; **top-decile
  overlap 89.2%** (8.9× over chance) — matching the DawDreamer numbers (ρ = 0.62, 90.8%).
- **Conclusion.** The hidden voice state lives in the **shared Dexed plugin binary, not the host**:
  both engines exhibit the same within-engine context leakage, of the same magnitude, and in both the
  leakage tail coincides with the cross-engine divergence tail. This rules out "Pedalboard is the clean
  anchor and the tail is a DawDreamer-only quirk" — neither host escapes the state in-process, exactly
  as D-REPRO predicts (only a fresh OS process isolates it). Per-patch data:
  `figures/data/context_leakage_pedalboard_seed0.csv`.

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
