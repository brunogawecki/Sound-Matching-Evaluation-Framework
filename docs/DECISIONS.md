# Design Decisions

Locked and open design decisions for the sound matching evaluation framework.
Decisions marked **LOCKED** are settled — do not re-litigate unless the user explicitly asks.
Decisions marked **OPEN** block the work listed under "Blocks".

Last updated: 2026-07-21 (D-SELFDESC — cluster Dexed feasibility spike: 1.0.1 fails on glibc, 0.9.8 confirmed working).

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

**Decomposed S&H/LFO leak attribution (2026-06-19)** — append-only; the decision/policy above is
unchanged. *Interventional* test of how much of the leak is sample-&-hold (the only mechanism
preset-gen-vae's `prevent_SH_LFO` mitigation targets) vs. general LFO vs. deeper non-LFO state, and
therefore whether that mitigation would remove the leak. `scripts/measure_context_leakage.py --cartridges`
runs the A/C-primer within-engine leak probe over all **1056 cartridge voices** under three arms, each a
parameter constraint applied to every rendered patch (primers + probe): **(1) baseline** (none),
**(2) S&H→square** (preset-gen-vae's `prevent_SH_LFO`: `LFO WAVE` sample&hold → square), **(3) LFO
disabled** (`LFO PM DEPTH` = `LFO AM DEPTH` = 0). Same render settings as the other cartridge runs
(22050 Hz, 4.0 s / 3.0 s note; Apple M5; ~68 s total).

- **The leak is entirely LFO-mediated.** Arm 3 (LFO disabled) drives the leak to **exactly 0.0 dB for
  all 1056 voices** (max 0.0000, not just the percentiles) — with no LFO modulation applied, rendering
  is perfectly context-independent. So there is **no non-LFO residual**: the hidden state is the LFO
  subsystem's running memory (free-running phase + the S&H held value), surfaced through non-zero LFO
  depth — not a generic uninitialized per-voice memory. This *refines* the earlier D-REPRO hypothesis
  ("stale/uninitialized per-voice memory"; consistent with the prior finding that KEY SYNC alone did
  not fix it — zeroing the applied depth does).
- **S&H is a small share; `prevent_SH_LFO` does NOT remove the leak.** Only **32/1056** voices use S&H,
  and Arm 2 (S&H→square) moves the population tail by ~2.5%: p90 8.56→8.40, **p95 10.50→10.24 dB**
  (median 0 throughout). It materially changes **23 voices** — for pure-S&H voices it removes the leak
  entirely (`S-H ZIBBLE` 20.71→0, `Randomize3` 14.90→0, `CRICKETS`/`RandomNots`→0), for mixed voices it
  roughly halves it (`COMPUTER 1` 24.83→12.70, `S&H BUBBLE` 13.37→6.30) — but it leaves the dominant
  **non-S&H LFO** tail untouched: the biggest divergers are *not* S&H and survive Arm 2, collapsing only
  under Arm 3 (`CIGALES` 53.25→53.25→0, `TECH PULSE` 24.37→24.37→0, `HORN MOD` 22.18→22.18→0,
  `SAW EM UP` 17.61→17.61→0).
- **Baseline sanity.** Within-engine baseline leak (p90 8.56 / p95 10.50) matches the cartridge
  *cross-engine* tail recorded above (reuse↔pedalboard p90 8.86 / p95 10.59), consistent with the
  cross-engine tail being this same in-engine mechanism.
- **Consequence for D1 / policy.** Constraining the subset to exclude S&H buys almost nothing
  (~2.5% of the tail); the only parameter-space constraint that removes the leak is disabling the LFO
  outright, which **584/1056 (55%) of real presets use** — too large a scope cut. This **strengthens the
  accept-and-document + fresh-process render discipline policy**: the leak cannot be cheaply constrained
  away, and `prevent_SH_LFO` (which targeted gross S&H artifacts, not the subtle phase state) is not a
  fix for it. (Supersedes the speculative "D1 may lock those parameters to shrink the leak" aside in the
  policy section above — locking *S&H specifically* is near-useless; only full LFO removal works, and is
  not worth it.) LFO WAVE option values were resolved by the plugin's displayed parameter text
  (S&H = 1.0, SQUARE = 0.6 on this VST3 build; preset-gen-vae's "0.8" was a different build's order).
  Per-voice data (1056 rows: `patch_label`, `leak_baseline_db`, `leak_sh_square_db`, `leak_lfo_off_db`):
  `figures/data/context_leakage_arms_cartridges.csv`.

### D1 — Final Dexed parameter subset (LOCKED 2026-06-19)

The models estimate **103 of the 152 exposed parameters**; the rest stay at init-patch defaults.

**Rule**: take the preset-gen-vae / Le Vaillant learnable voice (the full DX7 voice — all six
operators on, all 32 algorithms, master tune and the per-op OP switches fixed) and drop the
parameters that are **non-identifiable under D3** (a single fixed note, C4, at fixed velocity 100).

**Estimated (103)** = 19 globals + 14 per operator × 6:

- Globals: `PITCH EG RATE 1-4`, `PITCH EG LEVEL 1-4`, `ALGORITHM`, `FEEDBACK`, `OSC KEY SYNC`,
  `LFO SPEED`, `LFO DELAY`, `LFO PM DEPTH`, `LFO AM DEPTH`, `LFO KEY SYNC`, `LFO WAVE`,
  `P MODE SENS.`, `TRANSPOSE`.
- Per operator: `EG RATE 1-4`, `EG LEVEL 1-4`, `OSC DETUNE`, `A MOD SENS.`, `OUTPUT LEVEL`,
  `MODE`, `F COARSE`, `F FINE`.

**Dropped (42)** = per operator: `BREAK POINT`, `L SCALE DEPTH`, `R SCALE DEPTH`, `L KEY SCALE`,
`R KEY SCALE`, `RATE SCALING` (keyboard scaling, only revealed across notes) and `KEY VELOCITY`
(only revealed across velocities). At the fixed C4 / velocity-100 render their effect is a constant
offset confounded with `OUTPUT LEVEL` (level scaling, velocity) or `EG RATE` (rate scaling), so
estimating them would reward guessing and pollute the parameter-side (diagnostic) metrics while
contributing nothing to the perceptual (primary) metric.

**Also fixed at defaults**: the 6 `OP{1..6} SWITCH` (all on, never learnable — matches
preset-gen-vae) and `MASTER TUNE ADJ` (matches preset-gen-vae). Tally: 103 estimated + 42 dropped
+ 6 switches + 1 master tune = 152.

**Categorical (per D-KIND), 16 of 103**: `ALGORITHM` (32), `OSC KEY SYNC` (2), `LFO KEY SYNC` (2),
`LFO WAVE` (6), per-op `MODE` (2 ×6) and per-op `F COARSE` (32 ×6); the other 87 are continuous.
Low-cardinality ordered grids (`FEEDBACK`, `P MODE SENS.`, `A MOD SENS.`, `OSC DETUNE`) stay
continuous per D-KIND's "ordered + perceptually progressive" arm.

**Why**: the kept set is a documented subset of the strongest comparable prior Dexed work
(preset-gen-vae), differing only by a principled, render-contract-driven cut, so the benchmark sits
on the same problem family with an explicit rather than arbitrary deviation. The LFO is left intact
(per the Decomposed S&H/LFO leak attribution above, disabling it would cut 55% of real presets); the
render leak is handled by the fresh-process render discipline, not by the subset. The choice of a
~100-param set over a smaller core also keeps the cut defensible without sacrificing comparability;
it was made over a ~35-param alternative (which would have made all six model families, including
evolutionary search, more directly competitive) — revisit if dimensionality proves to handicap a
family unfairly. The subset lives in `synth/dexed/subset.py`; `build_parameter_space()` validates
all 103 names against the live wrapper.

**Unblocks**: real training-dataset generation (GitHub issues #4/#5).

### D-SILENCE — Dataset silence gate: integrated LUFS (LOCKED 2026-06-22)

The `DatasetBuilder` flags/redraws a render as near-silent by **integrated loudness** (ITU-R
BS.1770, via `pyloudnorm`) below a floor, **not** by peak amplitude.

**Why**: peak is a single sample — a patch with a brief attack click but no sustained body clears a
peak gate while being perceptually silent — and the prior `1e-3` peak floor (≈ −60 dBFS) was far too
permissive. Integrated LUFS reflects *perceived* loudness over the note (its gating discards the
silent release tail), which aligns with the perceptual-similarity primary metric axis. This follows
ben-hayes/synth-permutations (LUFS reject-and-redraw); Sound2Synth used a stricter peak gate
(`>0.01`, ≈ −40 dBFS); preset-gen-vae needs no audio gate (real human presets are audible).

**Threshold**: default **−34 LUFS**, the **5th percentile of the 1051 built-in Dexed presets'**
loudness at the D3 render contract (human p5 −34.1, p10 −30.8, median −24.0). Rationale: the floor
should reject not just silent patches but *quiet* ones, so synthetic patches are at least as loud as
the quietest ~5% of real presets. (An earlier −45 was the valley of the *uniform-random* loudness
histogram, but that admits patches quieter than any human preset — the source of the "barely
audible" complaint.) Recalibrate per synth / render contract. The metadata records `loudness_lufs`
per sound alongside `rms` so the gate can be re-evaluated post hoc.

Note: the ~13% / amplitude<1e-3 figures in the D-RENDERER study above are historical *measurements*
from that experiment, not this gate.

### D-AUDIBLE — Synthetic preset source is constrained to be audible (LOCKED 2026-06-22)

`SyntheticPresetSource` no longer draws **purely** uniformly: optional **per-parameter range overrides**
(`sampling_ranges`) narrow chosen continuous parameters to an audible sub-range **at sampling time**.
The override map is owned by the synth (`BaseSynthesizer.audible_sampling_ranges`, default empty) and
applied via `ParameterSpace.sample_constrained` — the constrained params are drawn directly from the
sub-range, never sampled-then-overwritten. For Dexed the map is `synth.dexed.AUDIBLE_SAMPLING_RANGES`.
Because the map is declarative it is recorded in `run_summary.json` (reproducibility) and applied
consistently everywhere synthetic material is generated, including `HybridPresetSource` blend draws.

**Why**: uniform draws over the subset are ~30 dB quieter than human presets (uniform median −55.5
LUFS vs human −24.0); a patch is audible only if a *carrier* operator is loud with an open envelope,
which uniform sampling rarely produces. Pure rejection-sampling to a human-like floor (D-SILENCE)
would reject **94%** of draws (~15 renders/sample, exceeding the redraw cap) — so the source must be
fixed, not just its output filtered. This mirrors diffmoog (guarantee an active oscillator) and
pcmbs/synth-proxy (RMS-range redraw).

**How (Dexed)**: **OP1 is a carrier in all 32 algorithms** (verified against the live plugin), so
constraining OP1 alone makes any patch audible. The constrained parameters and ranges are
**calibrated to the built-in presets**, which keep OP1 `OUTPUT LEVEL` and `EG LEVEL 1` (attack peak)
near max (p5 0.85 / 0.72) and the attack rate reasonably fast (p5 0.33), while `EG LEVEL 3` (sustain)
varies freely (median 0.32). So the map draws OP1 `OUTPUT LEVEL`/`EG LEVEL 1` from [0.9, 1.0] and
`EG RATE 1` from [0.3, 1.0], and **leaves sustain/decay, frequency, the other five operators and the
algorithm random**. Because it only pins parameters humans already pin, the synthetic/human (train/test)
distribution shift is minimal and confined to OP1's diagnostic param metrics; the primary perceptual
metric is unaffected. With the constraint, median loudness rises to ~−36 and the −34 floor rejects
~60% (~2.5 renders/sample) instead of 94%.

**Limitation / future**: the constraint always forces *OP1* specifically, so its degeneracy lands on
OP1 rather than being spread across each algorithm's actual carriers (which would need a sourced DX7
algorithm→carrier table). The other operators stay uniform, so the corpus is still ~10 dB quieter
than human overall; biasing all operator output levels toward the human distribution is a possible
later step. Both are revisitable without changing the interface (`audible_sampling_ranges` is
declarative per-synth; range overrides currently cover continuous params, and the design extends to
categorical option-restriction if a future synth needs it).

### D-REPR — Audio representation is the model's job, not the Dataset's (LOCKED 2026-06-24)

The PyTorch Dataset over a rendered corpus (`dataset/torch_dataset.py`,
`RenderedCorpusDataset`) returns the **raw rendered waveform** (a fixed-length mono `float32` tensor,
88200 samples at the D3 contract) paired with the ML-side target vector. It computes **no**
spectrogram / mel / features and applies no amplitude normalization. Converting audio to a
representation (e.g. a mel-STFT on GPU, hand-crafted features for evolutionary search, or the raw
waveform for an end-to-end model) is each model's own first stage.

**Why**: this is a comparative benchmark across model families that want **different inputs**.
Computing one representation inside the Dataset (as preset-gen-vae / InverSynth2 do in `__getitem__`
— both are single-model codebases) would force every family onto one representation or require a
corpus/Dataset variant per representation. A representation-agnostic Dataset lets all families share
one corpus. Consequences: audio is fixed-length, so default collation suffices (no custom
`collate_fn`); a per-model on-disk feature cache can sit on top later if a family proves I/O-bound,
without changing the Dataset contract.

### D-SELFDESC — A built corpus serializes its own ParameterSpace (LOCKED 2026-06-24)

Each corpus's `run_summary.json` carries the full serialized `ParameterSpace`
(`ParameterSpace.to_dict()` / `from_dict()`), so the ML-side target vector can be reconstructed
**offline with no live synthesizer or VST**. `RenderedCorpusDataset.load(corpus_dir)` rebuilds the
space from the summary; the Dataset otherwise takes a `ParameterSpace` by dependency injection.

**Why**: building a `ParameterSpace` requires a live `DexedWrapper` (it reads names / options /
bounds / defaults off the plugin, per D-NAMING). Training runs on an external (Linux) GPU cluster
where we deliberately do **not** install a VST + dawdreamer toolchain (the plugins do ship Linux
builds, so this is a setup choice, not a hard platform limit). A self-describing corpus decouples the
training and target-reconstruction path from the VST + dawdreamer, matching how every run is already
reproducible from `run_summary.json`. (Note: the *Evaluator* still needs the VST for its re-render
step and runs locally on the Mac — see the Evaluator record below.) Consequences: the consumption module (`dataset/torch_dataset.py`) is
deliberately **not** re-exported from `dataset/__init__`, and `dataset/__init__` exposes the
generation API lazily (PEP 562 `__getattr__`), so importing the Dataset never drags in the
synth / render stack. `torch` is added as a dependency (the framework's first torch user).

**Feasibility spike (2026-07-21)** — append-only; the decision above is unchanged. Tested whether the
"setup choice, not a hard platform limit" claim actually holds, on the real PUT cluster
(`slurm.cs.put.poznan.pl`, Ubuntu 22.04.5, glibc 2.35 / `libstdc++` `GLIBCXX_3.4.30` ceiling — no
Apptainer/Singularity available).

- **Dexed's current release (1.0.1) does not load.** Its prebuilt Linux binary requires
  `GLIBC_2.38` / `GLIBCXX_3.4.32`, newer than the cluster ships. `dlopen` fails; JUCE's own error
  reporting surfaces this as a misleading `attempt to map invalid URI` / `Unable to load plugin`
  rather than a version-mismatch message, which cost most of the debugging time.
- **Dexed 0.9.8 (Oct 2024) works.** It needs exactly `GLIBC_2.35` / `GLIBCXX_3.4.30` — an exact
  ceiling match for this cluster. Confirmed both plugin load and actual rendering (`dawdreamer`
  `RenderEngine` + `make_plugin_processor` + a held MIDI note) producing real, non-silent audio
  (max amplitude 0.13 over a 3 s render), from `~/plugins/dexed/dexed-0.9.8-lnx/Dexed.vst3`.
- **No X11/Xvfb workaround needed.** JUCE's known headless-display quirk was the original risk
  hypothesis; it never materialized here — plain `pip install dawdreamer` plus the correct plugin
  version was sufficient, with no virtual framebuffer in the loop.
- **`dawdreamer` itself installs fine** on the cluster's Python 3.10 via a manylinux wheel
  (`dawdreamer==0.8.3`).

**Still open**: whether cluster-side Dexed rendering actually becomes part of the pipeline (e.g. for
faster dataset generation) is a separate decision from this spike — this only establishes that it is
technically possible and pins the version that works. Before relying on it: parameter-name parity
between 0.9.8 and whatever Dexed build generated the existing Mac-side corpora is unverified (D-NAMING
resolves names dynamically from the live plugin, so a renamed/missing parameter between builds would
silently change the subset rather than error). Tracked as a follow-up issue.

### D-METRIC-SR — Sample rate vs. deep-embedding metrics (LOCKED 2026-06-27)

**Decision**: the render rate stays **22.05 kHz** (`config.py` `SAMPLE_RATE`; the D3 contract is
unchanged). Spectral perceptual metrics (log-spectral distance / spectral convergence) and all
parameter (diagnostic) metrics are computed **natively at 22.05 kHz**. Only the
**deep-embedding metrics** (CLAP-style similarity, FAD) resample the audio to the embedding model's
required rate **at metric time**, inside the embedding-metric stage of the panel.

**Resampling contract**: high-quality, anti-aliased, deterministic resampling (e.g.
`torchaudio.functional.resample` or `soxr`), applied **identically to target and prediction**, up to
the model's native rate (48 kHz for CLAP). No amplitude renormalization beyond what the embedding
model itself requires.

**Why**:

- Rendering at 22.05 kHz hard-limits all audio to **< 11.025 kHz** (Nyquist). Upsampling 22.05→48
  therefore recovers nothing above 11 kHz — it only hands the embedding model the format it expects
  (which it would resample to internally anyway). Re-rendering at a higher rate is the *only* way
  embeddings would ever see genuine > 11 kHz FM content, and that cost (regenerate corpora, ~2×
  compute/storage, longer waveform-model inputs, breaking the `D-REPR` 88200-sample constant, and
  losing the direct 22.05 kHz comparability with preset-gen-vae) is not justified for a *comparative
  ranking*.
- The band-limit is **fair**: target and prediction are equally band-limited, so it adds no bias to
  the between-model comparison — the core thesis result.
- 22.05 kHz matches **preset-gen-vae** (`paper_repos/preset-gen-vae/config.py:30`, whose subset D1
  matches) and the DX7-matching literature (16–22.05 kHz).

**Threat to validity (document in thesis)**: the benchmark cannot perceptually distinguish content
above 11.025 kHz (bright FM partials, metallic/bell timbres). This caps *absolute* embedding fidelity
but does not bias model *ranking*. Report it as a stated limitation; revisit only if a later analysis
shows the > 11 kHz blind spot materially changes conclusions.

**Consequences**: `config.py` `SAMPLE_RATE` and the `D-REPR` 88200-sample tensor are unchanged — no
corpus regeneration. The metric panel (GitHub issue #8) owns the resample; it is not a Dataset
concern (per `D-REPR`, audio representation is the consumer's job). The embedding-metric dependency
(CLAP/FAD library + its torch/torchaudio needs) is added to `requirements.txt` when #8 lands, not
now.

### D-METRIC-NORM — Audio metrics compare raw audio (LOCKED 2026-06-27, REVISED 2026-06-30)

**Decision**: audio metrics in the panel compare the **raw** target and re-rendered prediction
waveforms — no loudness matching, period. There is no normalization knob. This matches **all five**
reference implementations surveyed (`paper_repos/preset-gen-vae`, `paper_repos/InverSynth2`,
synth-permutations, SynthRL, Sound2Synth), none of which loudness-normalizes before its audio
distances.

**Why**: loudness is part of a sound's character, and the panel's `loudness_*` metrics exist
precisely to measure it; matching levels first would cancel exactly what they capture. Raw comparison
also keeps the panel faithful to the literature it is benchmarked against and avoids hiding genuine
loudness errors a model makes.

**Revision note (2026-06-30, Evaluator #9)**: the originally-locked version kept a per-metric
`normalize_level` flag (off by default) as an opt-in escape hatch for a speculative problem — D-REPRO
render-level drift fooling a magnitude metric. The flag shipped **unused**: nothing ever set it true,
and the fresh-process re-render contract (see the Evaluator record below) removes the drift it was
meant to guard against. It was deleted with the Evaluator — the field, its `__post_init__` guard, and
all metric-line args. Re-adding it later (loudness-match in the Evaluator before a flagged metric) is
~20 minutes of work if rank-correlation analysis ever shows a metric needs it; the decision to
default-raw is unchanged.

**Scope**: this is a *level-normalization* decision and is independent of `D-METRIC-SR` (which governs
sample rate only). The two are distinct knobs; do not conflate them.

### D-METRIC-PERCEPTUAL — Embedding (perceptual) metrics deferred to future work (2026-06-29)

**Decision**: the **embedding-based perceptual axis** (CLAP, and the optional OpenL3 / JTFS
candidates) is **not implemented** in this thesis. It is descoped to *potential future work*. The
metric panel ships with its core audio axes — **magnitude, timbre, loudness, pitch** — plus the
**parameter** diagnostics; these stand alone and require no embedding dependency.

**Why**: the core panel already covers the thesis's primary metric axis — *perceptual audio
similarity* in the broad sense (audio-based distances vs. parameter-space distances). The deep
embedding metrics would add heavy, fragile dependencies (`laion_clap`/torch, `openl3`/TensorFlow,
`kymatio`) and a resample stage for marginal benefit to a *comparative ranking*, at a real cost to the
panel's reproducibility and dependency footprint. Keeping the panel embedding-free keeps the core
deliverable light and self-contained.

**Relation to `D-METRIC-SR` (LOCKED)**: `D-METRIC-SR` already defined the resample-at-metric-time
contract and the deferred `requirements.txt` embedding dependency *for if/when embedding metrics are
added*. That contract is unchanged and is **not re-litigated** here — it simply does not activate
while the embedding axis stays unimplemented.

**Consequences**: the `"perceptual"` value in `MetricAxis` (`evaluation/registry.py`) is retained as a
**reserved, unused** axis so a future contributor can add embedding metrics as one function + one spec
line; no embedding deps are added to `requirements.txt`. The glossary (`docs/CONTEXT.md`) marks the
perceptual axis as defined-but-deferred. With this, the metric panel core (GitHub issue #8) is
complete; next is the Evaluator (#9).

### D-EVAL — The Evaluator: monolithic + local, contract from the corpus (LOCKED 2026-06-30)

**Decision**: the Evaluator (`evaluation/evaluator.py`, GitHub issue #9) is the consumer of the metric
panel. Given a **fitted** model and a loaded `RenderedCorpusDataset`, for each sample it calls
`model.predict` (CPU), re-renders the prediction, runs the whole `METRIC_PANEL`, and writes a
self-describing results folder. Three non-obvious, hard-to-reverse choices are locked:

1. **Monolithic + local boundary.** `predict` + re-render + metrics run as one step on the Mac. There
   is **no** predict/re-render split artifact. Training (GPU-heavy) stays cluster-side; checkpoints are
   pulled to the Mac and the entire Evaluator runs locally, because the re-render step needs the VST
   (D-REPRO) and we keep the VST off the cluster (D-SELFDESC). The self-describing corpus means a
   split *can* be introduced later with no rework if a model's inference ever can't run on the Mac.

2. **Render contract comes from the corpus, never `config.py`.** The Evaluator reconstructs
   `RenderSettings` + renderer + sample_rate + `default_params` from the target corpus's
   `run_summary.json` and **hard-fails** if any field is missing. `config.py` could have drifted since
   the corpus was built; silently re-rendering every prediction under the wrong contract would corrupt
   the whole benchmark, so a wrong/absent contract must be loud.

3. **Re-render only the prediction, fresh-process at pos 0; the target is never re-rendered.** Audio
   metrics compare the fresh re-render against the **stored target WAV** (itself rendered fresh-process
   at pos 0 for the test corpus). Target and prediction therefore share an identical clean pos-0
   context, so the benchmark has no hidden error floor — a perfect prediction floors the audio metrics
   at ~0 (verified by the `test_true_parameters_floor_audio_metrics_at_zero` plugin test).

**Persistence**: each eval run is a self-describing folder mirroring the corpus convention —
`results/<corpus_name>/<model_name>/` (nesting by corpus makes "all models on one test set" a single
folder — the benchmark-table shape). Two files: `per_sample.csv` (the N×M matrix, `NaN`s intact — the
source of truth for the metric-panel rank-correlation pruning) and `eval_summary.json` (the render
contract echoed from the corpus, the checkpoint path + sha256 fingerprint, and per-metric
mean/std/**valid-count**). The Evaluator both writes the files and returns the in-memory result
(like `DatasetBuilder.build`).

**Aggregation**: the per-sample matrix is the source of truth; `NaN` means "metric undefined for this
sample" (not zero, not error). Aggregates are `nanmean` + std + valid-count, and the count is always
reported next to the mean so an "undefined hides failure" case (e.g. a silent prediction making
`f0_rmse` undefined everywhere) is visible, not masked.

**Why**: see the three points above — each trades a small amount of generality (no split artifact, no
`config.py` fallback) for a benchmark that is reproducible and impossible to silently corrupt.

**Update (2026-07-07)**: the Evaluator can optionally persist a **seeded random subset** of its
re-rendered predictions to disk, so the dashboard's Results page can A/B-play target vs. prediction
(see `D-DASHBOARD-CLUSTER`). Opt-in (`save_audio: bool = False`, default off) and capped
(`save_audio_n`, default 20) rather than on-by-default, because a benchmark sweep over hundreds or
thousands of samples shouldn't pay the disk/time cost of writing audio nobody listens to. The sample
indices are drawn with `np.random.default_rng(save_audio_seed)` rather than taking the first N, to
avoid corpus-ordering bias (e.g. a corpus sorted by source cartridge). Written under
`results/<corpus_name>/<model_name>/audio/<sample_id>.wav`, same float32 WAV convention the dataset
builder already uses for target audio — this does not change the per-sample matrix or eval summary,
it's an orthogonal side artifact of the same re-render already being computed.

---

### D-FRAMEWORK — Deep-model training framework: PyTorch Lightning (LOCKED 2026-06-30)

**Decision**: the internal training harness shared by the deep families (discriminative — primary,
generative VAE — primary, neural-proxy — InverSynth II) is built on **PyTorch Lightning**, not a
hand-written PyTorch loop. This fixes only the *internal* harness; it does **not** touch the
`BaseModel` contract (`models/base_model.py`), which stays framework-agnostic — its docstring already
states the loop-vs-`Trainer` choice "must never leak into this interface."

**Why** — three inputs:

1. **User priority.** Saving training boilerplate is valued over line-by-line loop transparency, and
   the user has prior Lightning experience and prefers it to raw PyTorch.
2. **Cluster fit** (PUT Poznań SLURM cluster): the `hgx` partition (8× A100-80GB/node), conda
   user-space installs, and a **24 h wall-clock limit with SIGTERM → SIGKILL**. Lightning's
   `SLURMEnvironment(auto_requeue=True)` gives automatic checkpoint-and-requeue on SIGTERM (directly
   addresses the time limit), `strategy="ddp"` for multi-GPU, and one-flag bf16 on A100 — bespoke,
   easy-to-get-wrong harness work a raw loop would force us to own.
3. **Contract fit.** The one real cost — Lightning leaking into the Mac-side eval path (D-EVAL) — is
   designed out (see "Conventions" below). The closest reference, preset-gen-vae, uses a hand-rolled
   loop with a heavy custom `RunLogger`/metrics harness; Lightning replaces that bespoke layer rather
   than reimplementing it.

**Conventions imposed on the Phase 4 training-harness task (issue #22)** — detailed-designed in that
task's own session, recorded here as inputs:

- **Decoupling from the eval path (the key pattern).** The trainable network is a plain `nn.Module`
  ("inference core"); a `LightningModule` *wraps* it for training only. `BaseModel.save`/`load`
  round-trip a **plain `torch` `state_dict`** (+ minimal hparams), never a raw Lightning `.ckpt`. The
  Mac Evaluator (D-EVAL — runs locally, calls `model.load`) therefore needs only `torch`; **Lightning
  never becomes a Mac-side dependency**, leaving D-SELFDESC / D-EVAL unchanged.
- **SLURM survival.** `SLURMEnvironment(auto_requeue=True)` + a `ModelCheckpoint` callback so the 24 h
  SIGTERM checkpoints and requeues.
- **Logging.** `CSVLogger` (no-internet-friendly on compute nodes); avoid W&B unless outbound network
  from `hgx` nodes is confirmed.
- **Precision / scale.** bf16 mixed precision on A100; `devices` / `strategy="ddp"` left config-driven
  (a student GrpTRES quota may cap GPUs).
- **Reproducibility.** `pl.seed_everything(seed, workers=True)` + deterministic flags, recorded in the
  run config.
- **Dependency placement.** `lightning` goes in the **cluster/training** requirements set (created by
  the Phase 4 cluster-packaging task, issue #20), **not** the base `requirements.txt` (the local/VST
  side, which already has `torch` and is unchanged by this decision).

**Implementation notes (2026-07-02)** — append-only; the decision above is unchanged. Issue #22
shipped in PR #28; the conventions above are delivered as specified — Lightning is quarantined to
`models/training/`, `save`/`load` round-trip a plain-`torch` artifact, `SLURMEnvironment(auto_requeue=True)`
is attached only when `SLURMEnvironment.detect()`, logging is `CSVLogger`, precision is `bf16-mixed`,
and `lightning`/`pyyaml` live in `requirements-cluster.txt` (a smoke test asserts importing `models`
pulls in no Lightning). The build also settled these sub-decisions, recorded here for their *why*:

- **Loss weighting follows preset-gen-vae, not a fresh guess.** `models/training/loss.py`
  (`ParameterLoss`) routes losses off `ParameterSpace.loss_slices` (D2): MSE (or MAE) on the
  continuous slots, per-block `cross_entropy` on categorical logits **averaged over blocks**, combined
  as `continuous + categorical_loss_weight · categorical`. `LossConfig.categorical_loss_weight`
  defaults to **0.2**, matching preset-gen-vae's empirically-tuned `categorical_loss_factor` — cross-
  entropy is typically much larger in magnitude than MSE, so an unweighted sum would let categoricals
  dominate. Config-overridable; 0.2 is the starting point, not a locked value.
- **The held-out human test set is never used for training-time validation.** `DataConfig.val_fraction`
  is `Optional` and defaults to `None`; validation is opt-in. `CorpusDataModule.setup` source priority:
  explicit validation corpus → seeded sample-level `random_split` by `val_fraction`
  (`torch.Generator().manual_seed(seed)`) → no validation. With no validation source the val loop is
  *disabled* (`limit_val_batches=0`) and the monitored metric falls back from `val_loss` to
  `train_loss`; `CorpusDataModule.will_validate` (readable before `setup`) is the single source of
  truth the caller uses to pick the monitor. Keeps D4's human split out of the training signal.
- **Config is fail-loud and reproducible.** Training knobs are frozen dataclasses
  (`models/training/config.py`) that **reject unknown keys** at every nesting level
  (`_reject_unknown_keys`) — a YAML typo errors rather than silently no-op'ing — and round-trip via
  `to_dict` so the resolved config can be echoed next to the checkpoint and a run's exact settings
  recovered. `from_yaml` imports `yaml` lazily so the eval path never needs pyyaml.
- **Checkpoint is a self-contained, versioned `torch` artifact** (`models/training/checkpoint.py`):
  one `torch.save` dict of `{format_version, CPU state_dict, architecture_hparams,
  parameter_space.to_dict()}` — enough to rebuild and reload a model with no training data and no VST
  (extends D-SELFDESC to the model side). `CHECKPOINT_FORMAT_VERSION = 1` is guarded on load;
  `weights_only=False` is intentional (our own trusted artifact carries Python containers). Training
  writes Lightning `.ckpt`s; `fit` then exports the best one by stripping the `network.` prefix via
  plain `torch.load` (no Lightning import).
- **New `BaseDeepModel` base class** (`models/base_deep_model.py`) sits between `BaseModel` and the deep
  families: `_build_network(architecture_hparams)` (abstract, torch-only so `load` can rebuild the net
  before loading weights) plus shared, Lightning-free `save`/`load`/`predict`. `predict` decodes the
  network's raw output (continuous floats + categorical logits) into a valid synth-side dict via
  `ParameterSpace.ml_vector_to_synth_dict` (argmax + bounds-clip), honoring the `BaseModel` contract.
  The network is *injected* into the harness (featurization lives in its `forward`), keeping the
  harness architecture-agnostic; `tests/tiny_deep_model.py` is the reference wiring a real family
  mirrors.
- **Seeding is the caller's job** — `pl.seed_everything(seed, workers=True)` before `fit`, not inside
  `build_trainer`.

Provisional (not locked): `CSVLogger` is currently hardcoded (has a `TODO` to make it config-driven);
AdamW `3e-4` / `weight_decay=0.0` / constant LR / `bf16-mixed` are sensible, config-overridable
defaults from the discriminative-regressor lineage.

---

### D-CLUSTER — Cluster packaging: conda + pip, git-clone provenance, /home-in-place (LOCKED 2026-07-06)

**Decision**: how the training path is packaged for the PUT Poznań SLURM cluster (`hgx` partition,
A100-80GB, `slurm.cs.put.poznan.pl`). Six choices, all grounded in the cluster guide
(`put-gpu-access.pdf`) and the confirmed VST-free import chain. Delivered under `cluster/` + a
finalized root `requirements-cluster.txt`; **no library code changes** (the harness was already
SLURM-aware — `models/training/trainer_factory.py` attaches `SLURMEnvironment(auto_requeue=True)`
when SLURM is detected, per D-FRAMEWORK).

1. **Environment: conda + pip.** `conda create -n smef python=3.11`, then
   `pip install -r requirements-cluster.txt`. Conda is the guide's supported install route (no
   Docker) and only supplies a user-space Python 3.11 without root; pip installs the actual deps. The
   requirements file **is** the dependency-split artifact.
2. **`requirements-cluster.txt` is the complete VST-free split**, not an add-on. It finalizes the
   #22 stub (which listed only `lightning`/`pyyaml`) into the full base-minus-VST set: `numpy`,
   `scipy`, `pandas`, `python-dotenv`, `torch` + `lightning`, `pyyaml`. Dropped vs. base
   `requirements.txt`: `dawdreamer`, `librosa`, `pyloudnorm`, `streamlit`, `tqdm`
   (render/eval/dashboard, none reached by the training import chain — D-SELFDESC / D-EVAL). Pinned
   to the local dev versions
   for reproducible runs.
3. **Code sync: git clone + `git pull`.** The repo is public, so no auth on the cluster, and every
   run is traceable to a commit.
4. **Corpus: `rsync -avP` to `/home`, read in place.** `/home` is shared Lustre across all nodes; no
   node-local `/raid` staging (premature at ~10 GB). The cluster corpus path is passed through the
   existing `--corpus` flag.
5. **Machine-specific values via gitignored `cluster/cluster.env`** (+ committed
   `cluster.env.example`), mirroring the repo's `.env` convention — no SSH target, account, or path
   is hardcoded in a committed script. Sourced by both the sbatch job and the laptop transfer
   scripts. The SLURM billing account is passed as `sbatch -A "$SLURM_ACCOUNT" cluster/train.sbatch`
   because `-A` is a submission-time flag the `#SBATCH` body cannot read.
6. **Docs-first: `cluster/README.md` walkthrough + two transfer scripts** (`push_corpus.sh`,
   `pull_checkpoint.sh`). One-time setup and submit/monitor are documented, not scripted (they run
   rarely and vary); only the recurring rsync pair is scripted. The README doubles as the thesis
   Implementation-chapter source.

**Acceptance bar (smoke slice).** #20 closes on one end-to-end reduced-scale pass: a short sbatch job
(`smoke_config.yaml`, 2 epochs, single GPU, `--time` well under the 24 h cap) on the real corpus →
checkpoint pulled down → loads + predicts locally. This proves the packaging without waiting for a
full run. `auto_requeue` on SIGTERM is treated as an untested safety net, not verified here. The full
run reuses `train.sbatch` with a fuller config and a larger `--time`.

---

### D-DASHBOARD-CLUSTER — Dashboard drives the cluster: SSH shell-out, local job registry (LOCKED 2026-07-07)

**Decision**: the Streamlit dashboard (`dashboard/`) submits, tracks, and pulls **training jobs** on
the PUT cluster directly, so training no longer requires SSHing in and running `cluster/*.sh` /
`sbatch` by hand. Builds on `D-CLUSTER` (packaging) unchanged; this covers orchestration only.

1. **Remote execution: shell out over `subprocess` + `ssh`**, reusing the existing `cluster/*.sh`
   scripts and `command_runner.py`'s subprocess pattern. No new dependency (no paramiko/fabric) — the
   dashboard already shells out to local scripts, this is the same mechanism pointed at a remote host.
2. **Git-sync guard: warn, don't block.** Before submit, the dashboard checks local `git status` /
   unpushed-commit state and shows a warning if dirty or ahead of the remote, because the cluster only
   ever sees what's been `git pull`ed from GitHub (D-CLUSTER §3) — a stale or uncommitted local state
   silently trains against old code otherwise. Warning rather than a hard block, since there are
   legitimate reasons to submit anyway (e.g. testing an already-pushed commit while iterating locally).
3. **Corpus push: always `rsync`, every submit, no "already pushed" tracking.** `rsync -avP` is a
   stat-only no-op when the remote copy already matches (filename + size + mtime), so re-syncing an
   unchanged corpus costs a walk, not a transfer — tracking push state separately would be an
   optimization for a cost that's already negligible.
4. **Job tracking: local gitignored `cluster/jobs.json`** (see **Job registry** in `CONTEXT.md`), not
   a live cluster query. `sacct` history is not a reliable long-term job list (retention policy,
   requires knowing job ids), and the dashboard process itself is not always running, so job identity
   has to live in a file the dashboard reads back on restart.
5. **Progress display: poll, don't stream.** `sacct` for SLURM state and `ssh ... tail` of the SLURM
   stdout file, on a `st.fragment(run_every="5s")` timer (supported on the installed Streamlit 1.58).
   Reuses the `\r`-collapsing logic `command_runner.run_streaming` already applies to local `tqdm`
   output, so Lightning's live progress bar renders as one animating line under SSH tailing too,
   instead of scrolling duplicate lines.
6. **Checkpoint pull: manual button, not automatic on completion**, since the dashboard is not always
   open when a 12-hour job finishes; polling for completion just to auto-pull adds complexity for a
   trigger the user is already looking at the Jobs list to press.
   *Amended 2026-07-14 — the pull is **job-scoped**.* Training writes to
   `checkpoints/<job id>/` and `lightning_logs/<job id>/` (`train.sbatch` passes `$SLURM_JOB_ID` to
   `fit_model.py --run-id`), and the button pulls exactly that job. Previously every run of a family
   wrote one shared `checkpoints/<model>.pt`, so a re-run destroyed the earlier run's checkpoint
   before it could be pulled and the button silently served the newer file under the older job's row.
   Raw Lightning `.ckpt` files (~450 MB each) stay on the cluster behind an opt-in `--with-ckpt`:
   they only carry optimizer state for *resuming*, which happens cluster-side, and the exported `.pt`
   already holds the best epoch's weights. Jobs submitted before this change have no per-job
   directory; the pull falls back to the shared path and warns that the file may belong to a later run.
7. **Cancel: `scancel` via a button** on any job in a non-terminal state — cheap to add alongside the
   status/log-tail view and avoids a stuck job silently occupying the GPU allocation.

**Why**: the alternative (a lightweight job-queue service, or a persistent SSH-tunnel/websocket
process) would solve the same problem with materially more moving parts than this thesis's scope
justifies; the shell-out + polling design reuses everything the dashboard and `cluster/` already have.

---

### D-SPLIT — Post-render corpus splitting (LOCKED 2026-07-08)

**Decision**: a corpus that has already been rendered can be split into a **train** corpus and a
**test** corpus (`scripts/split_corpus.py`, `dataset/corpus_splitter.py`, and the dashboard's *Split
corpus* page). This is distinct from build-time splitting (`DexedPresetLoader`, D4), which splits
*presets before rendering*. It exists so a held-out test set can be carved out of an existing corpus
without re-rendering all of it — e.g. the ~30k-sample preset-gen corpus, built all-train in-process.

1. **Train audio is copied; the test partition is re-rendered fresh-process at position 0.** A
   copy-only split of an in-process corpus would produce test targets that carry context leakage and so
   violate the eval render contract (D-REPRO / D-EVAL: the Evaluator re-renders each prediction fresh
   at pos 0 and compares it to the target). Re-rendering only the test fraction is cheap and restores
   the contract; train render context is irrelevant to training, so those WAVs are copied verbatim.
   This mirrors exactly what `build_dataset.py human` already does (test fresh, train in-process). The
   test partition's presets are replayed from the source `metadata.csv` via `CorpusPresetSource` (the
   103 subset params are stored per row; dropped params fall back to the synth defaults as at build).
2. **Seeded row-partition, reusing the build-time algorithm.** `split_indices` (factored out of
   `split_presets`) is the single source of truth: permute positions with `split_seed`, take the first
   `round(n · test_fraction)` as test. So a corpus split and a build-time split shuffle identically.
3. **No deduplication at split time.** Human / preset-gen corpora were already deduplicated by the
   loader before rendering, and a deduplicated set stays deduplicated when partitioned; synthetic draws
   never near-collide. Re-running the O(n²) dedup scan would be redundant. (Dedup is still the build-time
   guard — see **Deduplication** in `CONTEXT.md`.)
4. **Hybrid corpora are refused.** Their augmented children (and repeated blend parents) derive from
   shared human parents, so a row-level split would scatter a parent and its derivatives across train
   and test — **train/test leakage** (see `CONTEXT.md`). Enforced in the script and surfaced in the UI
   (the corpus is shown but blocked with the reason). To get a held-out human test set, split the human
   source cartridges at build time instead. Synthetic and human corpora are leakage-free under a row
   split (each row is an independent draw or a unique already-deduplicated voice).

Both output corpora stay self-describing (D-SELFDESC): each carries the source's `parameter_space`,
`render_settings`, `subset_names`, and `default_params` unchanged, with a `source` block recording the
split provenance (`split_from`, `split_test_fraction`, `split_seed`, and the original construction
`method`). The test corpus records `render_process: fresh` (so discovery flags it eval-ready); the
train corpus keeps the source's render process.

**Why**: the framework's discipline is that eval targets are rendered fresh at pos 0 (D-REPRO), so a
useful post-render split cannot be a pure file copy — it has to re-render the held-out half. Doing that
for only the test fraction keeps the operation cheap while producing a contract-correct test corpus,
which is the whole point of holding data out.

---

### D-MELNORM — preset-gen-vae mel-dB front-end normalizes from corpus stats (LOCKED 2026-07-09)

**Decision**: the preset-gen-vae port's mel-dB spectrogram front-end (`models/presetgen_vae/network.py`)
min-max normalizes to [−1, 1] using the **actual min/max dB measured over the train corpus**, not a
hardcoded dB range. The two endpoints are computed in one pass at the start of the family's `fit()`
and folded into the checkpoint's `architecture_hparams` (exactly like `num_audio_samples` /
`sample_rate`), so `load()` rebuilds the identical normalization offline with no corpus and no VST
(D-SELFDESC-aligned). The dB **floor** stays a fixed constant (−120 dB); only the normalization
endpoints are corpus-derived.

**Why**: from Stage 2 on, the normalized spectrogram is also the decoder's **reconstruction target**.
Real Dexed mel-dB values occupy only part of a fixed [−120, 0] dB range (nothing reaches 0 dBFS), so
normalizing against fixed endpoints squashes the target into a sub-interval of [−1, 1] and wastes the
decoder's `Hardtanh` output range, weakening the reconstruction gradient. Corpus-derived endpoints
make the target fill [−1, 1]. This is the framework-native form of the paper's cached-spectrogram-stats
step (`utils/audio.py` + `data/abstractbasedataset.py` compute `spec_stats['min'/'max']` over the
training set and cache them to a JSON sidecar); deriving-at-fit and folding into hparams reuses the
existing corpus→hparams→checkpoint pattern instead of a separate cached file.

**Alternatives considered**:

- *Fixed [−120, 0] dB* (the Stage-1 placeholder, comment: "Stage 2 may swap in corpus stats") —
  rejected: squashed target, poorly-scaled reconstruction.
- *Full paper-faithful front-end* (corpus stats **plus** window-energy normalization ÷`rfft(hann).max()`,
  a linear-domain floor, dropping the upper clamp, non-periodic Hann, constant padding) — rejected.
  Every element beyond the corpus stats is either **absorbed** by the normalization (the window factor
  is a constant dB offset the corpus min/max cancel), **practically identical** (floor method),
  **dead code** once the endpoints are real (the upper clamp never fires below 0 dBFS), or **sub-percent**
  (periodic-vs-not window, reflect-vs-constant pad, at the signal edges only). It is more churn to the
  shared front-end for no change in what the network sees. Since the thesis reproduces the paper's
  *method*, not its *numbers* (different renderer / corpus / 103-vs-144 param space / categorical scheme
  already guarantee non-matching numbers — see D1, D-METRIC-SR), byte-faithfulness buys nothing here.

**Consequences**: `PresetGenVAENetwork` keeps `spectrogram_min_db` / `spectrogram_max_db`
as constructor args (the fixed floor stays the default lower value), but `PresetGenVAEMLPRegressor.fit()`
overwrites the normalization endpoints with the measured corpus values before building the network and
recording hparams. The −120 dB floor and the minor front-end details (eps-in-log floor, upper clamp,
periodic Hann, reflect pad) are unchanged and explicitly **not** pursued.

(An earlier Stage-1 no-VAE regressor shared this same corpus-stat wiring for input parity; it was
removed as out of scope, so the endpoints are now measured for the one VAE family only.)

---

### D-FLOW-CORPUS — The flow-matching families train on a synthetic-uniform corpus (LOCKED 2026-07-20)

**Decision**: the flow-matching families (`FlowMatchingMLP`, `FlowMatchingParam2Tok`) train on a
**synthetic-uniform** corpus (`ParameterSpace.sample_uniform` via `scripts/build_dataset.py
synthetic`), not the human preset corpus every other deep family trains on. The **test** corpus is
unchanged: the shared benchmark test set, same as every family (D4, Phase 6).

**Why**: the paper's claim (Hayes et al., ISMIR 2025) is that building the synth's permutation
symmetry into the vector field helps. That only holds if the training **parameter prior is
G-invariant** — invariant under the symmetry group being exploited. Uniform sampling over the
subset gives this for free: permuting an operator's parameters maps one uniform draw to another
equally likely one. Curated human presets do not — they are heavily biased toward particular
operator roles and algorithm choices, which breaks the invariance and removes the structure
Param2Tok is built to exploit. The paper attributes its own VAE+RealNVP collapse to exactly this
kind of preset bias. Training Param2Tok on human presets discards the reason it should win, so the
MLP-vs-Param2Tok comparison would measure nothing.

**Known confound — the corpus is not exactly G-invariant.** "Synthetic-uniform" here means uniform
in the D-AUDIBLE sense: `scripts/build_dataset.py synthetic` always applies
`synth.audible_sampling_ranges`, which for Dexed pins three OP1 parameters (`OP1 OUTPUT LEVEL` and
`OP1 EG LEVEL 1` to [0.9, 1.0], `OP1 EG RATE 1` to [0.3, 1.0]). Because the constraint names **OP1
specifically**, the prior is *not* invariant under permuting operators: a draw with OP1 swapped for
OP4 is not equally likely. That is a partial break of the exact property this decision exists to
secure, sitting directly on the axis the family is meant to demonstrate. D-AUDIBLE's own
"Limitation / future" paragraph anticipates it ("the constraint always forces OP1 specifically, so
its degeneracy lands on OP1").

**Accepted anyway**, for now: the break is 3 of 103 parameters and one operator of six — the
algorithm, all frequencies, sustain/decay, and the other five operators stay free, so the prior is
*approximately* invariant. That is a defensible pairing with a model the paper itself only claims to
be *approximately* equivariant. `FlowMatchingMLP` is the control that keeps this honest: it is
non-equivariant, so if the OP1 pin were destroying the effect, the two families should converge.

The alternatives were both rejected as disproportionate for now. Sampling with no audibility
constraint restores exact invariance but sends the D-SILENCE rejection rate to ~94% (~15
renders/sample, over the redraw cap), so it would mean bending two LOCKED decisions for one family.
Spreading the constraint across each algorithm's real carriers is the principled fix — it restores
invariance *and* improves D-AUDIBLE generally — but needs a sourced DX7 algorithm→carrier table and
is its own piece of work. Revisit if Param2Tok fails to separate from the MLP control: this
confound is then the first thing to rule out, before concluding against the paper's premise.

**Consequences**: this family deviates from the shared-training-corpus pattern deliberately, and
that is a fact the Methodology chapter must state rather than gloss. It also makes "dataset
construction method" a *usable comparison axis* for this family — training both flow-matching
families across synthetic / human / hybrid corpora, all scored on the same test set, is a direct
empirical test of the premise above. `FlowMatchingMLP` must be run alongside as the control on any
such sweep: without it, a drop under human-trained data cannot be attributed to symmetry-breaking
rather than to reduced training diversity.

Map and port fidelity: `docs/FLOW_MATCHING_PORT.md`.

---

### D-FLOW-PREDICT — Generative `predict` returns one seeded sample (LOCKED 2026-07-20)

**Decision**: `BaseFlowMatchingModel.predict` overrides the base single-forward-pass `predict` and
returns **one** sample drawn by integrating the learned ODE (CFG-guided RK4, the paper's test-time
protocol: 200 steps, guidance strength 2.0). The draw uses a **per-call seeded generator**
(`_predict_seed`, default 0), so repeated predictions of the same clip are identical.

**Why**: the base `predict` is a single forward pass, which is simply wrong for a sampler — the
network's `forward` is `sample`, not a regression. Beyond that, two properties matter: the result
must be **reproducible** (the Evaluator re-renders every prediction fresh-process and expects a
deterministic input — D-EVAL / D-REPRO), and it must be **comparable** to the discriminative
families, which emit exactly one parameter vector per target. One seeded sample gives both, and
matches the paper's own Table 1 protocol.

**Alternatives considered**:

- *Best-of-N* (sample N, re-render each, keep the closest) — deferred, not rejected. It is cheap to
  add because the Evaluator already re-renders, but it gives the generative families a re-ranking
  budget the discriminative ones do not get, so it is a **separate reported condition**, not the
  default.
- *Unseeded sampling* — rejected: non-reproducible predictions break the eval contract.

**Consequences**: a single draw does not measure the sampler's variance, which is a real property
these families have and the regression families do not. Reporting per-target sample statistics is
future work, noted in `docs/FLOW_MATCHING_PORT.md`.

---

## OPEN

### D4 — Human preset source for the test set (deferred by user; importer built 2026-06-24)

**What** specific presets form the held-out human test set is **deferred until the full ML pipeline
is finished** — an evaluation-design choice the user will make once the pipeline can be run
end-to-end, not a tooling gap.

**Importer is built (no longer a blocker).** The DX7 SysEx cartridge path is implemented, so any
`.syx` source can be turned into a corpus today: `synth.dexed.cartridge` validates and unpacks the
documented 32-voice bulk-dump format (4104 bytes: 6-byte header, 32 × 128-byte packed voices,
checksum, `0xF7`), mapping each voice onto Dexed's plugin-reported parameter names normalized to
[0, 1] exactly as Dexed normalizes them (raw / field-max; categoricals as index / (cardinality − 1)).
`dataset.dexed_preset_loader.DexedPresetLoader` projects each voice onto the estimated subset,
deduplicates near-twins on that projection, and makes a seeded, provably disjoint voice-level
train/test split. Surfaced via the `human` / `hybrid` subcommands of `scripts/build_dataset.py`;
test/eval corpora render with `--fresh-process` so generation and evaluation share an identical clean
render context (D-REPRO). (Offline-rendering constraint: DawDreamer ignores SysEx and MIDI Program
Change offline, so a voice is applied as parameters, not loaded as a patch — the importer does this.)

**Still open**: which cartridge collection(s) — or other source — actually become the benchmark test
set, and the final train/test composition. The built importer currently covers DX7 `.syx`; a
non-SysEx source (e.g. Surge `.fxp`) would need its own importer.

**Update (roadmap)**: the leading plan is now "train human → test human" on the
**preset-gen-vae human DX7 collection** (`paper_repos/preset-gen-vae/synth/dexed_presets.sqlite`,
~30k voices). That source is **parameter vectors, not `.syx`** (see `ROADMAP.md`, Phase 4 corpus
task), so it needs a name-based adapter rather than the SysEx importer. Under this plan D4 narrows to
a **voice-disjoint split of that same human corpus** (Phase 6). Still the user's call to finalize.

### D-FAMILIES — Final model-family set (OPEN, stub)

**What** model families enter the comparative benchmark. Working set: **discriminative** (primary) +
**generative** (primary, VAE — preset-gen-vae lineage) + **neural-proxy** (InverSynth II lineage — a
peer paper approach, **committed and built**: the staged `IS` / `IS2xITF` / `IS2` families, see
`docs/INVERSYNTH2_PORT.md`) + **conditional-generative flow matching** (Hayes et al. ISMIR 2025 —
**committed and built**: `FlowMatchingMLP` / `FlowMatchingParam2Tok`, the paper's own control and
its equivariant model, see `docs/FLOW_MATCHING_PORT.md`; trains on its own corpus per
D-FLOW-CORPUS). **Evolutionary search is dropped** (user: "probably no evolutionary
algorithms"); if ever reinstated, note it runs a per-target search locally with the live VST and does
**not** fit the cluster training harness.

**Why it's open**: the neural-proxy and flow-matching slots are now filled, but the final family set
is not frozen — the exact discriminative/generative architectures still evolve and a second synth
(Surge XT) may add families.

**Blocks**: Phase 5. Resolve here before the Phase 5 family tasks start.
