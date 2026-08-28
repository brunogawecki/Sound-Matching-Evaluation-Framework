# Design Decisions

Locked and open design decisions for the sound matching evaluation framework.
Decisions marked **LOCKED** are settled — do not re-litigate unless the user explicitly asks.
Decisions marked **OPEN** block the work listed under "Blocks".

Last updated: 2026-08-25 (D-DIVA-SUBSET locked — 237 of Diva's 281 parameters estimated; D-DIVA-RENDER locked — Diva reproduces fresh-process only; D-DIVA-START locked; D-NAMING amended for module-qualified names).

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
convention applies to every further wrapper.

**Amended 2026-08-25 (Diva).** A name only addresses a parameter if the plugin's names are
unique, which Dexed's are and Diva's are not (56 names shared by 147 parameters). Where they
collide, the canonical name is **module-qualified**, `<Module>.<Name>` — `LFO1.Rate`,
`VCF1.Model`. This is the same rule, not a new one: the requirement is a stable, unique,
human-readable key, and the module prefix is what makes one exist. Indices stay out of every
public API.

Two constraints follow. The module is **not** reported by VST3, so a wrapper whose plugin needs
qualification carries a committed name table validated against the live plugin, rather than
deriving names at `__init__` (`synth/diva/parameters.py`, `tests/test_diva_parameters.py`); and
each such wrapper owns the translation from its qualified names down to the bare names the plugin
answers to. Above the wrapper — subsets, `ParameterSpace`, corpus metadata, model code — only the
qualified name is ever seen. See **D-DIVA-START**.

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
table. The second synth's wrapper comes after, re-using the proven recipe. Rationale: fastest
end-to-end feedback; avoids a second subset decision while D1 is open.

**Exception granted 2026-08-25 (D-DIVA-START).** The second synth is **Diva**, not Surge XT, and
it starts before the benchmark table is finished. D-ORDER's condition is satisfied rather than
waived: the Dexed slice runs end to end and D1 is locked, so neither reason for the ordering still
applies. The rest of D-ORDER stands — Diva re-uses the proven recipe, it does not fork it.

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
- **Amended 2026-08-25: pluggability is per wrapper, not universal.** `DivaWrapper` accepts
  **DawDreamer only** and raises on any other renderer. Diva's parameter *index space* is
  engine-specific — Pedalboard reports 2271 parameters where DawDreamer reports 2362, because it
  drops the 91 whose names collide, so index 280 is a MIDI CC under Pedalboard and the last
  synthesis parameter under DawDreamer. Diva's name→index table (which cannot be rebuilt from the
  plugin, see D-DIVA-START) is written against DawDreamer, so another engine would silently
  repoint every parameter. Refusing is the only safe behaviour. Dexed stays pluggable: its names
  are unique, so its map is rebuilt per engine.

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

**Amendment (2026-08-28): the corpus also records which synth built it.** With a second synth in
the framework, the serialized `ParameterSpace` no longer identifies the plugin: Diva's narrowed
corpus space (see D-DIVA-SUBSET) names 58 parameters that mean nothing to Dexed. So
`run_summary.json` carries a `"synth"` field, written from the wrapper's `synth_name` and read back
as a key of `dataset/render_backends._SYNTH_REGISTRY`. The Evaluator uses it to re-render on the
synth the corpus was built with, instead of assuming Dexed (D-EVAL: the contract comes from the
corpus, never from `config.py`).

Unlike the render-contract fields, `"synth"` is **optional on read** and defaults to `dexed`: every
corpus built before the field existed is Dexed, so old corpora keep evaluating unchanged rather than
being invalidated.

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
waveforms — no loudness matching, period. There is no normalization knob. Four of the five
reference implementations surveyed (`paper_repos/preset-gen-vae`, `paper_repos/InverSynth2`,
synth-permutations, Sound2Synth) apply no level normalization before their audio distances.

**Correction (2026-07-22, SynthRL port)**: the originally-locked text claimed **all five**,
including SynthRL. That is wrong. SynthRL **peak**-normalizes the rendered prediction before every
audio distance — in the RL reward (`finetune.py:161, 202, 266`) and again at evaluation
(`evaluate.py:94`) — against targets themselves rendered with `normalize_audio=True`. Its distances
are therefore level-invariant. This is peak normalization, not the LUFS loudness matching this
record rules out, but the practical effect on its magnitude/timbre distances is the same. The
decision stands on its own reasoning below; only the "all five" evidence claim is retracted.
**Consequence to carry into the write-up**: our `SynthRL-i` reward includes level error where the
paper's does not, so the RL stage optimizes a slightly different objective than the published one
(see `docs/SYNTHRL_PORT.md`, deviation 3).

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

### D-RL-RENDER — The RL stage renders with the live VST inside the training loop (LOCKED 2026-07-22)

**Decision**: `SynthRLi` (the SynthRL in-domain RL stage) renders every sampled patch with the real
Dexed plugin **during training**, through a worker pool (`rl.num_render_workers`). The pool's
isolation is selected by `rl.render_isolation`: `"process"` (default) is the D-REPRO
fresh-process-per-render backend; `"reuse"` reuses one wrapper per worker for speed, at a measured
reward bias (see the 2026-07-27 amendment below). Either way this is a deliberate, **scoped**
deviation from D-SELFDESC: a `SynthRLi` *training* environment needs the VST. Nothing else moves.
The renderer import is training-only and lazy (D-FRAMEWORK), `predict` decodes class scores with no
synth involved, and the corpus/eval path is unchanged for every family including this one.

**Why**: the reward *is* the render. SynthRL's contribution is that the training signal comes from
audio similarity between the target and the render of the sampled action, which is what removes the
need for ground-truth parameters and lets the same recipe fine-tune on out-of-domain sounds. There
is no version of this family that does not render in the loop. Feasibility is not a guess: the
D-SELFDESC feasibility spike established that Dexed **0.9.8** loads and renders real audio on the
PUT cluster.

**Why fresh-process-per-render, not a reused in-process synth**: it keeps the RL reward and the
reported metric measuring the same thing. The Evaluator re-renders every prediction fresh-process at
sequence position 0 (D-EVAL / D-REPRO); if training rendered in a reused context, the reward would
be computed on context-leaked audio (D-REPRO quantifies this for LFO / S&H / noise voices) while the
results table scored clean audio. Widening to N workers changes throughput only — each patch still
gets its own single-use process (`spawn`, `maxtasksperchild=1`), so a parallel render equals a
serial one.

**Alternatives considered**:

- *In-process renderer reuse with a state reset between patches* — roughly two orders of magnitude
  faster per render, and the obvious optimization if stage 2 turns out to be render-bound. Was
  rejected here as future work; **subsequently built** once stage 2 proved render-bound and the
  leakage was measured — see the 2026-07-27 amendment. The "with a state reset" premise turned out
  to be false: no in-process reset exists.
- *A differentiable synthesizer proxy in place of the real render* (the InverSynth II route) —
  rejected. It replaces the paper's own mechanism with a second approximation layer, and that
  approach is already represented in the benchmark as its own family (`IS2xITF` / `IS2`), so folding
  it into SynthRL would blur two families into one.
- *Precomputing / caching renders* — not applicable. The policy samples fresh actions every step, so
  there is nothing to hit in a cache.

**Consequences**:

- The cluster training environment for this one family must install Dexed 0.9.8 + `dawdreamer`
  (version pinned by the D-SELFDESC spike — 1.0.1 does not load there). Every other family still
  trains VST-free.
- Stage 2 wall-clock is dominated by rendering **in `process` mode only** (~1.1 s/render). Under the
  `reuse` mode the cluster config actually uses, rendering is ~1% of a step and the cost is the
  per-sample reward and the REINFORCE pass over the parameter heads — see the 2026-08-16 amendment.
  `rl.num_render_workers` should still match the job's `--cpus-per-task`, and `rl.prefill_epochs`
  (default 5) costs that many gradient-free full passes over the training set before the first
  gradient step.
- **Parameter-name parity between Dexed builds now bites.** The corpus's `ParameterSpace` is rebuilt
  offline from `run_summary.json`, but the RL backend sets patches **by name** (D-NAMING) on the
  cluster's 0.9.8 build, while the existing corpora were rendered by the Mac build. A renamed or
  missing parameter would silently change the patch the reward is computed on. **Verified
  2026-07-30** for `full_preset-gen-vae_train` — all 103 parameters matched
  (`scripts/verify_parameter_parity.py`). Re-run it for any new corpus or Dexed build.
- `SynthRL-o` (stage 3, out-of-domain) reuses this machinery unchanged — only the corpus and synth
  differ — so this decision does not need revisiting when D-FAMILIES resolves.

**Amendment (2026-07-27) — in-process reuse built as an opt-in fast reward path.** The two revisit
conditions the original decision named are both met: stage 2 is render-bound (a fresh-process full
run measures in weeks at corpus scale, benchmarked locally at ~1.1 s/render), and the leakage is now
measured. A spike compared in-process renders against the fresh-process render of the same patch on
the reward's own `lsd`/`sc`/`mfcc` terms (12 seeded patches):

- Reusing one wrapper averages reward **7.9** against the **10.0** a faithful render scores; worst
  case is far lower, concentrated on free-running-LFO patches.
- The leak is **not resettable in-process**: `reload_graph` and `load_state(clean blob)` were
  byte-for-byte identical to no reset. Dexed's DSP state (LFO phase, S&H, noise) survives every
  reset dawdreamer exposes; only a fresh OS process clears it (this *confirms* D-REPRO empirically).
  So the original "with a state reset" framing was not achievable — the reused reward is simply
  biased, not corrected.
- The leak is fully **deterministic** (byte-identical re-runs), so reuse mode is reproducible.

Given that, `ParallelInProcessRenderBackend` was added alongside the fresh-process backend and is
selected by `rl.render_isolation` (`"process"` default, `"reuse"` opt-in). `synthrl_i_config.yaml`
sets `reuse`, because fresh-process is not viable at corpus scale. This is accepted rather than
gated on an A/B because **eval is unaffected** — the Evaluator always re-renders fresh-process, so
the reported metrics stay honest; only the *training reward* is approximate, and REINFORCE needs it
only to rank sampled patches. Fresh-process remains the faithful default for anyone who wants it,
and stays the sole eval-path renderer. Measurement: the spike is described in `docs/SYNTHRL_PORT.md`.

**Amendment (2026-08-16) — under `reuse`, stage 2 is *not* render-bound.** The first full stage-2
cluster run (job 1006799) measured **~35.3 min/epoch** (3.13 s/step, 660 steps), which the original
"dominated by rendering" consequence would attribute to the render pool. Profiling says otherwise.
Per 32-patch step, steady state:

| component | time |
|---|---|
| render 32 patches (8 workers, `reuse`) | 29 ms (~1%) |
| reward, 32 samples | 237 ms |
| sample actions | 9 ms |
| REINFORCE + backward | 128 ms |

Dexed renders 4 s of audio in ~2 ms in-process, so **raising `rl.num_render_workers` does not reduce
epoch cost**. The cost is the per-sample reward loop (serial librosa STFT/MFCC calls) and the
REINFORCE loop over the parameter heads. Re-measure with
`scripts/benchmark_render_throughput.py`. This does not change the decision — it corrects the
rationale, and it means the "render-bound" framing applies to `process` mode only. Consequence for
run planning: 200 epochs needs ~118 h, so `synthrl_i_config.yaml` truncates to 36 epochs with
`ramp_epochs` scaled to match (deviation 11 in `docs/SYNTHRL_PORT.md`).

Map and port fidelity: `docs/SYNTHRL_PORT.md`.

### D-DIVA-START — u-he Diva is the second synthesizer (LOCKED 2026-08-25)

The framework gets a second synthesizer, **u-he Diva**, before the Dexed benchmark table is
finished. Diva is subtractive / analog-modelling where Dexed is FM, so the pair spans two synthesis
paradigms rather than two instances of one — which is what turns the benchmark from a single-synth
result into a comparative one.

**This is an approved exception to D-ORDER and to the ROADMAP's Dexed-only scope**, taken by the
user. It does not reopen D-ORDER: the Dexed vertical slice is already proven end to end (Phase 4
plus every Phase 5 port), and D1 is locked, so the risk D-ORDER guarded against — a second subset
decision taken while the first one was still open — no longer exists.

**Why Diva** (feasibility survey over Surge XT, Diva, TAL-NoiseMaker and Vital, 2026-08-25):

- It is a plain VST3, so `DawDreamerRenderer` hosts it unchanged. Vital has no usable headless
  Python path.
- It has a large public preset dataset that ships **parameter vectors, not just audio** (below).
  Surge XT and TAL-NoiseMaker have no comparable dataset, and Surge's `.fxp` presets are an opaque
  plugin chunk neither renderer will load.
- It has direct precedent in the sound-matching literature (Esling et al., *Flow Synthesizer*,
  DAFx 2019 / MDPI 2020), which is the source of both the parameter map and the dataset below.
- It is installed and licensed on the user's machine.

Surge XT is no longer the planned second synth. Where earlier entries name it (D-ORDER,
D-RENDERER, D-FAMILIES) they should be read as "the second synth".

**Parameter addressing: module-qualified names.** Diva reports **2362 VST3 parameters**. 2080 are
`MIDI CC` JUCE passthroughs and one is `Program`, leaving **281 real parameters at indices 0–280** —
comparable to Dexed's 152, not to the raw 2362. The 2081 non-synthesis parameters are excluded above
the wrapper exactly as Dexed's are (D-EXCLUDED).

Diva's plugin-reported names are **not unique**: 56 names are shared by 147 parameters (six `Rate`,
six `Wet`, five `Model`, five `Feedback`, …). Taken as-is they would silently collapse 91 parameters
in any name→index map, and `ParameterSpace.__init__` would raise on the duplicates. So a Diva
parameter is addressed above the wrapper by its **module-qualified name** — `LFO1.Rate`,
`VCF1.Model` — per the D-NAMING amendment.

The module is not recoverable from the plugin. VST3 reports the bare display name only, and Diva
ships no machine-readable parameter list (checked the `.vst3` bundle Resources, the
`Locale/en.uhe-locale` strings, `NKS/u-he-Diva.xml` and the GUI `Scripts/EditorSetup.txt`); the Flow
Synthesizer authors state they established the correspondence by hand. Changing renderer does not
help: `PedalboardRenderer` only appears to report unique names because it drops the 91 colliding
parameters, while DawDreamer reports them honestly. RenderMan remains unsupported (D-RENDERER).

**The map is therefore a static, committed table**, `synth/diva/parameters.py`
(`DIVA_PARAMETER_NAMES`; position in the list *is* the plugin parameter index), never recomputed at
import time. It was derived from `code/synth/diva_params.txt` in `acids-ircam/flow_synthesizer`,
which lists all 281 as `Module: Name` for Diva ~1.4. Diva 1.4.7 inserted 16 parameters, so the
indices had drifted; realigning on the names pairs 265 exactly, and the 16 additions
(`LFO1.Polarity`, `OSC.DigitalShape2..4`, `VCF1.ShapeMix`, `Phase1.Depth`, …) sit interior to known
module blocks and take their neighbours' module. Result: 281/281, unique, no collisions.

**Validated twice, independently**: `tests/test_diva_parameters.py` checks the table against the
live plugin index by index (plugin-gated, and fails loudly if a Diva update reshuffles indices), and
the preset dataset's own `param` keys overlap the table 281/281.

**Preset source: the Flow Synthesizer Diva dataset**, `diva_raw.zip` (1.32 GB,
https://nubo.ircam.fr/index.php/s/nL3NQomqxced6eJ) — **11,218 `.npz` files**, one per preset, named
`<md5>_60_100.npz`. Each carries `param` (all 281 module-qualified names → values already normalized
to [0, 1], keyed `'VCF1: Model'` where this project's table uses `'VCF1.Model'`), `audio` (88320
float16 samples, ≈4.0 s at 22.05 kHz) and `chars` (a (10, 3) semantic-descriptor array).

Because it ships **parameters and not only audio**, the shipped audio is not used: every preset is
**re-rendered under this project's own contract** (D3 / D-REPRO / D-EVAL), exactly as the Dexed human
corpora are. Its note 60 / velocity 100 / 4.0 s / 22.05 kHz happening to match D3 is a convenience,
not a licence to reuse someone else's renders.

One thing to check on first contact with the data, before building anything large from it: the
realized distribution of every subset parameter across the 11,218 presets. If the corpus was
generated rather than collected, parameters Flow Synthesizer froze are constant in it and must be
dropped from the subset. See the deferred rule under **D-DIVA-SUBSET**.

This **supersedes** the earlier plan to parse Diva's 486 factory `.h2p` presets with `THIRD PARTY`
opt-in. Two consequences: **no `.h2p` parser is built** (the planned `synth/diva/patch.py` is
dropped), and the third-party-preset licensing question does not arise. The `.h2p` path stays
available as a fallback if the dataset turns out to be unusable.

**Scope of the Diva port.** Wrapper (`synth/diva/synth.py`), subset (`synth/diva/subset.py`,
D-DIVA-SUBSET), preset loader, and a `--synth {dexed,diva}` flag on `scripts/build_dataset.py`.
Shared machinery moves into synth-neutral modules rather than being duplicated: `_make_wrapper` in
`dataset/render_backends.py` becomes registry-backed, and the synth-agnostic half of
`dataset/dexed_preset_loader.py` moves to a common module.

`SynthRLi` is **out of scope for Diva**. It is the only registered family that renders with a live
VST inside the training loop (D-RL-RENDER, verified across `models/registry.py`), and its cost was
measured on Dexed only; every other family trains from a corpus with no plugin present (D-SELFDESC),
so a built Diva corpus is enough for them. Consequently only `InProcessRenderBackend` and
`FreshProcessRenderBackend` need to work with Diva — the two `Parallel*` backends are not validated
against it.

**Answered when the wrapper landed** (both were flagged here as not-to-be-assumed-from-Dexed, and
both were measured rather than inherited): Diva does not reproduce in-process at all, far worse than
Dexed's hidden-voice-state leak, but is bit-identical fresh-process — see **D-DIVA-RENDER**. And Diva
needs a *wider* noise workaround than Dexed's, because it writes to stdout as well as stderr;
`suppressed_stderr` moved to `synth/plugin_output.py` and gained a both-descriptor sibling.

### D-DIVA-RENDER — Diva renders fresh-process only (LOCKED 2026-08-25)

Diva corpora and evaluations render **one fresh process per render, at position 0**. The
in-process reuse path (`InProcessRenderBackend`, and the `ParallelInProcessRenderBackend` the RL
reward loop uses) is **not valid for Diva** and must refuse to run with it.

This is stronger than D-REPRO's Dexed finding. Dexed leaks a little hidden voice state between
in-process renders; Diva does not reproduce in-process at all.

**Measurements (2026-08-25, `DivaWrapper`, DawDreamer, 22050 Hz, D3 render settings, Diva 1.4.7
on Apple M5 / macOS 26.6.2).** Four consecutive renders of one identical patch through a single
wrapper, parameter state re-applied before each:

| comparison | waveform, max abs diff / peak | log-spectral distance | RMS drift |
|---|---|---|---|
| render 2 vs 1 | 1.393 | 7.72 dB | 0.24% |
| render 3 vs 1 | 1.379 | 7.97 dB | 0.24% |
| render 4 vs 1 | 1.400 | 7.85 dB | 0.18% |

No two renders agreed, and the divergence starts at sample 16. Loudness is stable while the
waveform is not, which is what per-note randomized oscillator phase / voice assignment looks like
— Diva is an analog emulation and this is a feature of it. For scale, ~7.9 dB log-spectral
distance is as large as the p90 *cross-engine* (DawDreamer vs Pedalboard) disagreement recorded
for Dexed under D-RENDERER. It is not a rounding artefact.

**Ruled out**: re-applying the full parameter state before each render (the fix that works for
Dexed); zeroing all five `OPT.*Slop` parameters; zeroing `OSC.Drift`; raising `OPT.Accuracy` to
`divine`. `VCC.MultiCore` is already `Off` in the loaded patch, so it is not the cause either.
Each was tested and each left the divergence intact.

**Amendment (2026-08-28): "loudness is stable" does not generalize.** The 0.18-0.24% RMS drift
above holds for the patch it was measured on, not for Diva in general. Re-measured over eight
uniform-sampled subset patches, consecutive in-process renders drifted **17.3%** in RMS on a loud
one (peak 20.7) and 97.8% on a quiet one. Worse, five of those eight patches rendered *digitally
silent* in-process while the same patches rendered audibly through fresh processes — so an
in-process Diva corpus would not merely be noisy, it would contain empty audio for patches that
are fine. The decision is unchanged and the case for it is stronger; only the "energy is stable
while the waveform is not" reading is retired.

**Fresh processes are bit-identical.** Three separate processes each constructing a `DivaWrapper`
and rendering the same patch produced byte-identical audio (max abs diff exactly 0.0). So the
render contract D-REPRO already mandates is sufficient for Diva as it stands; what changes is
that for Diva it is the *only* option, not the strict setting of two.

**Consequences.**

- `FreshProcessRenderBackend` / `ParallelFreshProcessRenderBackend` are the only valid Diva
  backends. The synth registry work must make choosing an in-process backend for Diva an error,
  not a silent quality loss.
  Implemented 2026-08-28: `BaseSynthesizer.supports_in_process_render` (True by default, False
  on `DivaWrapper`), checked by both in-process backends.

  **This costs almost nothing** (measured 2026-08-28, 12 random patches, 4.0 s at 22050 Hz,
  10 workers, per render):

  | synth | construct | in-process | fresh | parallel fresh | fresh / in-process |
  |---|---|---|---|---|---|
  | Dexed | 0.024 s | 0.004 s | 0.121 s | 0.031 s | 28.3x |
  | Diva | 0.038 s | 0.138 s | 0.279 s | 0.058 s | **2.0x** |

  Diva's own render dominates its cost, so process isolation is a 2x tax rather than Dexed's
  28x, where a 4 ms render is swamped by process overhead. At 0.058 s per render the full
  11,218-preset corpus is ~11 minutes, so throughput is not a reason to revisit this.
- `SynthRLi` stays out of scope for Diva (already recorded under D-DIVA-START). Its fast reward
  path is in-process reuse, which Diva cannot use, so it would be forced onto the slow path.
- Nothing changes for training or evaluation of the other families: they consume a built corpus
  and never render (D-SELFDESC), and the Evaluator already re-renders fresh-process (D-EVAL).
- **`ParallelFreshProcessRenderBackend.render_batch` was not delivering a fresh process per
  render** (found while measuring D-DIVA-SUBSET; **fixed 2026-08-28**). It called `Pool.map`
  with the default chunksize, which packs `ceil(n / (4 * workers))` payloads into one worker
  *task*; `maxtasksperchild=1` retires a worker after one task, not after one render, so every
  render past the first in a chunk was an in-process render. It bit only above `4 * num_workers`
  payloads, which is why no existing test caught it: the isolation test used 4 payloads on 2
  workers and `test_parallel_matches_serial_with_real_dexed` uses 2 on 2, both below the
  threshold and therefore chunked to 1 anyway. Symptom, with 4 workers and 53 payloads: renders
  diverged from their chunk's first render by ~8 dB LSD, exactly Diva's in-process figure above,
  following the payload's position modulo the worker count. The fix is `chunksize=1`, guarded by
  `test_isolation_survives_a_batch_longer_than_the_chunking_threshold` (9 payloads on 2 workers,
  which yields 5 distinct processes instead of 9 without it). This was a Dexed bug as much as a
  Diva one, and it mattered for the SynthRL reward path, which renders a whole batch per step —
  though no completed run is affected, because `cluster/training_configs/synthrl_i_config.yaml`
  selects `render_isolation: reuse`, a different backend. `ParallelInProcessRenderBackend` is
  deliberately left chunked: it reuses one wrapper per worker by design, so chunking costs it
  nothing.

**Also observed, cosmetic**: Diva writes a machine report, a revision banner and a long run of
`makeAutomatable` warnings to **stdout** as well as stderr on every instantiation, and it writes
a log file to `~/Desktop/Diva.log` that the host cannot redirect. `synth/plugin_output.py`
provides `suppressed_plugin_output()` (both file descriptors) alongside the existing
`suppressed_stderr()`; the desktop log file is unavoidable.

### D-DIVA-SUBSET — Diva parameter subset (LOCKED 2026-08-25)

The models estimate **237 of Diva's 281 real parameters**. The other 44 stay at the plugin's
freshly-loaded patch state and are never estimated. `synth/diva/subset.py` is the definition,
`tests/test_diva_subset.py` pins it.

This is the Diva analogue of **D1** and it takes D1's rule unchanged: keep the learnable voice, drop
what is **non-identifiable under D3** (one fixed note, C4, at fixed velocity 100, rendered to mono).
Diva's non-identifiable set is larger and much less obvious than Dexed's, so every drop below was
**measured on the live plugin** rather than argued from the manual.

| dropped | n | why |
|---|---|---|
| `main.Output`, `PCore.LED Colour` | 2 | Already hidden by the wrapper: a master gain outside the patch, and a GUI tint. |
| `OPT.*` (whole module) | 15 | `Accuracy` / `OfflineAcc` are CPU-versus-fidelity render settings that must stay fixed for a comparable corpus, and D-DIVA-RENDER's bit-identity rests on them. The five `*Slop` knobs scale pseudo-random per-voice drift: audible (`TuneSlop` 0 → 1 is worth 1.49 max-diff / 12.6 dB LSD) but the realization is a property of our render process, not of the patch. `V1Mod`..`V8Mod` are per-voice modulation offsets. |
| `Scope1.Frequency`, `Scope1.Scale` | 2 | The on-screen oscilloscope. **Bit-identical audio at every setting** (measured). |
| `VCC.*` except `Voice Stack` and `Transpose` | 12 | Voice allocation and inter-note behaviour, none of it revealable by one held note: polyphony count, keyboard mode, note priority, glide (needs a note to glide from, and **bit-identical at every setting**), pitch-bend range (no bend sent), `TuningMode` (no microtuning table loaded), `MultiCore` (a threading switch that must stay off to render reproducibly). `FineTuneCents` is the master-tune knob, the analogue of the `MASTER TUNE ADJ` that D1 drops for Dexed. |
| `ENV{1,2}.Velocity`, `ENV{1,2}.KeyFollow`, `HPF.KeyFollow`, `VCF1.KeyFollow` | 6 | D1's dropped class under Diva's names. |
| `VCA1.Pan`, `VCA1.PanModulation`, `VCA1.PanModDepth` | 3 | The render contract is mono. |
| `Rtary{1,2}.Controller`, `ARP.Direction`, `ARP.Order` | 4 | Each selects a MIDI controller or reorders held notes. Neither exists under D3: no CC, wheel or aftertouch is sent, and the arpeggiator has exactly one note. **All options of each render bit-identically** (measured). |

**On key-follow and velocity depth.** Unlike Dexed's keyboard scaling, these are not literally
inaudible at a fixed note: `VCF1.KeyFollow` 0 → 1 moves the sound by 2.84 dB LSD. They are dropped
because at one fixed note and velocity each collapses to a *constant* offset on the thing it scales,
so it is confounded with that parameter. Measured: the `VCF1.KeyFollow = 1.0` render is matched to
2.04 dB by simply moving `VCF1.Frequency` from 0.55 to 0.54, i.e. most of key-follow's whole range
is reproducible by a ~0.02 shift in cutoff. That is the many-to-one D-KIND warns about, and it is
the same reason D1 drops the Dexed equivalents.

**On pan.** Panning survives a mono render only as the pan law's level trim, and it is symmetric
about centre: pan 0.25 and pan 0.75 render **identically** (0.3338 max-diff / 3.1 dB LSD against
centre, the same figure for both), and hard-left versus hard-right differ by 0.3 dB. That is a
two-to-one map onto a level change `VCA1.Volume` already covers.

**On what is kept that looks droppable.** `VCC.Voice Stack` (unison, up to 6) survives: it is
strongly audible (up to 3.06 max-diff / 12.8 dB LSD), fully reproducible fresh-process at every
setting, and its render cost is bounded and small (0.05 s at 1 voice → 0.25 s at 6, against a
~0.2 s plugin instantiation that dominates either way). `ARP.OnOff` / `Octaves` / `Multiply` /
`Restart` survive because an arpeggiator does change a single held note audibly (`Octaves` 1 → 4 is
worth 10.7 dB LSD); only the two that permute the order of held notes are dropped.

**Kind (D-KIND).** Kind follows the plugin's own discreteness with no exception list: all 102
stepped parameters in the subset are categorical, the other 135 continuous. This is safe for Diva
in a way it was not for Dexed, because every parameter Diva reports as stepped is a mode, model,
waveform, switch or source selector whose adjacent steps are perceptually discontinuous. Diva has
no stepped-but-smooth parameter of the 0-99 level kind that forced D1's per-parameter judgement.
One consequence worth naming: `VCC.Transpose` is **categorical (49)** whereas the Dexed `TRANSPOSE`
is continuous. That is not a change of rule. Dexed's VST3 does not report `TRANSPOSE` as discrete
at all, so the wrapper had no grid to snap to; Diva reports its 49 semitone steps, and semitone
steps are exactly the perceptual discontinuity D-KIND makes categorical.

**Resulting ML-side vector**: 135 continuous + 102 one-hot blocks = **1100 dimensions**, against
Dexed's 103 parameters / 333 dimensions. Diva's space is roughly 3.3x wider, which is the honest
cost of a second synth of a different type and is itself a result worth reporting.

**Comparison point, Flow Synthesizer** (Esling et al.), now read off rather than assumed. Their
repo carries their estimated sets as `code/synth/params/{16,32,64,128}contparams.txt`. The name
says it: they estimate **continuous parameters only**. Their largest set is 128 of Diva's 164
continuous parameters and contains **none** of the 117 stepped ones, so `OSC.Model`, `VCF1.Model`,
`LFO*.Waveform` and every switch is frozen. Their 64- and 128-parameter sets *do* include
`ENV{1,2}.KeyFollow`, `ENV{1,2}.Velocity` and `VCF1.KeyFollow`, which this subset drops for the
reason above. Their 281 names match `synth/diva/parameters.py` exactly after `': '` → `'.'`, which
is a third independent confirmation of that table.

In particular **they never estimate a modulation route**. All 27 source selectors are constants of
their experiment, fixed per dataset in `code/synth/synthesize.py`:

| their config | used for | live routes |
|---|---|---|
| `param_default_32.json` | the real datasets | `OSC.Tune1ModSrc=Env2`, `OSC.ShapeSrc=LFO2`, `VCF1.FreqModSrc=Env2`, `VCF1.FreqMod2Src=LFO2`, `VCF1.ShapeModSrc=LFO2`; the other 22 `none` |
| `param_nomod.json` | the `toy` dataset | all `none` except `HPF.FreqModSrc=Env2` |

So their task is *fixed modulation topology, estimate the depths and knobs through it*: every preset
shares one wiring and the model never infers routing. This framework does **not** follow them, for
two reasons. `ParameterSpace` supports categoricals natively (D-KIND), so the stepped parameters
cost nothing structurally. And the benchmark is a comparison *across synths*: D1 estimates Dexed's
`ALGORITHM`, 32 one-hot dimensions that are precisely the DX7's modulation topology, and Diva's
source selectors are its direct analogue. Freezing Diva's routing while estimating Dexed's would
put a structural asymmetry into the one thing the benchmark measures, and would flatter Diva's
numbers for a reason unrelated to the model families. Their choice suits their research question
(latent control of a fixed patch topology), not ours.

**Deferred to corpus evidence: the modulation-source selectors.** 26 of the 237 kept parameters are
24-option source selectors (27 exist; `VCA1.PanModulation` is dropped with the rest of pan), so they
alone account for **624 of the 1100** ML dimensions. Some of those 24 options cannot be told apart
under D3 — five name MIDI controllers that are never sent — which means irreducible classification
error and a wide output layer spent on it. Restricting the option lists is mechanically supported:
`ParameterSpecification.options` is an explicit list, so a categorical may carry a subset of the
plugin's grid.

It is **not** restricted here, and the reason is that the only evidence available at this point was
bad. The measured collapse partitions the 24 options cleanly by the **parity of their index**, which
is not a semantic explanation, and the derived-source options (`Rectify`, `Invert`, `Adder`, ...)
measured as inert only because the probe had forced the `MOD.*` sources to `none`. Locking an option
list on an unexplained mechanism is worse than carrying the full grid, which is also what D1 does
for `ALGORITHM` (32) and `F COARSE` (32).

**The rule to apply instead**, once the preset corpus is in hand (it is not yet downloaded) and
**before any large Diva corpus is built**: compute the realized distribution of every subset
parameter across all 11,218 presets, then

- **zero variance across the corpus → drop the parameter.** It is a constant of the preset source,
  not something a model can learn or be scored on.
- **categorical whose realized options are a strict subset → restrict `options` to the realized
  set.**

That is mechanical, synth-agnostic, catches dead parameters outside the modulation section too, and
rests on a defensible statement ("the corpus never uses these") rather than on a parity coincidence.

**Why this may matter a lot here.** 11,218 is far more than any Diva preset bank ships, which
suggests the corpus is generated rather than collected. If it was generated under Flow Synthesizer's
`param_default_32.json` (see above), the routing is constant across every preset and the rule strips
most of those 624 dimensions automatically. If instead they are genuine user presets with varied
routing, the full grids stay and the Dexed-symmetry argument above carries. The data decides.

**The data decided (2026-08-28).** `diva_raw.zip` was downloaded and every one of its 11,217
presets read (`dataset/diva_preset_loader.py`; the archive holds 11,217 `.npz` files, not the
11,218 quoted above). Its `param` keys are Diva's own module-qualified names written
`'VCF1: Model'`, and translating that separator to `.` maps **all 281 onto
`synth/diva/parameters.py` exactly**, nothing left over on either side. Values are already in the
plugin's `[0, 1]` scale and every discrete parameter lands exactly on its option grid.

The variance result is unambiguous:

| | count |
|---|---|
| parameters in the corpus | 281 |
| parameters that vary at all | **64** |
| parameters constant across all 11,217 presets | 217 |
| D-DIVA-SUBSET parameters that vary | **58** of 237 |
| D-DIVA-SUBSET parameters that are constant | **179** of 237 |

The 64 varying names are exactly Flow Synthesizer's own
`code/synth/params/64contparams.txt`, and all 217 constants agree to 1e-5 with every value in
`code/synth/param_default_32.json`. So the corpus is real presets **projected onto Flow
Synthesizer's 64 continuous parameters and re-seated on their fixed base patch**. Not one
categorical parameter varies, which confirms from the data what the routing table above inferred
from their code: they estimate continuous parameters only.

The six parameters that vary in the corpus but are *not* in D-DIVA-SUBSET are
`ENV1.KeyFollow`, `ENV1.Velocity`, `ENV2.KeyFollow`, `ENV2.Velocity`, `VCF1.KeyFollow` and
`VCA1.PanModDepth` — precisely the key-follow, velocity-depth and pan parameters dropped above as
non-identifiable under D3. Flow Synthesizer estimates them because it does not hold note and
velocity fixed the way D3 does.

**Resolution (2026-08-28): build on `diva_raw` and let the rule narrow the corpus.** The rule is
applied mechanically, so the Diva human corpus estimates the **58 continuous parameters the corpus
actually varies**, ML dimension 58 (down from 1100), and no categoricals. The alternative, parsing
the 561 factory `.h2p` presets, was priced and rejected for now: their keys are abbreviated
(`Freq`, `KeyScl`, `SkRev`) and do not match the VST3 names, a file carries 341 keys against the
plugin's 281 so positional alignment drifts inside a module, and the values are in real-world units
needing a per-parameter inversion built from `get_parameter_text` sweeps. That is a hand-verified
281-entry translation table plus a value inversion, for 4% of the preset count. See the follow-up
below.

**D-DIVA-SUBSET's 237-parameter list is unchanged.** It stays the synth's canonical subset. What
narrows is the **corpus**, not the synth: `ParameterSpace.restrict` and
`dataset.preset_loader_common.restrict_to_realized` apply the rule to a loaded preset set, and
`DatasetBuilder` takes the resulting space through its `parameter_space` argument. Each corpus
serializes its own space into `run_summary.json`, which D-SELFDESC already required, so two Diva
corpora with different spaces coexist. Models trained on one are comparable only within it.

**The dropped parameters are locked at the corpus' base patch, not Diva's init patch.** 49 of the
217 constants disagree with the freshly-loaded Diva patch, so locking at the wrapper's defaults
would render the presets on a different instrument from the one they were written for. Measured on
one preset: peak 0.185 on the corpus base patch against 0.117 on the init patch. `DatasetBuilder`
therefore also takes `default_params`, which `restrict_to_realized` returns alongside the narrowed
space.

**Three module groups keep this framework's defaults instead**, all outside the 237-parameter
subset already: `OPT` (Accuracy and the five slop knobs -- render-quality settings, and
D-DIVA-RENDER's bit-identity was measured at our values, not at the corpus' draft-quality
`Accuracy=0`), `Scope1` (the oscilloscope, measured bit-identical at every setting), and `VCC`
(voice allocation, glide and pitch-bend range, non-identifiable under D3 by construction). This
means our renders are not sample-comparable with the audio shipped in `diva_raw`, which was already
true: every sound is re-rendered under D3 regardless.

**Follow-up: a second, synthetic Diva corpus over the full 237.** `SyntheticPresetSource` already
takes the wrapper's own space, so a uniform-sampled Diva corpus needs no new machinery and would
exercise the 102 categoricals and the one-hot path that the `diva_raw` corpus leaves untouched.
Not scheduled.

What it needs first is **not** what Dexed needed. 100 uniform draws over the 237-parameter subset,
rendered fresh-process under D3, came out: 0 digitally silent, 2 near-silent (peak < 0.01), 62
usable, and **36 clipping above full scale**, max peak 25.3. So Diva's uniform-sampling problem is
level, not silence — the opposite of the Dexed case D-AUDIBLE was written for, where draws collapse
to inaudible. `audible_sampling_ranges` is the wrong tool; what this needs is a level constraint or
a normalization step, plus a `min_loudness_lufs` recalibration. Measured 2026-08-28.

**Two constraints on whoever applies it.** The `ParameterSpace` is serialized into
`run_summary.json` (D-SELFDESC), so narrowing an option list invalidates any corpus already built —
which is exactly why this must land before the big build, and why keeping the grids wide now costs
nothing. And the subset is shared by the synthetic and human preset sources: whatever the rule
strips must be stripped for both, or uniform sampling would vary parameters the human corpus holds
constant.

**Measurement provenance** (2026-08-25, `DivaWrapper`, DawDreamer, 22050 Hz, D3 render settings,
Diva 1.4.7 on Apple M5 / macOS 26.6.2). Every figure above comes from fresh-process renders
(D-DIVA-RENDER), one process per render, with a control patch rendered first, mid-run and last and
required to be bit-identical all three times. That control matters: the first attempt at these
measurements used `Pool.map` without `chunksize=1`, which batches several renders into one worker
task and therefore into one process, and every parameter then appeared audible at ~8 dB LSD, which
is simply Diva's in-process divergence. See the note under **D-DIVA-RENDER**.


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

**Scope**: D4 is about **Dexed**. Diva's preset source is settled separately (D-DIVA-START: the
Flow Synthesizer 11k dataset); how that corpus splits into train and test is its own open question,
raised when the Diva loader lands.

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
D-FLOW-CORPUS) + **reinforcement learning** (SynthRL, Shin & Lee IJCAI-25 — **committed and built**:
the staged `SynthRLp` / `SynthRLi`, see `docs/SYNTHRL_PORT.md`; renders in the training loop per
D-RL-RENDER). **Evolutionary search is dropped** (user: "probably no evolutionary
algorithms"); if ever reinstated, note it runs a per-target search locally with the live VST and does
**not** fit the cluster training harness.

**Why it's open**: the neural-proxy, flow-matching and RL slots are now filled, but the final family
set is not frozen — the exact discriminative/generative architectures still evolve and a second synth
(Surge XT) may add families.

**Gated on this decision**: SynthRL's stage 3, `SynthRL-o` (RL-only fine-tuning on out-of-domain
sounds), is the paper's headline cross-domain result and is **not ported**, because it needs a second
synthesizer — the paper uses Surge XT. The RL machinery is corpus-agnostic and reused unchanged, so
porting it later is "point stage 2 at a second synth's corpus with the parameter loss off". What it
actually needs is the second-synth commitment here, plus a `BaseSynthesizer` wrapper, a corpus, and a
cluster feasibility spike for that plugin.

**Update 2026-08-25**: the second synth is now committed — **Diva** (D-DIVA-START), not Surge XT.
That does **not** unblock `SynthRL-o`. Its remaining cost is the in-training-loop render (D-RL-RENDER)
against a plugin whose throughput has never been measured, and `SynthRLi` is explicitly out of scope
for Diva for that reason; stage 3 inherits the same blocker. Whether a second synth adds *families*
is still open here.

**Blocks**: Phase 5. Resolve here before the Phase 5 family tasks start.

