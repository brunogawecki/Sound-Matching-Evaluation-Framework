# Notes for the thesis writing session

Remarks left by the **code** session for the **LaTeX writing** session. These are things the code
session wants surfaced in the thesis; they are not new decisions. Authoritative detail and the raw
numbers live in `DECISIONS.md` (D-REPRO and the D-RENDERER benchmark entries) — this file is a
reading guide, not a second source of truth.

**Topic: Dexed's hidden per-voice engine state ("context leakage").** Dexed carries hidden per-voice
state, so the *same* patch can render audibly differently depending on what was rendered before it.
**Policy (D-REPRO, locked 2026-06-17): accept and document this as a threat to validity — do not fix
it at the engine level.** The thesis should describe the phenomenon, the render discipline that keeps
it from biasing results, and cite the characterization data below. The three points below are what
Bruno explicitly wants in the write-up.

---

## 1. The leak is concentrated in S&H / LFO / noise voices — say this explicitly

The context leakage is **not uniform across patches**. Most musical pads and basses are bit-identical
across render contexts (median LSD ≈ 0). The divergence is concentrated in **LFO / sample-&-hold /
noise** voices — exactly the patch class the hidden-per-voice-state mechanism predicts (an LFO/S&H
internal value that is not reset between renders).

- **Evidence (named real presets):** the most cross-method-divergent voices in the 1056-voice
  cartridge run are overwhelmingly LFO/S&H/noise: `CIGALES` (69.68 dB LSD), `CROSSING` (32.63),
  `S-H ZIBBLE` (23.92), `COMPUTER 1` (23.53), `SCHLBELL` (22.92). Source:
  `figures/data/host_agreement_3way_cartridges.csv`.
- **Why it matters for scope (link to D1):** because the leak's footprint is this class of
  parameters, **D1** (the final Dexed subset) can shrink the problem by locking the LFO / S&H
  parameters — the same move `preset-gen-vae` made with its `prevent_SH_LFO` constraint. Worth
  framing as a deliberate, defensible scope choice rather than a workaround.

## 2. We tested candidate mitigations — describe the arms, not just the conclusion

The thesis should show this was investigated empirically, not asserted. Four rendering strategies
("arms") were compared on the same patches (`scripts/benchmark_renderers.py`,
`scripts/render_divergence_examples.py`):

| Arm | What it does | Result on the hidden state |
|---|---|---|
| **reuse** | one persistent instance renders every patch (the framework default) | carries the leak |
| **reload-per-render** | a fresh wrapper rebuilt *in-process* per render (the `preset-gen-vae` approach) | **does NOT fix it** — produces a third, equally-divergent realization |
| **pedalboard** | a different VST host (Pedalboard instead of DawDreamer) | **leaks identically** — so the state is in the shared plugin binary, not the host |
| **subprocess** | each patch rendered in a fresh **OS process** (spawn) | **the only thing that resets the state** — two independent fresh-process renders agree to ~0 |

Narrative for the thesis: in-process teardown (reload-per-render) is **insufficient** — only OS-level
process isolation resets the state, and the leak is a property of the **Dexed plugin binary**, not of
the host library. This is why the render discipline (deterministic generation; fresh-process
re-render at evaluation) is what neutralizes the bias, rather than an engine patch.

## 3. Graphs / tables that prove the leak is real

Data already exists for all of these (under `figures/data/`); Bruno styles the actual figures. Each
item below names the claim, the source CSV, and a suggested form + caption stub.

- **Table — within-engine leakage predicts cross-engine divergence (both engines).** The patches
  that diverge most *between* engines are the same ones most context-dependent *within* one engine,
  at the same magnitude. DawDreamer: Spearman ρ = 0.62, top-decile overlap 90.8%. Pedalboard:
  ρ = 0.620, overlap 89.2%. Sources: `context_leakage_seed0.csv`,
  `context_leakage_pedalboard_seed0.csv`.
  *Caption stub:* "Within-engine context leakage vs. cross-engine divergence per patch; the tails
  coincide, and Pedalboard behaves identically to DawDreamer."

- **Figure (scatter) — the bimodal structure.** x = within-engine context-leakage LSD,
  y = cross-engine LSD, one point per patch (n = 2601). Shows a dense near-zero cluster plus a shared
  divergent tail. Same two CSVs (overlay both engines or show side by side).

- **Table — all three in-process arm-pairs share the same tail.** reuse↔pedalboard, reload↔pedalboard,
  reuse↔reload all have the same LSD p90/p95 — i.e. reload does not collapse the tail. Random patches:
  `host_agreement_3way_seed0.csv` (≈ 7.1 / 8.6 dB). Real cartridge voices:
  `host_agreement_3way_cartridges.csv` (≈ 8.9 / 11 dB). Pairs replicate across both populations.

- **Table — most-divergent real presets (the S&H/LFO story, point 1).** Voice name + LSD for the top
  divergers. Source: `host_agreement_3way_cartridges.csv` (has a `patch_label` column).

- **Figure (positive control, optional) — fresh processes are deterministic.** `subprocess_a_vs_b`
  ≈ 0 while `reuse_vs_reload` keeps a full tail — the cleanest single demonstration that the fix is
  process isolation, not in-process reload. Regenerate with
  `python scripts/benchmark_renderers.py --subprocess --dump-agreement-csv <path>`.

- **Listenable examples (optional appendix / supplementary material).** Side-by-side WAVs of a
  sensitive patch rendered through each arm, including the clean subprocess reference. Not committed;
  regenerate with `python scripts/render_divergence_examples.py [--cartridges]` (writes to
  `dataset/audio/{,cartridge_}divergence_examples/`).

---

*Headline numbers, for quick reference (all from real runs recorded in `DECISIONS.md`):*
*per-render speed reuse 3.4 ms / reload 30.8 ms (~9× slower) / pedalboard 18.2 ms;*
*within-engine leakage p90/p95 ≈ 6.9 / 8.5 dB (DawDreamer) and 7.1 / 8.5 dB (Pedalboard);*
*0/1056 cartridge voices near-silent vs ~13% of uniform-random subset draws.*

---

**Topic: the preset-gen-vae port — what the Implementation chapter should say.** The generative
family is a port of Le Vaillant et al. (DAFx 2021), implemented in `models/presetgen_vae/` as two
registered families (`PresetGenVAEMLPRegressor` / `PresetGenVAEFlowRegressor` — the paper's two
reported models, differing only in the regressor head). The authoritative source for this section is
`docs/PRESETGEN_VAE_PORT.md`: the architecture explanation, the piece-by-piece code↔paper
counterpart table, and every documented deviation. Rationale lives in `DECISIONS.md` (D-MELNORM,
D-FRAMEWORK, D-METRIC-SR, D-SELFDESC). The angles below are what the write-up should not miss.

## 1. The parity-test verification method is itself thesis material

"Is the reimplementation faithful?" is usually answered by assertion. Here it is answered by test:
`tests/test_paper_parity.py` builds each network component from *both* codebases, transplants the
paper's randomly-initialized weights into ours, and asserts numerically equal outputs on identical
inputs (flows additionally checked on their log-determinants, in train and eval mode). Worth a
paragraph in Implementation (or a methodology aside): it turns port fidelity from a claim into a
reproducible result, and it delimits exactly which parts are proven equal vs. documented-different
(the mel front-end is documented, not parity-tested — see `PRESETGEN_VAE_PORT.md` for why).

## 2. The dead-code finding — why verification against the *executed* code path matters

The paper's published `'speccnn8l1_bn'` architecture listing ends in a 1024-wide bottleneck, but
that layer is dead code under the paper's own shipped config: the composed encoder actually runs a
2048-wide 1×1 mixer. Our port initially copied the listing and was wrong in exactly the way a
faithful-looking port can be; the review caught and fixed it (details and file/line references in
`PRESETGEN_VAE_PORT.md`, "Deviations found in review"). Good discussion material: reproducing a
paper means reproducing what its code *ran*, not what its code *lists*. A second, smaller instance:
the paper's declared mel band edges (30–11000 Hz) are marked TODO in its code and never applied.

## 3. Intentional deviations — we reproduce the method, not the numbers

State up front that reproduced numbers will not match the paper's tables, by design: 103 Dexed
parameters (D1) instead of 144; our own categorical scheme (close to NumCat, not NumCat++); a
single train/test split, not 5-fold; dawdreamer rendering, not RenderMan; no useless-parameter
masking in the controls loss. All listed with rationale in `PRESETGEN_VAE_PORT.md` (Caveats +
Deviations). The benchmark needs all families on one common protocol; per-paper protocol quirks are
deliberately normalized away.

## 4. The front-end moved inside the network — a framework-driven design choice

The paper precomputes mel-dB spectrograms offline with dataset-wide min-max stats. Our corpora
store raw audio and must stay self-describing (D-SELFDESC), so the same STFT → mel → dB → min-max
math runs inside the network, with the normalization endpoints measured from the training corpus at
fit time (D-MELNORM) — the exact analogue of the paper's `spec_stats` pass. Nice illustration of
how a standardized benchmark reshapes a per-paper pipeline without changing its math.
