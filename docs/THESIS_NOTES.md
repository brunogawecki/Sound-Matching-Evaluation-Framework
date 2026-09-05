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

---

**Topic: the flow-matching port — what the Implementation chapter should say, and one threat to
validity.** The conditional-generative family is a port of Hayes, Saitis & Fazekas (ISMIR 2025),
implemented in `models/flow_matching/` as two registered families (`FlowMatchingMLP` /
`FlowMatchingParam2Tok` — the paper's own control and its equivariant model, differing only in the
vector field). The authoritative source is `docs/FLOW_MATCHING_PORT.md`: the architecture
explanation, the code↔paper counterpart table, and every documented deviation. Rationale lives in
`DECISIONS.md` (D-FLOW-CORPUS, D-FLOW-PREDICT, D-MELNORM, D-REPR). The four angles below are what
the write-up should not miss. **Note that as of this writing neither family has been trained — there
are no numbers yet, only method and implementation.**

## 1. The training corpus is part of the method — and it carries a documented confound

Every other family trains on human presets. This one trains on a **synthetic-uniform** corpus
(D-FLOW-CORPUS), because the paper's equivariance argument assumes a **G-invariant parameter
prior**: one that respects the synth's operator-permutation symmetry. Uniform sampling gives that;
curated human presets break it. Train Param2Tok on human presets and the reason it should win has
been discarded. This is not a data-plumbing detail, it is a premise of the experiment, and the
Methodology chapter should present it as one.

**The threat to validity:** the corpus we can actually build is only *approximately* G-invariant.
`scripts/build_dataset.py synthetic` always applies the D-AUDIBLE range overrides, which pin three
**OP1** parameters so that draws are audible. Naming one operator makes the prior non-invariant
under operator permutation — a partial break of exactly the property the family depends on. It is
3 parameters of 103 and 1 operator of 6, which is why it was accepted, but it must be stated. The
consequence for the results chapter: **if Param2Tok fails to separate from the MLP control, that is
not evidence against the paper's premise until this confound is ruled out.** D-FLOW-CORPUS records
the two rejected alternatives and their costs (unconstrained sampling rejects ~94% of draws under
D-SILENCE; the principled fix needs a DX7 algorithm→carrier table we do not have).

## 2. The MLP variant is a control, not a baseline — say so, or the result means nothing

`FlowMatchingMLP` is easy to misread as a weak floor. It is not: it is the paper's own control, and
the **MLP↔Param2Tok gap is the quantity the port exists to measure**. Both share the corpus, the
encoder, the loss, and the sampler. Reporting Param2Tok's absolute score without the gap says
nothing about symmetry, which is the entire contribution being tested. Worth framing explicitly,
since the framework's other families genuinely *are* arranged as baselines-vs-approaches.

## 3. `predict` returns one seeded sample — the generative variance is unmeasured

This is the framework's first true **sampler**: `predict` integrates a learned ODE (200 RK4 steps,
CFG strength 2.0) rather than doing a forward pass, so `BaseFlowMatchingModel` overrides the base
`predict`. The draw is seeded per call (D-FLOW-PREDICT) so the Evaluator's re-render stays
reproducible and the family sits in the same results table as the regressors.

The honest limitation: a generative model has a per-target *distribution* of solutions, and one
seeded draw does not measure its spread. Two runs of the same model could land at different table
positions by sampling luck alone. Best-of-N and per-target sample statistics are deferred, and the
limitations section should name this rather than let a single-draw number read as the model's
capability. Related cost note, if run time is discussed: at 200 steps × 2 field evaluations,
Param2Tok measured ~3.7× slower per sample than the MLP variant on CPU (13.9 s vs 3.8 s).

## 4. No parity tests here — unlike preset-gen-vae, and for a reason worth one sentence

Point 1 of the preset-gen-vae topic above makes the weight-transplant parity tests thesis material.
This port has none, and the contrast should be explained rather than left as an apparent
inconsistency: the reference ships **no trained checkpoints**, and its task is **Surge XT**, not
Dexed. Fidelity is instead established by behavioral test (`tests/test_flow_matching.py`) — RK4
checked against a closed-form ODE, and the permutation-equivariance property asserted directly on
`DiffusionTransformerBlock` (permuting the tokens permutes the output identically). Choosing the
verification method that the available artifacts permit is itself a defensible methodological
point.

*(Also for the Implementation chapter: the "AST" row in the paper's Table 1 is a separate
discriminative model (Bruford et al. DAFx24) and is **deliberately not ported** — discriminative
coverage already exists via `sound2synth` and `inversynth2`. Mind the name collision: our
`AudioSpectrogramTransformer` is the flow's conditioning encoder, not that baseline.)*

---

**Topic: the SynthRL port — what the Implementation chapter should say, and one threat to validity
that affects the whole benchmark table.** The reinforcement-learning family is a port of Shin & Lee
(IJCAI-25), implemented in `models/synthrl/` as two registered families (`SynthRLp` / `SynthRLi` —
stages 1 and 2 of the paper's three). The authoritative source is `docs/SYNTHRL_PORT.md`: the
architecture, the code↔paper counterpart table, and all eleven documented deviations. Rationale
lives in `DECISIONS.md` (**D-RL-RENDER**, D-METRIC-NORM, D-KIND, D-MELNORM, D-REPRO). The five
angles below are what the write-up should not miss. **As of this writing `SynthRLp` has one
completed cluster run but no evaluation, and `SynthRLi` has not completed a run at all — there are
no numbers yet, only method and implementation.**

## 1. The RL reward is built from the evaluation panel's own metrics — say this, it is a confound

The reward is `1 / (w₁·LSD + w₂·SC + w₃·MFCC)`, computed with the framework's own metric callables
(`lsd`, `spectral_convergence`, `mfcc_mae`). As an implementation fact this is a virtue and worth
one sentence: the training signal and the results table measure similarity through one shared
definition rather than two drifting ones.

**The consequence is the problem.** `SynthRLi` is the only family in the benchmark that directly
optimizes quantities the panel scores it on. Every other family optimizes a parameter loss, an ELBO,
or a flow-matching objective, and is then measured on audio similarity as an *independent* test.
SynthRL is measured on part of its own objective. A strong `SynthRLi` row on `spectral_convergence`
or `mfcc_mae` is therefore not directly comparable to the other families' rows on those same
metrics.

This is not a reason to change the reward — it *is* the paper's method, and changing it would stop
being a port. It is a reason to state the asymmetry wherever the comparative table is discussed, and
to lean on the metric axes the reward does **not** contain (the parameter, loudness, and pitch axes)
when arguing that `SynthRLi` genuinely improved rather than gamed three numbers. The
`SynthRLp` → `SynthRLi` delta stays interpretable throughout, since both are scored the same way.

## 2. We port the in-domain half of a cross-domain paper — do not oversell the contribution

The paper is titled *Cross-domain* Synthesizer Sound Matching, and its headline claim is stage 3
(`SynthRL-o`): RL-only fine-tuning on sounds from a **different synthesizer**, which is what removes
the need for ground-truth parameters. That stage is **deferred**, because it needs the second synth
and D-FAMILIES is open.

So the thesis tests the paper's machinery, not the paper's main claim. Be explicit about it. The
framing that holds up: stages 1 and 2 establish that the method works in-domain on Dexed, and the
cross-domain claim is out of scope for this benchmark rather than refuted by it. Note also that
nothing in the port blocks stage 3 — it is stage 2's recipe with the parameter loss switched off and
a different corpus — so it is a scope decision, not a technical limitation.

## 3. The staging is the experiment, and it parallels InverSynth II

`SynthRLi` warm-starts from a finished `SynthRLp` checkpoint through the generic `--init-from` hook.
Same network, same corpus, same evaluation — the only difference is that stage 2 adds the RL
objective and ramps the parameter loss out. The `SynthRLp` → `SynthRLi` difference therefore
isolates the contribution of reinforcement learning, cleanly, and it is the number this family
exists to produce.

Worth pointing at the `IS` → `IS2xITF` → `IS2` staging in the same chapter rather than re-explaining
the idea: two of the five families are staged ports where the intermediate stage is a registered,
separately-evaluated model. That is a property of how this benchmark was built, and it is more
informative than either family's absolute score.

## 4. The context-leakage story from topic 1 resurfaces here, and was measured a second time

Stage 2 renders every sampled patch with the live Dexed **inside the training loop** (D-RL-RENDER) —
the only family that does, and a deliberate, scoped deviation from the self-describing-corpus rule.
Fresh-process rendering is too slow at corpus scale, so training reuses one plugin instance per
worker, which walks straight into the hidden per-voice state documented at the top of this file.

It was quantified again in this new setting: a reused instance computes an average reward of **7.9**
where a faithful fresh-process render of the same patch scores **10.0**, worst case far lower and
concentrated on free-running-LFO patches. The leak is fully deterministic, so training stays
reproducible, and `reload_graph` / `load_state` were byte-for-byte identical to no reset at all —
independently confirming the topic-1 finding that only OS-level process isolation clears the state.

Two things the write-up must pair with that number, or it reads as a broken experiment:

- **Evaluation is unaffected.** The Evaluator always re-renders fresh-process (D-EVAL / D-REPRO), so
  every reported metric is computed on clean audio. Only the *training reward* is biased.
- **REINFORCE only needs the reward to rank patches**, not to be calibrated. A deterministic,
  monotone-ish distortion of the reward is a far weaker requirement than an accurate one.

This is a good discussion beat: the same plugin defect that topic 1 frames as a threat to validity
turns out to be tolerable in one specific place, for a stated reason. That is a more interesting
claim than either "it does not matter" or "it invalidates the run".

## 5. The truncated run understates the method — and the reason is not what it looks like

The paper runs 200 epochs per stage. Stage 2 here runs **36**, with the curriculum ramp scaled from
100 down to 18 so the paper's half-ramp-then-RL-only shape survives the truncation. The reason is
measured: **~35.3 min/epoch** on an A100 (job 1006799), so 200 epochs needs ~118 h. A first attempt
at the full 200 hit the job wall-time at epoch 41 and exported nothing.

The reward was **still climbing** when the run was truncated (`val_reward` 1.337 → 1.413). State
that wherever a `SynthRLi` number appears: this configuration is a lower bound on the method, not a
measurement of it, and a weak result is under-training before it is evidence about the approach.

**Do not write that rendering is the bottleneck.** It is the intuitive explanation and it is wrong —
rendering is ~1% of a training step under the reuse backend; the cost is the per-sample reward
computation and the REINFORCE pass over the ~100 parameter heads. The earlier "render-bound" framing
in the docs was corrected on 2026-08-16 (D-RL-RENDER amendment) and applies only to the
fresh-process mode. This is worth a sentence in its own right if the thesis discusses engineering
cost: the expensive part of RL-based sound matching here is *scoring* the audio, not *making* it.

*(One structural note for the Implementation chapter: this family is the only one that treats every
synthesis parameter as a **classification** problem rather than a regression — numerical parameters
are discretized onto 25 ordinal levels with Gaussian-smoothed targets. It wraps the shared
`ParameterSpace` and never modifies it, so the class-index view stays private to this family and
D-KIND is untouched. Unlike the preset-gen-vae port there are no weight-transplant parity tests: the
reference is AGPL-3.0 and deliberately not vendored, so fidelity rests on the counterpart table in
`SYNTHRL_PORT.md` plus behavioral tests — the same "verify with the artifacts you actually have"
argument made for flow matching above.)*

---

**Topic: out-of-domain evaluation on NSynth.** The benchmark gains a second axis whose targets are
real instrument recordings no synthesizer produced (**D-OOD**, locked 2026-09-04). It is the most
interesting evaluation result the framework can produce, and also the easiest to overstate, so the
five points below are what the write-up has to get right.

## 1. The parameter axis vanishes, and that is the argument, not an inconvenience

Out-of-domain there is no ground-truth parameter vector, so `param_mae` / `param_mse` /
`param_accuracy` are undefined and reported as `NaN` with `valid_count: 0`. Ten of the thirteen
metrics survive.

This is worth a paragraph rather than a footnote, because it is the thesis's own claim made
concrete: perceptual audio similarity is the primary axis and parameter distance is a secondary
diagnostic. In-domain that ordering is an argument. Out-of-domain it is forced — the parameter
metrics cannot be computed at all, and the benchmark still works. Point at the `valid_count: 0`
column rather than quietly omitting three rows from the table.

## 2. The error floor is not zero, so the two tables are not on the same scale

In-domain, a perfect prediction floors the audio metrics at ~0, because the target was rendered by
the same synth under the same contract (D-EVAL point 3, verified by a plugin test). Out-of-domain
that guarantee is gone: an NSynth flute is generally unreachable by Dexed, so every score carries an
unknown, target-dependent offset for "how close can this synth get *at all*".

The baseline run already demonstrates the trap. `MeanParameterBaseline` scores **better**
out-of-domain than in-domain on several metrics — `f0_rmse` 126 → 74 on Dexed and 155 → 40 on Diva,
`lsd` 1.13 → 0.89 and 1.04 → 0.81. That is not "real instruments are easier". NSynth notes are all
cleanly pitched at C4 by construction, while the preset test sets are full of noise, inharmonic FM
and percussive voices with no stable f0, so the pitch metric is reading a property of the target
population. In the other direction `integrated_loudness_error` rises ~9 dB, tracking the measured
+9.4 dB loudness offset almost exactly. Both effects are constant across models, so rankings inside
the OOD table are unaffected — but a reader shown both tables together will draw the wrong
conclusion unless told.

Consequence: **do not put in-domain and out-of-domain numbers side by side as if comparable.**
Within the OOD table the offset is shared by every model, so rankings and per-model differences are
meaningful. Across tables, absolute values are not. The genuinely interesting result is whether the
in-domain *ranking* survives — if it does, the benchmark is measuring something that generalizes; if
it inverts, that is a finding about which families overfit the preset distribution.

## 3. We use NSynth's train split, and the reason is not laziness

Each NSynth instrument contributes roughly one note per pitch/velocity pair, and D3 pins us to one
pitch and one velocity, so the `valid` + `test` splits together yield only **46** notes. All three
splits yield ~836.

The obvious objection — that the train split is off-limits — does not apply. That boundary exists to
stop instrument leakage between training *on NSynth* and evaluating *on NSynth*. No family in this
benchmark ever sees NSynth: they learn from Dexed and Diva presets. There is no leakage channel, so
the split is a constraint belonging to a different experiment. State this explicitly, because a
reader will flag it otherwise.

## 4. Two mismatches to disclose, both of which cancel in the ranking

- **Band limit.** NSynth is 16 kHz, so its targets carry nothing above 8 kHz, against the render's
  11.025 kHz ceiling. Targets are upsampled once at build time with a deterministic anti-aliased
  resampler; the upsample recovers nothing, it only supplies the rate the panel expects. This is the
  same shape of argument D-METRIC-SR already makes: an equal band-limit on every model caps absolute
  fidelity without biasing ranking.
- **Loudness offset.** NSynth's level convention is unrelated to this project's renders, and audio
  metrics compare raw audio (D-METRIC-NORM), so the two loudness metrics carry a corpus-level offset
  out-of-domain. `scripts/build_ood_corpus.py` prints the measured offset against the reference
  corpus — quote that number rather than describing the problem qualitatively.

Both are equal across models, so neither touches the comparison. Both cap absolute interpretation.

## 5. This is the evaluation half of SynthRL's deferred stage 3 — say which half

SynthRL's headline claim is `SynthRL-o`: RL fine-tuning **on** out-of-domain sounds, which is what
removes the need for ground-truth parameters. That stage is not ported (D-FAMILIES).

What this axis adds is the *evaluation* setting, not the *training* one: in-domain-trained models,
scored on out-of-domain targets. It is a weaker claim than the paper's and must not be presented as
a replication of it. But it is the same question one step back — does a model trained on presets
transfer to real sounds — and it is answerable with no new training, which is why it is worth having.
Pair it with the note under the SynthRL topic that stages 1 and 2 test the machinery rather than the
main claim; together they make the scope of the RL contribution honest and legible.
