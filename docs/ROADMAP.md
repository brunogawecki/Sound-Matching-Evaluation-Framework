# Roadmap — Dexed pipeline → first comparative benchmark

This is the **decomposition** of the work between today's state (a corpus, a model interface, a
trivial baseline, and a working Evaluator) and the project's goal (a comparative benchmark across
model families on Dexed). It is high-level on purpose: each task below gets its own detailed-design
session later. The benchmark path is **Dexed-only** (D-ORDER) — the full pipeline had to be proven on
Dexed before any second synth — and the roadmap **ends at "benchmark results produced."** Thesis
prose/figures and the #12 dashboard (built — see `ARCHITECTURE.md`, but tangential to the benchmark
path) are out of scope.

**A second synth is now in scope, in parallel** — see "Diva" below. It runs alongside Phase 6 rather
than replacing it; the benchmark table is still Dexed.

The split with the rest of `docs/` is the usual one: **this file owns the *decomposition and
ordering*; `DECISIONS.md` owns the *why*; GitHub issues own the *do*.** Open decisions resolve in
`DECISIONS.md`, never as issues — only the work they unblock becomes an issue.

## Where we are

Built and run end-to-end: Layer 2 (data), Layer 3 (models + training), and Layer 4 (evaluation).
**Phase 4 is complete** — the Lightning training harness (#28), cluster packaging (#20, D-CLUSTER),
the preset-gen-vae training corpus (SQLite loader + `full_preset-gen-vae*` corpora), and the basic
Sound2Synth regressor (#19/#31), with the exit criterion met: models train on the PUT cluster,
checkpoints come down, and the Evaluator has scored **three deep families** on a held-out corpus
(`results/dexed_builtin_test/`). Phase 5's **generative family is also done**: the preset-gen-vae
port (#23/#35/#36) — full VAE with latent RealNVP flow, two registered families
(`PresetGenVAEMLPRegressor` / `PresetGenVAEFlowRegressor`), parity-tested against the paper's code
(`docs/PRESETGEN_VAE_PORT.md`). Full preset-gen-vae training runs are in flight on the cluster.

What does **not** exist yet:

- **No trained `SynthRLi`.** Stage 1 (`SynthRLp`) has a completed cluster run; stage 2 has not
  finished one, so the RL family has no evaluated result yet.
- **No fuller Sound2Synth architecture.** The landed model is a single-spectrogram-branch first cut;
  the paper's multi-modal encoder + grouped-FC parameter classifier is still future work.
- **No final human test set (D4), no benchmark orchestration, no benchmark table** — Phase 6. The
  existing results rows are pipeline shakedowns, not benchmark numbers.

## Diva (second synth)

**u-he Diva is the second synthesizer** (D-DIVA-START, LOCKED 2026-08-25), an approved exception to
D-ORDER's ordering and to this file's former Dexed-only scope. Surge XT is no longer the plan. Diva
is subtractive where Dexed is FM, it hosts as a plain VST3, and it has an 11,218-preset public
dataset that ships parameter vectors (Flow Synthesizer, Esling et al.), so a human Diva corpus can be
re-rendered under this project's own contract.

Landed so far: `synth/diva/parameters.py`, the committed 281-parameter module-qualified name table,
plus its plugin-gated test. Diva's plugin names are not unique, so parameters are addressed
module-qualified (`VCF1.Model`) — see the D-NAMING amendment.

`DivaWrapper` is built: 279 exposed parameters (281 minus master output and the GUI LED tint),
continuous/discrete split read off the plugin and frozen, DawDreamer-only. Rendering Diva is
**fresh-process only** — it does not reproduce in-process at all (**D-DIVA-RENDER**).

The estimated subset is settled: **237 of the 281** parameters (**D-DIVA-SUBSET**), a 1100-dimension
ML-side vector against Dexed's 333.

The render layer and the preset loaders are synth-neutral (`dataset/render_backends.py` carries a
synth registry, `dataset/preset_loader_common.py` holds the shared dedup/split half), and
`dataset/diva_preset_loader.py` reads the 11,217-preset Flow Synthesizer corpus.

That corpus varies only 64 of Diva's 281 parameters and no categoricals at all -- the paper kept
continuous parameters only and fixed the rest. So the **Diva human corpus estimates 58 parameters,
not 237**: D-DIVA-SUBSET's list is unchanged, but the corpus-variance rule narrows the space per
corpus (`restrict_to_realized`). A second, synthetic Diva corpus over the full 237 is the planned
way to exercise the categoricals; it is not scheduled.

Still to build: a `--synth {dexed,diva}` flag on `scripts/build_dataset.py`. `SynthRLi` is out of
scope for Diva (it is the only family that renders inside the training loop, D-RL-RENDER).

## Sequencing — vertical slice first

Stand up the training framework **and** cluster packaging by driving them end-to-end with a single
discriminative model, before building the other families on the proven foundation. This de-risks the
unknowns (orchestration, packaging, cluster I/O) once, against the lowest-risk architecture, rather
than discovering them family-by-family. It mirrors D-ORDER one level down.

## Gating decisions (resolve in `DECISIONS.md`, not as issues)

| Decision | Status | Blocks | Note |
|---|---|---|---|
| **D-FRAMEWORK** — PyTorch Lightning vs. raw PyTorch loop | LOCKED (Lightning) | — (unblocked) | Locked 2026-06-30; conventions for the harness recorded in `DECISIONS.md`. |
| **D-FAMILIES** — final model-family set | OPEN (stub) | Phase 5 | Discriminative + generative (primary) + neural-proxy (baseline); evolutionary dropped. |
| **D4** — human test-set composition | OPEN | Phase 6 | Importer built; Phase 4 has landed, so the final split is unblocked and awaits the user's call. |

## Phase 4 — Training foundation, proven by one real model

Goal: a real (non-trivial) results row, produced by training a discriminative model on the cluster
and scoring it through the existing Evaluator.

- **Training harness** — config system, train/val loop, logging, checkpoint convention consumable by
  `BaseModel.load`, seeding/reproducibility. *(Gated by D-FRAMEWORK.)* **DONE** (#28): PyTorch
  Lightning harness under `models/training/`.
- **Discriminative parameter regressor** — first real model family (spectrogram→params, the
  InverSynth / preset-gen-vae lineage; lowest-risk architecture). First real `BaseModel.fit`. **DONE
  (basic cut)** (#19/#31): `Sound2SynthSpectrogramRegressor` — a VGG11-BN log-power-STFT net with a
  plain MLP head. The fuller paper architecture (multi-modal encoder + grouped-FC classifier) is
  deferred to a later sub-project.
- **Cluster packaging** — dependency split (cluster requirements **without** VST/dawdreamer, per
  D-SELFDESC), environment/container spec, job-submission scripts, corpus-up / checkpoint-down
  transfer, entrypoint. **DONE** (#20): `requirements-cluster.txt` finalized as the complete VST-free
  split, plus `cluster/` (sbatch job, `cluster.env` template, smoke config, `push_corpus.sh` /
  `pull_checkpoint.sh`, README walkthrough) for the PUT SLURM cluster. No library changes — the
  harness was already SLURM-aware. See **D-CLUSTER** in `DECISIONS.md`.
- **Training corpus from preset-gen-vae** — the human DX7 collection at
  `paper_repos/preset-gen-vae/synth/dexed_presets.sqlite` (~30k voices, stored as parameter vectors,
  not `.syx`). **DONE** (#21): a name-based adapter (`dataset/dexed_sqlite_preset_loader.py`,
  D-NAMING) + `scripts/build_presetgen_corpus.py`; the `full_preset-gen-vae` corpus and its
  D-SPLIT train/test derivatives exist on disk.

**Exit criterion: MET.** Train on cluster → pull checkpoint → Evaluator scores a held-out split —
run end-to-end for Sound2Synth and both preset-gen-vae families (`results/dexed_builtin_test/`).
(The final human test set is finalized in Phase 6.)

## Phase 5 — Remaining model families

On the proven foundation; gated by **D-FAMILIES**. Each family is its own later sub-project reusing
the Phase 4 harness + packaging.

- **Generative family** (VAE — preset-gen-vae lineage) — **DONE** (#23/#35/#36): the full paper
  architecture (latent RealNVP flow included) as two registered families,
  `PresetGenVAEMLPRegressor` / `PresetGenVAEFlowRegressor`; trains on cluster. Map and port
  fidelity: `docs/PRESETGEN_VAE_PORT.md`.
- **Neural-proxy family** (InverSynth II — differentiable synth proxy) — **DONE**: the paper's three
  staged models as registered families, `IS` / `IS2xITF` / `IS2` (the last adds per-sample
  inference-time finetuning); all train on cluster. A peer paper approach, not a baseline. Map and
  port fidelity: `docs/INVERSYNTH2_PORT.md`.
- **Conditional-generative flow-matching family** (Hayes et al. ISMIR 2025 — approximately
  equivariant flow matching) — **DONE**: the paper's two Surge models as registered families,
  `FlowMatchingMLP` (non-equivariant control) / `FlowMatchingParam2Tok` (equivariant, the headline
  model). The only family built around synthesizer *symmetry*, and the first true sampler —
  `predict` integrates an ODE rather than doing a forward pass (D-FLOW-PREDICT). Trains on its own
  synthetic-uniform corpus (D-FLOW-CORPUS), unlike every other family. The paper's AST regression
  baseline is deliberately not ported (discriminative coverage already exists). Map and port
  fidelity: `docs/FLOW_MATCHING_PORT.md`.
- **Reinforcement-learning family** (SynthRL — Shin & Lee IJCAI-25) — **DONE**: two of the paper's
  three staged models as registered families, `SynthRLp` (stage 1, parameter loss only) and
  `SynthRLi` (stage 2, in-domain RL, warm-started from a stage-1 checkpoint via `--init-from`).
  Stage 3 `SynthRL-o` is deferred — it needs a second synth, so it is blocked on D-FAMILIES, not on
  the port. The only family that treats every parameter as a **classification** head, and the only
  one that renders with the live VST **inside the training loop** (D-RL-RENDER); `predict` and the
  eval path stay VST-free. Stage 2 is truncated to 36 epochs against the paper's 200 on measured
  cost. Map and port fidelity: `docs/SYNTHRL_PORT.md`.

*(Evolutionary search is dropped pending D-FAMILIES. If ever reinstated it runs its per-target search
locally with the live VST — it does not fit the cluster training harness.)*

## Phase 6 — Test set, benchmark, results

- **Human test corpus** — per D4; voice-disjoint from the training split; rendered fresh-process.
- **Benchmark orchestration** — run every family on the test set → `results/<corpus>/<model>/`.
- **Results aggregation** — comparative table across families, plus the metric-panel rank-correlation
  pruning (D-EVAL names `per_sample.csv` as the source of truth). **Finish line.**

**Out of scope:** the #12 dashboard, thesis prose/figures. The Diva work above runs in parallel and
does not enter this table.
