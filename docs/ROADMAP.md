# Roadmap — Dexed pipeline → first comparative benchmark

This is the **decomposition** of the work between today's state (a corpus, a model interface, a
trivial baseline, and a working Evaluator) and the project's goal (a comparative benchmark across
model families on Dexed). It is high-level on purpose: each task below gets its own detailed-design
session later. Scope is **Dexed-only** (D-ORDER) — the full pipeline must be proven on Dexed before
any second synth — and the roadmap **ends at "benchmark results produced."** Thesis prose/figures,
the #12 dashboard, and Surge XT are out of scope.

The split with the rest of `docs/` is the usual one: **this file owns the *decomposition and
ordering*; `DECISIONS.md` owns the *why*; GitHub issues own the *do*.** Open decisions resolve in
`DECISIONS.md`, never as issues — only the work they unblock becomes an issue.

## Where we are

Built and run end-to-end once: Layer 2 (data), the Layer 3 `BaseModel` interface + trivial
`MeanParameterBaseline`, and Layer 4 (evaluation). Corpora exist on disk; the baseline is fitted and
scored; Phase 2 / Phase 3 milestones are closed.

What does **not** exist yet:

- **No training orchestration.** `BaseModel` (`models/base_model.py`) deliberately defers the
  training-framework choice to "the first deep family." There is no train/val loop, config system,
  logging, checkpoint convention, or seeding harness — only the mean baseline's trivial `fit`.
- **No cluster packaging.** Training is meant to run on an external Linux GPU cluster with no VST
  (D-SELFDESC), but nothing splits dependencies, submits jobs, or moves corpora up / checkpoints
  down.
- **No real models.** `models/` holds only the ABC + mean baseline.
- **No real training corpus, no human test set, no benchmark table.**

## Sequencing — vertical slice first

Stand up the training framework **and** cluster packaging by driving them end-to-end with a single
discriminative model, before building the other families on the proven foundation. This de-risks the
unknowns (orchestration, packaging, cluster I/O) once, against the lowest-risk architecture, rather
than discovering them family-by-family. It mirrors D-ORDER one level down.

## Gating decisions (resolve in `DECISIONS.md`, not as issues)

| Decision | Status | Blocks | Note |
|---|---|---|---|
| **D-FRAMEWORK** — PyTorch Lightning vs. raw PyTorch loop | OPEN (stub) | Phase 4 training harness | New record needed before the harness task. |
| **D-FAMILIES** — final model-family set | OPEN (stub) | Phase 5 | Discriminative + generative (primary) + neural-proxy (baseline); evolutionary dropped. |
| **D4** — human test-set composition | OPEN | Phase 6 | Importer built; final split unblocked once Phase 4 lands. |

## Phase 4 — Training foundation, proven by one real model

Goal: a real (non-trivial) results row, produced by training a discriminative model on the cluster
and scoring it through the existing Evaluator.

- **Training harness** — config system, train/val loop, logging, checkpoint convention consumable by
  `BaseModel.load`, seeding/reproducibility. *(Gated by D-FRAMEWORK.)*
- **Discriminative parameter regressor** — first real model family (spectrogram→params, the
  InverSynth / preset-gen-vae lineage; lowest-risk architecture). First real `BaseModel.fit`.
- **Cluster packaging** — dependency split (cluster requirements **without** VST/dawdreamer, per
  D-SELFDESC), environment/container spec, job-submission scripts, corpus-up / checkpoint-down
  transfer, entrypoint.
- **Training corpus from preset-gen-vae** — the human DX7 collection at
  `paper_repos/preset-gen-vae/synth/dexed_presets.sqlite`. Note: that path is currently a **Git LFS
  pointer** (the ~25.6 MB DB is not pulled), and the data is stored as **parameter vectors, not
  `.syx`** — the `preset` table holds one `pickled_params_np_array` per voice plus a `param` table of
  index→name. So this needs a **name-based adapter** (map preset-gen-vae's parameter *names* onto our
  wrapper's plugin-reported names — never by index, per D-NAMING; preset-gen-vae used a different
  Dexed build), projected onto the D1 subset and rendered via the existing `DatasetBuilder` /
  `FreshProcessRenderBackend` (D-REPRO). Reuse the dedup + voice-disjoint split logic from
  `dataset/dexed_preset_loader.py`.

**Exit criterion:** train on cluster → pull checkpoint → Evaluator scores it on a held-out split →
first real results row. (The final human test set is finalized in Phase 6.)

## Phase 5 — Remaining model families

On the proven foundation; gated by **D-FAMILIES**. Each family is its own later sub-project reusing
the Phase 4 harness + packaging.

- **Generative family** (e.g. VAE — preset-gen-vae lineage) — trains on cluster.
- **Neural-proxy baseline** (differentiable synth proxy) — trains on cluster; **baseline, not a
  primary family.**

*(Evolutionary search is dropped pending D-FAMILIES. If ever reinstated it runs its per-target search
locally with the live VST — it does not fit the cluster training harness.)*

## Phase 6 — Test set, benchmark, results

- **Human test corpus** — per D4; voice-disjoint from the training split; rendered fresh-process.
- **Benchmark orchestration** — run every family on the test set → `results/<corpus>/<model>/`.
- **Results aggregation** — comparative table across families, plus the metric-panel rank-correlation
  pruning (D-EVAL names `per_sample.csv` as the source of truth). **Finish line.**

**Out of scope:** the #12 dashboard, the second synth (Surge XT), thesis prose/figures.
