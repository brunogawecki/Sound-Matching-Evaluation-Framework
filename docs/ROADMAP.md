# Roadmap — Dexed pipeline → first comparative benchmark

This is the **decomposition** of the work between today's state (a corpus, a model interface, a
trivial baseline, and a working Evaluator) and the project's goal (a comparative benchmark across
model families on Dexed). It is high-level on purpose: each task below gets its own detailed-design
session later. Scope is **Dexed-only** (D-ORDER) — the full pipeline must be proven on Dexed before
any second synth — and the roadmap **ends at "benchmark results produced."** Thesis prose/figures,
the #12 dashboard (built — see `ARCHITECTURE.md`, but tangential to the benchmark path), and
Surge XT are out of scope.

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

- **No neural-proxy baseline** — and whether it is built at all is the open half of D-FAMILIES.
- **No fuller Sound2Synth architecture.** The landed model is a single-spectrogram-branch first cut;
  the paper's multi-modal encoder + grouped-FC parameter classifier is still future work.
- **No final human test set (D4), no benchmark orchestration, no benchmark table** — Phase 6. The
  existing results rows are pipeline shakedowns, not benchmark numbers.

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
- **Neural-proxy baseline** (differentiable synth proxy) — trains on cluster; **baseline, not a
  primary family.** Whether it is built at all is the open half of D-FAMILIES.

*(Evolutionary search is dropped pending D-FAMILIES. If ever reinstated it runs its per-target search
locally with the live VST — it does not fit the cluster training harness.)*

## Phase 6 — Test set, benchmark, results

- **Human test corpus** — per D4; voice-disjoint from the training split; rendered fresh-process.
- **Benchmark orchestration** — run every family on the test set → `results/<corpus>/<model>/`.
- **Results aggregation** — comparative table across families, plus the metric-panel rank-correlation
  pruning (D-EVAL names `per_sample.csv` as the source of truth). **Finish line.**

**Out of scope:** the #12 dashboard, the second synth (Surge XT), thesis prose/figures.
