# Run ledger — what has been trained and evaluated

Status of every benchmark cell: which model has a usable checkpoint, on which corpus and synth,
and whether it has been scored. This is the working record behind the thesis results chapter.

**Scope.** `docs/DECISIONS.md` owns *why*; GitHub Issues own *what to build*; this file owns
*what has actually run*. Keep it to facts that are checkable from `cluster/jobs.json`,
`checkpoints/` and `results/`.

Last updated: 2026-09-07 (flow-matching pair scored on both synths; 21 of 22 cells).

## Benchmark status

Legend: **done** = scored, result on disk · **pending** = checkpoint ready, not yet scored ·
**running** = training on the cluster · **blocked** = cannot be scored as-is, reason given ·
**not started** = never submitted.

### Dexed — 11 of 11 scored

Train `full_preset-gen-vae_train` (23,448) · test `full_preset-gen-vae_test_1500` (1,500, D4).

| Model | Job | Config | Status |
|---|---|---|---|
| `MeanParameterBaseline` | local fit | — | **done** |
| `Sound2SynthSpectrogramRegressor` | 1073236 | `full` | **done** |
| `PresetGenVAEMLPRegressor` | 1073699 | `presetgen_full` | **done** |
| `PresetGenVAEFlowRegressor` | 1073700 | `presetgen_full` | **done** |
| `IS` | 992632 | `inversynth_is` | **done** |
| `IS2xITF` | 993074 | `inversynth2` | **done** |
| `IS2` | 993088 | `inversynth2` | **done** |
| `SynthRLp` | 996865 | `synthrl_p` | **done** |
| `SynthRLi` | 1007123 | `synthrl_i` | **done** |
| `FlowMatchingMLP` | 994884 | `flow_matching` | **done** — hybrid-corpus arm only, see Blockers |
| `FlowMatchingParam2Tok` | 994886 | `flow_matching` | **done** — hybrid-corpus arm only, see Blockers |

### Diva — 10 of 11 scored

Train `diva_h2p_hybrid_train` (23,448) · test `diva_h2p_test` (271, voice-disjoint from train).

| Model | Job | Config | Status |
|---|---|---|---|
| `MeanParameterBaseline` | local fit | — | **done** |
| `Sound2SynthSpectrogramRegressor` | 1073242 | `full` | **done** |
| `PresetGenVAEFlowRegressor` | 1073698 | `presetgen_full` | **done** |
| `PresetGenVAEMLPRegressor` | 1073776 | `presetgen_full` | **done** |
| `IS` | 1073817 | `inversynth2` | **done** |
| `IS2xITF` | 1073818 | `inversynth2` | **done** |
| `IS2` | 1073819 | `inversynth2` | **done** |
| `SynthRLp` | 1073820 | `synthrl_p` | **done** |
| `FlowMatchingMLP` | 1073243 | `flow_matching` | **done** — hybrid-corpus arm only, see Blockers |
| `FlowMatchingParam2Tok` | 1073244 | `flow_matching` | **done** — hybrid-corpus arm only, see Blockers |
| `SynthRLi` | — | `synthrl_i` | **blocked** — needs the live VST in the training loop (D-RL-RENDER); Diva is not installed on the cluster and `train.sbatch` only knows `DEXED_PATH` |

### Out-of-domain (D-OOD) — 1 of 11 per synth

NSynth audio-only corpora, no ground-truth parameters, so the 3 parameter metrics report `NaN`
and the 10 audio metrics run unchanged. Reuses the checkpoints above, no extra training.

| Corpus | Synth | n | Scored |
|---|---|---|---|
| `nsynth_c4_dexed` | dexed | 884 | `MeanParameterBaseline` |
| `nsynth_c4_diva` | diva | 884 | `MeanParameterBaseline` |

## Blockers

**Flow-matching: the synthetic-uniform arm is missing, both synths.** All four runs trained on the
standard corpora (`full_preset-gen-vae_train` / `diva_h2p_hybrid_train`), confirmed from their
`slurm-*.out` headers. D-FLOW-CORPUS (LOCKED) requires a synthetic-uniform, G-invariant corpus
instead, because human presets are biased toward particular operator roles and remove the
permutation structure Param2Tok exists to exploit.

These four are scored anyway, and are best read as the **hybrid arm** of the sweep D-FLOW-CORPUS's
own Consequences paragraph calls for ("training both flow-matching families across synthetic /
human / hybrid corpora, all scored on the same test set"), with `FlowMatchingMLP` present as the
required non-equivariant control. What is missing is the synthetic arm to contrast against. Without
it, a small MLP-vs-Param2Tok gap cannot distinguish "the G-invariance argument holds" from "the
model is simply not better" — the prediction is unfalsifiable on one arm. **Do not report these as
a plain flow-matching result**; the caveat belongs with the number.

Observed on the hybrid arm: on Dexed the two are indistinguishable (`param_mae` 0.1005 vs 0.1009),
which is what D-FLOW-CORPUS predicts. On Diva they separate (0.2111 vs 0.1481), but both score
*worse* than `MeanParameterBaseline` on `spectral_convergence` (3.71 / 2.12 against 1.12), so the
parameter and audio axes disagree there and the Diva pair needs a closer look before use.

Building the synthetic arm: `synthetic_uniform_train` on the cluster holds 15,910 of the 23,448
WAVs its `metadata.csv` claims and there is no local copy, so it needs a real rebuild, not a
re-sync. Diva has no synthetic-uniform corpus at any stage, and D-DIVA-RENDER's fresh-process
plus warm-up render makes building one materially more expensive than the Dexed equivalent.

**Diva `SynthRLi`.** Needs the Diva plugin on the cluster. Not resolvable without a plugin install
plus an `sbatch` change.

## Superseded and failed runs

Kept so a checkpoint directory can always be traced back to why it is not the one in use.

| Job | Model | Corpus | Outcome |
|---|---|---|---|
| 966543 / 966550 | `Sound2SynthSpectrogramRegressor` | `dexed_builtin` | superseded by 1073236 (wrong corpus: 841 samples, not the benchmark train set) |
| 966549 | `Sound2SynthSpectrogramRegressor` | `dexed_builtin` | CANCELLED |
| 966922 / 966923 / 966924 | PresetGenVAE both heads | `full_preset-gen-vae` | pre-split corpus, superseded |
| 980157 | `PresetGenVAEMLPRegressor` | `full_preset-gen-vae_train` | FAILED after 1:04 |
| 981447 / 982169 | PresetGenVAE both heads | `full_preset-gen-vae_train` | superseded by 985319 / 985760 |
| 985319 / 985760 | PresetGenVAE both heads | `full_preset-gen-vae_train` | superseded by 1073699 / 1073700 (trained before `gradient_clip_val`) |
| 996864 | `SynthRLp` | `full_preset-gen-vae_train` | FAILED after 0:14 |
| 1006799 | `SynthRLi` | `full_preset-gen-vae_train` | TIMEOUT at 25 h; superseded by 1007123 |
| 1073237 | `PresetGenVAEMLPRegressor` | `diva_h2p_hybrid_train` | `val_loss=nan` at epoch 6; superseded |
| 1073241 | `PresetGenVAEFlowRegressor` | `diva_h2p_hybrid_train` | healthy but pre-`gradient_clip_val`; superseded by 1073698 |
| 1073696 | `PresetGenVAEMLPRegressor` | `diva_h2p_hybrid_train` | `val_loss=nan` at epoch 6 again, with clipping on; superseded by 1073776 |

### The Diva `PresetGenVAEMLPRegressor` NaN

Two runs died identically at epoch 6 before the cause was found. Training was never diverging —
train loss fell monotonically and only `val_latent_loss` went non-finite. In eval the VAE skips
reparameterization and `z0` **is** `mu` (`network.py:396-401`, matching the paper), so
`(samples - mu)` is exactly zero and the density's quadratic term becomes `0 / exp(logvar)`. Diva's
`logvar` spread to ±20 by epoch 4 against Dexed's ±2, driven by `latent_dimension` being pinned to
`ml_dimension` (892 for Diva, 333 for Dexed, against the paper's 256). Once `exp(logvar)`
underflowed to zero the term was `0/0` = NaN. Fixed by clamping `logvar` in
`models/training/loss.py` (`9a577ca`); regression test in `tests/test_latent_loss_numerics.py`.

The rerun (1073776) confirms the fix. Its `val_latent_loss` is bit-identical to 1073696 for
epochs 0-3 (`0.43633389472961426`, `0.3770040273666382`, `0.2780931293964386`,
`0.22457392513751984`), diverges first at epoch 4, and reads a finite `0.3197` at epoch 6 where
the old run read `nan` — so the clamp is inert on healthy values and engages only where the run
used to die. It then trained 242 epochs to `val_loss=0.346` with no non-finite value in any
logged column, and the finished model's `logvar` sits at -2.82 to +0.10, back in the same band as
the healthy Dexed run, with the clamp biting 0 of 57,088 entries. The 892-wide latent was
recoverable, so no change to `latent_dimension` is needed.

The clamp cannot change any score: `evaluation/`, `BaseDeepModel.predict` and `families.py` never
call `gaussian_log_probability`, `flow_latent_loss` or `forward_training`. Results scored before
`9a577ca` stay valid.

Gradient clipping was added while chasing this and did not fix it, but was kept: it changed the
Dexed results by under 0.01 val_loss and removes a real hazard, and applying it to both synths
avoids a per-synth exception in the methodology.

## Corpora

| Corpus | Synth | n | Role |
|---|---|---|---|
| `full_preset-gen-vae_train` | dexed | 23,448 | Dexed training |
| `full_preset-gen-vae_test_1500` | dexed | 1,500 | Dexed benchmark test (D4) |
| `full_preset-gen-vae_test` | dexed | 5,862 | full test split, subsampled to the 1,500 above |
| `diva_h2p_hybrid_train` | diva | 23,448 | Diva training (15,704 synthetic + 6,660 augment + 1,084 human) |
| `diva_h2p_test` | diva | 271 | Diva benchmark test, voice-disjoint from train |
| `nsynth_c4_dexed` / `nsynth_c4_diva` | both | 884 | out-of-domain, audio-only (D-OOD) |
| `dexed_builtin*` | dexed | 1,051 | retired pilot corpus |
| `synthetic_uniform_train` | dexed | 23,448 claimed | **broken**: 15,910 WAVs on the cluster |

Both benchmark corpora share one render contract: 22,050 Hz, 4.0 s, note held 3.0 s, MIDI note 60,
velocity 100, dawdreamer, fresh-process at position 0 (D-REPRO / D-DIVA-RENDER).

## Refreshing this file

```bash
# training runs + whether the checkpoint is local
python -c "import json;from pathlib import Path;[print(j['job_id'],j['model'],j['corpus'],j['config'],'ckpt' if any((Path('checkpoints')/j['job_id']).glob('*.pt')) else '-') for j in json.load(open('cluster/jobs.json'))]"

# SLURM outcome for every registered job
ids=$(python -c "import json;print(','.join(j['job_id'] for j in json.load(open('cluster/jobs.json'))))")
ssh "$(grep CLUSTER_SSH cluster/cluster.env | cut -d= -f2)" "sacct -j $ids --format=JobID,State,Elapsed -X -n"

# evaluations on disk
python -c "import json;from pathlib import Path;[print(r.name,m.name,json.load(open(m/'eval_summary.json'))['num_samples'],json.load(open(m/'eval_summary.json'))['checkpoint']['path']) for r in sorted(Path('results').iterdir()) if r.is_dir() for m in sorted(r.iterdir()) if (m/'eval_summary.json').is_file()]"
```

`cluster/jobs.json` is gitignored and local-only, so it is the one input here that does not travel
with the repo.
