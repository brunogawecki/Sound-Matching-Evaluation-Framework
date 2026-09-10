# Run ledger — what has been trained and evaluated

Status of every benchmark cell: which model has a usable checkpoint, on which corpus and synth,
and whether it has been scored. This is the working record behind the thesis results chapter.

**Scope.** `docs/DECISIONS.md` owns *why*; GitHub Issues own *what to build*; this file owns
*what has actually run*. Keep it to facts that are checkable from `cluster/jobs.json`,
`checkpoints/` and `results/`.

Last updated: 2026-09-10 (out-of-domain sweep complete, all 42 cells scored; flow-matching
synthetic-uniform arm complete, all 4 cells scored, reported rows unchanged).

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

### Out-of-domain (D-OOD) — complete

NSynth audio-only corpora, no ground-truth parameters, so the 3 parameter metrics report `NaN`
and the 10 audio metrics run unchanged. Reuses the checkpoints above, no extra training.

| Corpus | Synth | n | Scored |
|---|---|---|---|
| `nsynth_c4_dexed` | dexed | 884 | all 11 |
| `nsynth_c4_diva` | diva | 884 | all 10 (no `SynthRLi`) |

## Reporting

The in-domain half is complete enough to report, and the thesis tables are generated from it by
`scripts/build_results_tables.py` (tables), `figures/plot_metric_correlation.py` and
`figures/plot_cross_synth_ranks.py` (figures). Output goes to `../thesis_latex/{tables,figures}/`
plus `results/aggregate/` for the machine-readable long form. Regeneration is deterministic: the
bootstrap is seeded, and a rebuild is byte-identical.

Reported from: `full_preset-gen-vae_test_1500` (11 models) and `diva_h2p_test` (10). Everything else
under `results/` is excluded and the reasons are listed in `docs/THESIS_NOTES.md` (§7).

The metric panel is reported at three levels under D-METRIC-PRUNE: 6 in the headline tables, the
pruned 10 in the cross-synth analysis, all 13 in the appendix. Pruning drops `param_mse`, `mel_mse`
and `mfcc_mse` only. Out of domain the same two table levels exist with the parameter axis removed,
so 4 in the headline and the 10 audio metrics in the appendix.

## Out-of-domain sweep — complete

All 21 OOD cells are scored: 11 models on `nsynth_c4_dexed` and 10 on `nsynth_c4_diva`, 884 samples
each, every one carrying the seeded 20-sample prediction-audio subset. No training was involved --
the in-domain checkpoints were reused unchanged and only the Evaluator ran. Reproduce with
`scripts/run_ood_sweep.sh`, which is resumable and skips any cell already on disk.

That closes the benchmark: **42 cells scored** -- 11 Dexed and 10 Diva, in-domain and out-of-domain
alike. Against a full 2 x 2 x 11 grid of 44, the two missing are Diva `SynthRLi` in both domains,
from the single blocker below. Every cell that can be scored has been.

Reported through `results-ood-{dexed,diva}.tex` (headline) and
`results-appendix-ood-{dexed,diva}-{magnitude,timbre-loudness-pitch}.tex` (full panel), audio
metrics only -- the parameter axis is undefined out of domain -- plus `results-domain-transfer.tex`. Per D-OOD these numbers rank models
against each other and are **not** absolute fidelity figures, and must not be tabled beside the
in-domain values as if the scales were comparable.

## Flow-matching synthetic-uniform arm (D-FLOW-CORPUS)

**Complete 2026-09-10.** Built to close the blocker below. A **separate arm**, not new benchmark
cells: the 42 cells above are untouched and keep the hybrid/human-trained flow-matching rows.
Scored into `results_synthetic_arm/` rather than `results/`, because the model class name is the
results directory name and the two arms would otherwise overwrite each other.

**Decision (Bruno, 2026-09-10): the thesis reports the standard-corpus rows.** The synthetic arm
stays out of the benchmark tables. Rationale and the result that motivates it are in
`docs/DECISIONS.md` under D-FLOW-CORPUS.

All four cells, both synths, with the training corpus as the only variable:

| Synth | Model | Job | Elapsed | Train | Eval |
|---|---|---|---|---|---|
| Dexed | `FlowMatchingMLP` | 1079944 | 03:23:02 | **done** | **done** |
| Dexed | `FlowMatchingParam2Tok` | 1079945 | 08:57:55 | **done** | **done** |
| Diva | `FlowMatchingMLP` | 1079974 | 03:27:48 | **done** | **done** |
| Diva | `FlowMatchingParam2Tok` | 1079975 | 09:34:31 | **done** | **done** |

Each cell carries the seeded 20-sample prediction-audio subset and all 13 metrics; Dexed at
n=1500, Diva at n=271, the same test corpora the benchmark uses.

### Dexed corpus repaired

`synthetic_uniform_train` was never rebuilt. Its `metadata.csv` and `run_summary.json` were
complete all along and only the audio was short, contiguously from `sample_015910` -- an
interrupted rsync, not an interrupted build. `scripts/render_missing_audio.py` re-rendered the
missing rows from the corpus's own metadata, so no parameters were re-drawn.

The repair is exact, not approximate. Re-rendering `sample_015908` / `sample_015909` from a fresh
wrapper reproduced the stored WAVs bit for bit (max |diff| 0.000e+00), against originals that
carried 15,908 renders of leaked in-process voice state. Dexed renders are fully determined by
their parameters. Cluster and local copies now both hold 23,448 WAVs, one distinct file size,
matching `metadata.csv` row for row.

### Diva corpus built

`diva_synthetic_uniform_train`, 23,448 samples, fresh-process per D-DIVA-RENDER, 1.49 s/preset
serial (~9.7 h). 0 near-silent and 0 render-timeout drops, so uniform Diva draws need no D-AUDIBLE
constraint and the prior is **exactly** G-invariant where Dexed's is only approximately -- Diva
inherits the empty `audible_sampling_ranges` default and so carries no OP1-style pin.

Built with `build_dataset.py synthetic --like-corpus dataset/diva_h2p_test`, a flag added for this
run. Without it the draw spans the wrapper's full 237-parameter / 1100-dimension subset, while both
Diva benchmark corpora are narrowed by `restrict_to_realized` to 231 / 892, and a model trained on
one cannot be scored on the other. The built corpus's parameter space matches `diva_h2p_test`
exactly, same names in the same order. Pushed and verified on the cluster: 23,448 WAVs, 23,448
metadata rows, one distinct file size.

### Outcome

**What the corpus swap cost.** Same model, same test set, training corpus the only change: 15 of 16
headline cells got worse, most by 50-280%. `param_accuracy` falls from ~0.79 to ~0.23 on Dexed and
from ~0.68 to ~0.35 on Diva. Both test sets are human presets, so a uniform training prior is
off-distribution and the parameter axis takes the worst of it. The lone exception is Diva
`FlowMatchingMLP`, better on `mss` (-12%) and `lsd` (-24%) while worse on the parameter axis --
broader audio coverage bought with parameter precision. Param2Tok shows no such trade.

**What it says about the symmetry claim.** D-FLOW-CORPUS predicts Param2Tok separates from its
`FlowMatchingMLP` control *when the prior is G-invariant*. Paired Wilcoxon over identical sample
sets, within each arm, says otherwise:

| Arm | Prior | Param2Tok vs MLP |
|---|---|---|
| Dexed standard | not invariant | mixed, small (`param_mae` p=0.97) |
| Dexed synthetic | approximately invariant | **worse**, `param_mae` +0.0196, p=1e-124 |
| Diva standard | 67% invariant | **better on all 6**, `lsd` -41.6%, `mss` -32.3% |
| Diva synthetic | exactly invariant | mixed, small |

The advantage appears on one arm only, Diva standard, which is the arm the symmetry argument does
not predict. On Dexed synthetic, where the prior is invariant, Param2Tok is significantly worse.
Diva's synthetic prior is *exactly* invariant, so the OP1 confound D-FLOW-CORPUS flags for Dexed
cannot explain it away: the required condition was met and the effect did not appear. The pattern
that does fit is train/test distribution match, not symmetry.

**Not yet done on these numbers**, and required before any of it is quoted:
- No Holm correction across the 6 metrics within an arm. `aggregate.py` corrects within a metric
  column across models, a different comparison.
- Diva synthetic `spectral_convergence` disagrees with itself: bootstrap CI [-9.80, +0.82] spans
  zero while Wilcoxon reads p=0.0086, so outliers drive the mean. Do not quote that cell.
- Diva is n=271, so its intervals are far wider than Dexed's.

## Blockers

**Flow-matching: the synthetic-uniform arm is missing, both synths.** RESOLVED 2026-09-10 -- the
arm is built and scored (see above), and the reported rows stay the standard-corpus ones by
decision. Kept here because the caveat below still attaches to those reported rows. All four
benchmark runs trained on the standard corpora (`full_preset-gen-vae_train` / `diva_h2p_hybrid_train`), confirmed from their
`slurm-*.out` headers. D-FLOW-CORPUS (LOCKED) requires a synthetic-uniform, G-invariant corpus
instead, because human presets are biased toward particular operator roles and remove the
permutation structure Param2Tok exists to exploit.

The two synths miss it by different margins, from the `method_counts` in each corpus's
`run_summary.json`:

| Train corpus | human | synthetic | augment |
|---|---|---|---|
| `full_preset-gen-vae_train` (Dexed) | 23,448 | 0 | 0 |
| `diva_h2p_hybrid_train` (Diva) | 1,084 | 15,704 | 6,660 |

Dexed is the total miss: an all-human prior, no invariance at all. Diva is a partial hit at 67%
uniform draws, and those draws are *exactly* uniform, since Diva inherits the empty
`audible_sampling_ranges` default (`synth/base_synth.py`) and so carries none of the OP1 pin that
D-FLOW-CORPUS names as its own confound on Dexed. Do not write the two arms off equally.

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

**Resolved 2026-09-10.** Both synthetic-uniform corpora were built and all four cells scored; see
"Flow-matching synthetic-uniform arm" above. The unfalsifiability worry in the paragraph above is
now answered rather than open: the synthetic arm exists, and Param2Tok does not out-earn its
control on it. The reported benchmark rows remain the standard-corpus ones by decision, so the
caveat above still belongs with those numbers.

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
