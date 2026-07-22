# Training on the PUT GPU cluster

Runs `scripts/fit_model.py` on the PUT SLURM cluster (`slurm.cs.put.poznan.pl`, `hgx`
partition — A100-80GB). Training reads a self-describing corpus (D-SELFDESC) and needs only
`../requirements-cluster.txt`. One exception: **`SynthRLi` renders with a live Dexed inside its
training loop** (D-RL-RENDER) and needs the plugin on the node — see [Dexed](#dexed-synthrli-only).

Loop: **push corpus → submit job → pull checkpoint → evaluate locally.** Only training runs
on the cluster.

## Files

| File | What it does |
|---|---|
| `train.sbatch` | SLURM job: activates the conda env, runs `fit_model.py`. |
| `cluster.env(.example)` | Machine-specific paths / SSH target / account (gitignored; copy the example). |
| `push_corpus.sh` | `rsync` a corpus up to the cluster (laptop). |
| `pull_checkpoint.sh` | `rsync` the trained checkpoint (+ logs) back down (laptop). |
| `training_configs/smoke_config.yaml` | Acceptance run (2 epochs, any model family). |
| `training_configs/full_config.yaml` | Full run (30 epochs, Sound2Synth hyperparameters). |
| `training_configs/presetgen_full_config.yaml` | Full run for the preset-gen-vae families (400 epochs, the paper's hyperparameters). Third `train.sbatch` arg: `presetgen_full`. |
| `training_configs/synthrl_p_config.yaml` | SynthRL stage 1 (`SynthRLp`, parameter loss only). Third arg: `synthrl_p`. |
| `training_configs/synthrl_i_config.yaml` | SynthRL stage 2 (`SynthRLi`, RL fine-tune). Third arg: `synthrl_i`; needs a stage-1 checkpoint as the **fourth** arg and a Dexed install. |

## Config

`cluster.env` (gitignored; copy `cluster.env.example`) holds only **set-and-forget
infrastructure**, on both the laptop and the cluster checkout. Every **per-run choice — corpus,
model, training config — is a CLI arg**, so day-to-day runs never touch `cluster.env`:

- `CLUSTER_SSH`, `SLURM_ACCOUNT` — login target and billing account.
- `REMOTE_REPO_DIR`, `CONDA_BASE`, `CONDA_ENV` — cluster checkout and conda env.
- `LOCAL_CORPORA_DIR` / `REMOTE_CORPORA_DIR` — base dirs the corpus name is appended to
  (`dataset` locally, e.g. `/home/<login>/corpora` remotely).
- `LOCAL_CHECKPOINT_DIR` — where pulled checkpoints land.
- `DEXED_PATH` — the plugin on the compute node; only `SynthRLi` reads it (see below).
- `WANDB_*` — experiment tracking (see below).

Corpus, model, and config are passed on the CLI, and `fit_model.py` names the checkpoint from
the model's registry entry (`mean_parameter_baseline.json` / `spectrogram_cnn.pt`), so none of
`MODEL` / `TRAINING_CONFIG` / a checkpoint path live in `cluster.env`. One-time provisioning
(access, repo clone, conda env) is assumed already done.

## Train + pull

1. **Push the corpus** (laptop) — pass the corpus name (a dir under `LOCAL_CORPORA_DIR`):

   ```bash
   cluster/push_corpus.sh <corpus_name> -n     # preview
   cluster/push_corpus.sh <corpus_name>        # transfer
   ```

2. **Submit the job** (login node), syncing code first. Pass the corpus name, the model, and
   optionally the config (`full` default, `smoke`, or a `.yaml` path):

   ```bash
   cd "$REMOTE_REPO_DIR" && git pull
   sbatch -A "$SLURM_ACCOUNT" cluster/train.sbatch <corpus_name> <model_name>          # full run
   sbatch -A "$SLURM_ACCOUNT" cluster/train.sbatch <corpus_name> <model_name> smoke    # quick 2-epoch run
   ```

   `<model_name>` is any key in `MODEL_REGISTRY` (`models/registry.py`) — e.g.
   `MeanParameterBaseline`, `Sound2SynthSpectrogramRegressor`, `SynthRLp`.

   **Staged families** take a warm-start checkpoint as a fourth arg. `SynthRLi` continues from
   a finished `SynthRLp` run (path relative to the remote repo):

   ```bash
   sbatch -A "$SLURM_ACCOUNT" cluster/train.sbatch <corpus_name> SynthRLp synthrl_p
   sbatch -A "$SLURM_ACCOUNT" cluster/train.sbatch <corpus_name> SynthRLi synthrl_i \
       checkpoints/<stage1_jobid>/synthrl_p.pt
   ```

3. **Monitor** (login node): `squeue -u "$USER"` · `tail -f slurm-<JOBID>.out` ·
   `sacct -j <JOBID>` · `scancel <JOBID>`.

4. **Pull the checkpoint** (laptop) — pass the same model name; it resolves the checkpoint
   filename from the registry. Saved under a timestamped name with a `<stem>-latest` symlink
   tracking the newest, so the eval command stays stable across runs:

   ```bash
   cluster/pull_checkpoint.sh <model_name>
   ```

5. **Evaluate locally** (laptop, base VST env — the checkpoint is self-describing). Point
   `--checkpoint` at the `-latest` symlink the pull just printed:

   ```bash
   python scripts/evaluate.py --model <model_name> \
       --checkpoint checkpoints/spectrogram_cnn-latest.pt --corpus dataset/<test_corpus>
   ```

## Configs + wandb

- Switch smoke ↔ full with the 3rd `train.sbatch` arg (`smoke` / `full`, default `full`; a
  `.yaml` path also works). `train.sbatch`'s `--time=12:00:00` covers the full run; lower it
  for smoke. Configs live in `cluster/training_configs/`.
- wandb is off by default (a `CSVLogger` always writes `lightning_logs/`). To stream metrics,
  add `logger:\n  wandb: true` to the training config and set `WANDB_API_KEY` in `cluster.env`
  (from <https://wandb.ai/authorize>). If the node has no internet, set `WANDB_MODE=offline`
  and later `wandb sync lightning_logs/wandb/offline-run-*`.

## Dexed (`SynthRLi` only)

`SynthRLi` computes its RL reward by rendering each sampled patch and comparing it to the
target, so it renders with the real plugin **during training** (D-RL-RENDER). Every other
family is VST-free here. One-time setup on the cluster:

1. **Install Dexed 0.9.8** (Oct 2024), not the current release. 1.0.1's Linux build needs
   `GLIBC_2.38` / `GLIBCXX_3.4.32`; the cluster tops out at `2.35` / `3.4.30`, so it fails to
   `dlopen` — and JUCE reports that as `attempt to map invalid URI`, which looks like a bad
   path rather than a version mismatch. The D-SELFDESC spike confirmed 0.9.8 loads and renders:

   ```bash
   mkdir -p ~/plugins/dexed && cd ~/plugins/dexed
   # unpack the 0.9.8 Linux release here -> dexed-0.9.8-lnx/Dexed.vst3
   ```

2. **Point `cluster.env` at it**: `DEXED_PATH=$HOME/plugins/dexed/dexed-0.9.8-lnx/Dexed.vst3`.
   `train.sbatch` exports it and refuses to start a `SynthRLi` job if the path is missing,
   rather than letting it die inside the render pool.

3. **`dawdreamer` is already in `requirements-cluster.txt`** (pinned to `0.8.3`, the version the
   spike verified). No X11/Xvfb is needed — that risk never materialized.

Two things to know before a stage-2 run:

- **Rendering dominates the wall clock.** One fresh process per patch per step, plus
  `rl.prefill_epochs` gradient-free passes up front. `rl.num_render_workers` in the training
  config must match `--cpus-per-task` (both 8 today). The `--time=12:00:00` in `train.sbatch`
  will **not** cover 200 epochs on a large corpus.
- **Verify parameter-name parity first.** The corpus's `ParameterSpace` was built by the Mac's
  Dexed; the reward sets patches by name (D-NAMING) on 0.9.8 here. A renamed or missing
  parameter would silently change what the reward scores instead of erroring. Unverified —
  this is the open follow-up from the D-SELFDESC spike.

## Notes

- `--time` cap is 24 h per job; a timed-out job checkpoints and requeues (safety net, not a
  tested path).
- Corpus is read in place over shared `/home` — no node-local staging.
- No `.env` needed on the cluster; `config.py` has defaults for every value except
  `DEXED_PATH`, which `cluster.env` supplies for `SynthRLi` (its default is a macOS path).
