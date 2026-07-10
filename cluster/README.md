# Training on the PUT GPU cluster

Runs `scripts/fit_model.py` on the PUT SLURM cluster (`slurm.cs.put.poznan.pl`, `hgx`
partition — A100-80GB). Training reads a self-describing corpus and never touches a VST
(D-SELFDESC), so the cluster only needs `../requirements-cluster.txt`.

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

## Config

`cluster.env` (gitignored; copy `cluster.env.example`) holds only **set-and-forget
infrastructure**, on both the laptop and the cluster checkout. Every **per-run choice — corpus,
model, training config — is a CLI arg**, so day-to-day runs never touch `cluster.env`:

- `CLUSTER_SSH`, `SLURM_ACCOUNT` — login target and billing account.
- `REMOTE_REPO_DIR`, `CONDA_BASE`, `CONDA_ENV` — cluster checkout and conda env.
- `LOCAL_CORPORA_DIR` / `REMOTE_CORPORA_DIR` — base dirs the corpus name is appended to
  (`dataset` locally, e.g. `/home/<login>/corpora` remotely).
- `LOCAL_CHECKPOINT_DIR` — where pulled checkpoints land.
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

   `<model_name>` is `MeanParameterBaseline` or `Sound2SynthSpectrogramRegressor`.

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

## Notes

- `--time` cap is 24 h per job; a timed-out job checkpoints and requeues (safety net, not a
  tested path).
- Corpus is read in place over shared `/home` — no node-local staging.
- No `.env` needed on the cluster; `config.py` has defaults for every value.
