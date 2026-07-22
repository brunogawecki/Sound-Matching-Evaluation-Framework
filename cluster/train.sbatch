#!/bin/bash
# SLURM batch script — fit a sound-matching model on the PUT hgx partition.
# Positional args: corpus name (under REMOTE_CORPORA_DIR), model family, an optional
# training config (default `full`), and an optional warm-start checkpoint for staged
# families (SynthRLi from a SynthRLp checkpoint). Submit from the repo root:
#
#     sbatch -A "$SLURM_ACCOUNT" cluster/train.sbatch <corpus_name> <model_name> [config] [init_from]
#
# -A is a submission-time flag, so it's passed on the command line, not read from
# cluster.env. [config] is `full`/`smoke` (bare names) or a literal .yaml path.
# [init_from] is a checkpoint path relative to the remote repo (e.g. the -p run's
# checkpoints/<jobid>/synthrl_p.pt); omit it for from-scratch families.
#
#SBATCH --job-name=smef-train
#SBATCH --partition=hgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

source cluster/cluster.env

# corpus and model are required; missing either fails fast under set -u.
corpus="${1:?usage: sbatch -A \"\$SLURM_ACCOUNT\" cluster/train.sbatch <corpus_name> <model_name> [config]}"
model="${2:?usage: sbatch -A \"\$SLURM_ACCOUNT\" cluster/train.sbatch <corpus_name> <model_name> [config]}"
REMOTE_CORPUS_DIR="$REMOTE_CORPORA_DIR/$corpus"

# A .yaml path is used as-is; a bare name (e.g. full, smoke) maps to a configs file.
config_arg="${3:-full}"
case "$config_arg" in
  */* | *.yaml) TRAINING_CONFIG="$config_arg" ;;
  *)            TRAINING_CONFIG="cluster/training_configs/${config_arg}_config.yaml" ;;
esac

# Optional warm-start checkpoint (staged families only). Empty by default.
INIT_FROM="${4:-}"

# Defaults keep set -u happy when cluster.env omits these.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

cd "$REMOTE_REPO_DIR"

# A batch shell has no `conda init`, so source it before activating.
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "--- host $(hostname) | env $CONDA_ENV | model $model | config $TRAINING_CONFIG | corpus $REMOTE_CORPUS_DIR ---"
nvidia-smi || true

# No --out: fit_model.py names the checkpoint from the model's registry entry.
# --run-id scopes this job's artifacts to checkpoints/<job id>/ and
# lightning_logs/<job id>/, so repeat runs of a family never overwrite each other
# and the dashboard can pull exactly one job's output.
python scripts/fit_model.py \
    --model "$model" \
    --corpus "$REMOTE_CORPUS_DIR" \
    --config "$TRAINING_CONFIG" \
    --run-id "$SLURM_JOB_ID" \
    ${INIT_FROM:+--init-from "$INIT_FROM"}

echo "--- done; checkpoint under checkpoints/$SLURM_JOB_ID/ ---"
