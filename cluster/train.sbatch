#!/bin/bash
# SLURM batch script — fit a sound-matching model on the PUT hgx partition
# (issue #20). The model family is chosen with MODEL in cluster.env; there is no
# default, so submission fails fast (set -u) if it is not set. Submit from the
# repo root on the cluster:
#
#     sbatch -A "$SLURM_ACCOUNT" cluster/train.sbatch
#
# The billing account (-A) is a submission-time scheduling flag, so it cannot be
# read from cluster.env inside the body — pass it on the command line (see README).
# Everything else (model, paths, env name, config) is read from cluster/cluster.env
# at run time.
#
#SBATCH --job-name=smef-train
#SBATCH --partition=hgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Resolve machine-specific paths / env name (gitignored; see cluster.env.example).
source cluster/cluster.env

# Export wandb settings so the python process inherits them. Defaults keep
# `set -u` happy when cluster.env omits them.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

cd "$REMOTE_REPO_DIR"

# Activate the conda env non-interactively (a batch shell has no `conda init`).
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "--- host $(hostname) | env $CONDA_ENV | corpus $REMOTE_CORPUS_DIR ---"
nvidia-smi || true

python scripts/fit_model.py \
    --model "$MODEL" \
    --corpus "$REMOTE_CORPUS_DIR" \
    --config "$TRAINING_CONFIG" \
    --out "$CHECKPOINT_OUT"

echo "--- done; checkpoint at $CHECKPOINT_OUT ---"
