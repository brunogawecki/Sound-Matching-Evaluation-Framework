#!/usr/bin/env bash
# Download a trained checkpoint (+ Lightning CSV logs) from the cluster to the laptop.
# The model name (same one passed to train.sbatch) picks which checkpoint to pull.
#
#     cluster/pull_checkpoint.sh <model_name>        # download
#     cluster/pull_checkpoint.sh <model_name> -n     # dry run; extra flags pass to rsync
#
# Each pull is timestamped (never overwrites); the <stem>-latest symlink tracks the newest.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
source cluster/cluster.env

model="${1:?usage: cluster/pull_checkpoint.sh <model_name> [rsync flags]}"
shift

mkdir -p "$LOCAL_CHECKPOINT_DIR"

# Registry gives this model's checkpoint filename (matches what fit_model.py wrote).
filename=$(python -c "from models.registry import MODEL_REGISTRY; print(MODEL_REGISTRY['$model'].default_checkpoint_filename)") || {
  echo "Could not resolve a checkpoint filename for model '$model'." >&2
  echo "Check the name and that the base env (with torch) is active." >&2
  exit 1
}
stem="${filename%.*}"
extension="${filename##*.}"
remote_checkpoint="$REMOTE_REPO_DIR/checkpoints/$filename"

# Timestamp by remote mtime (formatted on the cluster for consistent date flags);
# a re-pull of the same run reuses the name. Fall back to pull-time.
timestamp=$(ssh "$CLUSTER_SSH" "date -r '$remote_checkpoint' +%Y%m%d-%H%M%S" 2>/dev/null) \
  || timestamp=$(date +%Y%m%d-%H%M%S)
destination_name="${stem}-${timestamp}.${extension}"
latest_link="${stem}-latest.${extension}"

echo "Pulling checkpoint $CLUSTER_SSH:$remote_checkpoint -> $LOCAL_CHECKPOINT_DIR/$destination_name"
rsync -avP "$@" "$CLUSTER_SSH:$remote_checkpoint" "$LOCAL_CHECKPOINT_DIR/$destination_name"

# Only if the file arrived (skips on -n dry run). Relative target survives a dir move.
if [[ -f "$LOCAL_CHECKPOINT_DIR/$destination_name" ]]; then
  ln -sfn "$destination_name" "$LOCAL_CHECKPOINT_DIR/$latest_link"
  echo "Saved -> $LOCAL_CHECKPOINT_DIR/$destination_name (latest: $latest_link)"
  echo "Eval:  python scripts/evaluate.py --model $model \\"
  echo "         --checkpoint $LOCAL_CHECKPOINT_DIR/$latest_link --corpus <local_test_corpus>"
fi

echo "Pulling lightning_logs (if any) -> lightning_logs/"
rsync -avP "$@" "$CLUSTER_SSH:$REMOTE_REPO_DIR/lightning_logs/" "lightning_logs/" || true
