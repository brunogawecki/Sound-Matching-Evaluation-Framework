#!/usr/bin/env bash
# Download one training job's checkpoint (+ its Lightning CSV logs) from the cluster.
#
#     cluster/pull_checkpoint.sh <job_id> <model_name>              # download
#     cluster/pull_checkpoint.sh <job_id> <model_name> --with-ckpt  # + raw Lightning .ckpt files
#     cluster/pull_checkpoint.sh <job_id> <model_name> -n           # dry run; extra flags pass to rsync
#
# Jobs submitted since per-job scoping write to checkpoints/<job_id>/ and
# lightning_logs/<job_id>/, so a pull names exactly one run and the local layout
# mirrors the remote. Older jobs shared one checkpoint path per model family; for
# those this falls back to the flat path and says so.
#
# The raw .ckpt files (~450 MB each) carry optimizer state for resuming training,
# which happens on the cluster, so they stay there unless --with-ckpt is passed.
# The exported .pt already holds the best epoch's weights.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
source cluster/cluster.env

job_id="${1:?usage: cluster/pull_checkpoint.sh <job_id> <model_name> [--with-ckpt] [rsync flags]}"
model="${2:?usage: cluster/pull_checkpoint.sh <job_id> <model_name> [--with-ckpt] [rsync flags]}"
shift 2

# --with-ckpt is ours; everything else passes through to rsync (e.g. -n).
with_ckpt=0
rsync_flags=()
for arg in "$@"; do
  if [[ "$arg" == "--with-ckpt" ]]; then
    with_ckpt=1
  else
    rsync_flags+=("$arg")
  fi
done

remote_job_checkpoints="$REMOTE_REPO_DIR/checkpoints/$job_id"
quoted_remote_job_checkpoints=$(printf '%q' "$remote_job_checkpoints")

# Legacy: the job ran before per-job scoping, so its output went to the shared
# per-family path, where a later run of the same family may have overwritten it.
if ! ssh "$CLUSTER_SSH" "test -d $quoted_remote_job_checkpoints"; then
  echo "No checkpoints/$job_id/ on the cluster — job $job_id predates per-job scoping."
  echo "WARNING: falling back to the shared path for '$model'. That file is whatever the"
  echo "         most recent run of this family wrote. It may NOT be job $job_id's."
  echo "         Its lightning_logs can't be attributed to a job id, so they are skipped."

  filename=$(python -c "from models.registry import MODEL_REGISTRY; print(MODEL_REGISTRY['$model'].default_checkpoint_filename)") || {
    echo "Could not resolve a checkpoint filename for model '$model'." >&2
    echo "Check the name and that the base env (with torch) is active." >&2
    exit 1
  }
  stem="${filename%.*}"
  extension="${filename##*.}"
  remote_checkpoint="$REMOTE_REPO_DIR/checkpoints/$filename"

  mkdir -p "$LOCAL_CHECKPOINT_DIR"

  # Timestamp by remote mtime (formatted on the cluster for consistent date flags);
  # a re-pull of the same run reuses the name. Fall back to pull-time.
  timestamp=$(ssh "$CLUSTER_SSH" "date -r '$remote_checkpoint' +%Y%m%d-%H%M%S" 2>/dev/null) \
    || timestamp=$(date +%Y%m%d-%H%M%S)
  destination_name="${stem}-${timestamp}.${extension}"
  latest_link="${stem}-latest.${extension}"

  echo "Pulling $CLUSTER_SSH:$remote_checkpoint -> $LOCAL_CHECKPOINT_DIR/$destination_name"
  rsync -avP ${rsync_flags[@]+"${rsync_flags[@]}"} \
    "$CLUSTER_SSH:$remote_checkpoint" "$LOCAL_CHECKPOINT_DIR/$destination_name"

  # Only if the file arrived (skips on -n dry run). Relative target survives a dir move.
  if [[ -f "$LOCAL_CHECKPOINT_DIR/$destination_name" ]]; then
    ln -sfn "$destination_name" "$LOCAL_CHECKPOINT_DIR/$latest_link"
    echo "Saved -> $LOCAL_CHECKPOINT_DIR/$destination_name (latest: $latest_link)"
    echo "Eval:  python scripts/evaluate.py --model $model \\"
    echo "         --checkpoint $LOCAL_CHECKPOINT_DIR/$latest_link --corpus <local_test_corpus>"
  fi
  exit 0
fi

# Per-job layout. The job's directory holds exactly one checkpoint, so there is no
# filename to resolve — the whole directory comes down as-is.
local_job_checkpoints="$LOCAL_CHECKPOINT_DIR/$job_id"
mkdir -p "$local_job_checkpoints"

echo "Pulling checkpoint $CLUSTER_SSH:$remote_job_checkpoints/ -> $local_job_checkpoints/"
rsync -avP ${rsync_flags[@]+"${rsync_flags[@]}"} \
  "$CLUSTER_SSH:$remote_job_checkpoints/" "$local_job_checkpoints/"

# wandb/ duplicates the W&B cloud copy; the raw .ckpt files are opt-in.
log_excludes=(--exclude "wandb/")
if [[ "$with_ckpt" -eq 0 ]]; then
  log_excludes+=(--exclude "checkpoints/")
  echo "Pulling lightning_logs (CSV only; --with-ckpt also fetches the raw .ckpt files)"
else
  echo "Pulling lightning_logs including the raw .ckpt files (--with-ckpt)"
fi

# A job cancelled before it logged anything has no lightning_logs; not an error.
rsync -avP "${log_excludes[@]}" ${rsync_flags[@]+"${rsync_flags[@]}"} \
  "$CLUSTER_SSH:$REMOTE_REPO_DIR/lightning_logs/$job_id/" "lightning_logs/$job_id/" || true

echo "Saved -> $local_job_checkpoints/ (job $job_id, $model)"
echo "Eval:  python scripts/evaluate.py --model $model \\"
echo "         --checkpoint $local_job_checkpoints/<checkpoint> --corpus <local_test_corpus>"
