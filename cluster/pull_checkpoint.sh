#!/usr/bin/env bash
# Download the trained checkpoint (+ Lightning CSV logs) from the cluster to the
# laptop, where the local Evaluator scores it. Sources cluster/cluster.env for paths.
#
#     cluster/pull_checkpoint.sh        # download
#     cluster/pull_checkpoint.sh -n     # dry run; extra flags pass through to rsync
#
# Each pull is saved under a timestamped name and never overwrites a prior run; the
# `<stem>-latest` symlink tracks the newest, giving the eval command one stable path.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
source cluster/cluster.env

mkdir -p "$LOCAL_CHECKPOINT_DIR"

# Derive the name from the remote basename so it adapts to whatever CHECKPOINT_OUT is.
filename=$(basename "$CHECKPOINT_OUT")
stem="${filename%.*}"
extension="${filename##*.}"

# Timestamp by remote mtime (formatted on the cluster to dodge macOS/Linux date
# differences); a re-pull of the same run reuses the name. Fall back to pull-time.
timestamp=$(ssh "$CLUSTER_SSH" "date -r '$REMOTE_REPO_DIR/$CHECKPOINT_OUT' +%Y%m%d-%H%M%S" 2>/dev/null) \
  || timestamp=$(date +%Y%m%d-%H%M%S)
destination_name="${stem}-${timestamp}.${extension}"
latest_link="${stem}-latest.${extension}"

echo "Pulling checkpoint $CLUSTER_SSH:$REMOTE_REPO_DIR/$CHECKPOINT_OUT -> $LOCAL_CHECKPOINT_DIR/$destination_name"
rsync -avP "$@" "$CLUSTER_SSH:$REMOTE_REPO_DIR/$CHECKPOINT_OUT" "$LOCAL_CHECKPOINT_DIR/$destination_name"

# Only when the file arrived (skips on -n dry run, so no dangling symlink). Relative
# target keeps the link valid if the dir moves.
if [[ -f "$LOCAL_CHECKPOINT_DIR/$destination_name" ]]; then
  ln -sfn "$destination_name" "$LOCAL_CHECKPOINT_DIR/$latest_link"
  echo "Saved -> $LOCAL_CHECKPOINT_DIR/$destination_name (latest: $latest_link)"
  echo "Eval:  python scripts/evaluate.py --model $MODEL \\"
  echo "         --checkpoint $LOCAL_CHECKPOINT_DIR/$latest_link --corpus <local_test_corpus>"
fi

echo "Pulling lightning_logs (if any) -> lightning_logs/"
rsync -avP "$@" "$CLUSTER_SSH:$REMOTE_REPO_DIR/lightning_logs/" "lightning_logs/" || true
