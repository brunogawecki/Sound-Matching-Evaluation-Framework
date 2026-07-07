#!/usr/bin/env bash
# Download the trained checkpoint (and Lightning CSV logs) from the cluster back
# to the laptop (issue #20), where the local Evaluator scores it. Sources
# cluster/cluster.env for the paths.
#
#     cluster/pull_checkpoint.sh        # download
#     cluster/pull_checkpoint.sh -n     # dry run (rsync -n), extra flags pass through
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
source cluster/cluster.env

mkdir -p "$LOCAL_CHECKPOINT_DIR"

echo "Pulling checkpoint $CLUSTER_SSH:$REMOTE_REPO_DIR/$CHECKPOINT_OUT -> $LOCAL_CHECKPOINT_DIR/"
rsync -avP "$@" "$CLUSTER_SSH:$REMOTE_REPO_DIR/$CHECKPOINT_OUT" "$LOCAL_CHECKPOINT_DIR/"

# Lightning CSV logs are handy for plotting loss curves; pull them if present.
echo "Pulling lightning_logs (if any) -> lightning_logs/"
rsync -avP "$@" "$CLUSTER_SSH:$REMOTE_REPO_DIR/lightning_logs/" "lightning_logs/" || true
