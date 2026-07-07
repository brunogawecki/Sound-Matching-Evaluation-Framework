#!/usr/bin/env bash
# Upload a rendered corpus from the laptop to the cluster (issue #20). Run from
# anywhere; sources cluster/cluster.env for the source/target paths.
#
#     cluster/push_corpus.sh            # upload
#     cluster/push_corpus.sh -n         # dry run (rsync -n), extra flags pass through
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
source cluster/cluster.env

echo "Pushing $LOCAL_CORPUS_DIR -> $CLUSTER_SSH:$REMOTE_CORPUS_DIR"
rsync -avP "$@" "$LOCAL_CORPUS_DIR/" "$CLUSTER_SSH:$REMOTE_CORPUS_DIR/"
