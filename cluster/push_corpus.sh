#!/usr/bin/env bash
# Upload a rendered corpus from the laptop to the cluster. The corpus name is
# appended to LOCAL_CORPORA_DIR (source) and REMOTE_CORPORA_DIR (target), so push
# and train pick the same corpus by name.
#
#     cluster/push_corpus.sh <corpus_name>        # upload
#     cluster/push_corpus.sh <corpus_name> -n     # dry run; extra flags pass through to rsync
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
source cluster/cluster.env

corpus="${1:?usage: cluster/push_corpus.sh <corpus_name> [rsync flags]}"
shift

echo "Pushing $LOCAL_CORPORA_DIR/$corpus -> $CLUSTER_SSH:$REMOTE_CORPORA_DIR/$corpus"
rsync -avP "$@" "$LOCAL_CORPORA_DIR/$corpus/" "$CLUSTER_SSH:$REMOTE_CORPORA_DIR/$corpus/"
