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
# --progress, but deliberately not -P: -P also means --partial, which leaves a
# half-written WAV at its real destination path when a transfer is interrupted. The
# corpus then looks complete while one file is truncated, and training only dies much
# later inside the DataLoader. Plain rsync stages to a temp file and renames on
# success, so an interrupted file is simply absent and the next push re-sends it.
rsync -av --progress "$@" "$LOCAL_CORPORA_DIR/$corpus/" "$CLUSTER_SSH:$REMOTE_CORPORA_DIR/$corpus/"

# A push can exit 0 and still be short a file, so confirm the remote matches before a
# SLURM job is ever queued against it. Skipped for dry runs, which transfer nothing.
case " $* " in *" -n "*|*" --dry-run "*) exit 0 ;; esac
echo "Verifying remote corpus matches local..."
pending="$(rsync -an --itemize-changes \
    "$LOCAL_CORPORA_DIR/$corpus/" "$CLUSTER_SSH:$REMOTE_CORPORA_DIR/$corpus/" \
    | grep '^[<>]' || true)"
if [ -n "$pending" ]; then
    echo "ERROR: remote corpus still differs from local after the push:" >&2
    printf '%s\n' "$pending" >&2
    echo "Re-run this script; do not train until it verifies clean." >&2
    exit 1
fi
echo "Verified: '$corpus' on the cluster matches local."
