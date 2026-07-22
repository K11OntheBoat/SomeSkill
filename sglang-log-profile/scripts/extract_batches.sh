#!/usr/bin/env bash
# Extract Prefill/Decode batch lines from raw SGLang worker logs.
# Usage: extract_batches.sh '<raw log glob>' <output file>
# Example: extract_batches.sh '/path/logs/sgl_worker_node*_*.log' run_dir/data/batches_all.log
set -euo pipefail
GLOB="$1"
OUT="$2"
mkdir -p "$(dirname "$OUT")"
# shellcheck disable=SC2086
grep -h -E "(Prefill batch|Decode batch)" $GLOB > "$OUT"
echo "wrote $(wc -l < "$OUT") batch lines to $OUT"
