#!/usr/bin/env bash
# Usage: check_triton_config.sh <model_repo_dir> <triton_image>
# Validates Triton model configs for proto/syntax errors

set -euo pipefail
MODEL_REPO="$1"
IMAGE="$2"

echo "Running Triton dry-run on ${MODEL_REPO}..."

# Run Triton in dry-run mode; capture logs
docker run --rm \
  -v "$(pwd)/${MODEL_REPO}:/models" \
  "${IMAGE}" \
  tritonserver \
    --model-repository=/models \
    --dry-run \
    --log-verbose=0 \
    --exit-on-error=true \
| tee triton_lint.log

echo "Scanning log for proto/config errors..."
if grep -Ei '(Unknown field|invalid argument|failed to|ERROR:)' triton_lint.log; then
  echo "❌ Triton config validation failed"
  exit 1
fi

echo "✅ Triton configs lint-clean"
