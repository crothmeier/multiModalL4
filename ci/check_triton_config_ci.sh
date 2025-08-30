#!/usr/bin/env bash
# CI-safe Triton config validation (no Docker required)
set -euo pipefail

MODEL_REPO="${1:-triton/models}"

echo "Running CI validation on ${MODEL_REPO}..."

# Basic structural validation for Triton configs
for config in "${MODEL_REPO}"/*/config.pbtxt; do
  if [ -f "$config" ]; then
    model_name=$(basename "$(dirname "$config")")
    echo "Checking $model_name..."

    # Check required fields
    grep -q "name:" "$config" || {
      echo "❌ Missing 'name' field in $config"
      exit 1
    }
    grep -q -E "(backend:|platform:)" "$config" || {
      echo "❌ Missing 'backend' or 'platform' in $config"
      exit 1
    }

    # Check for invalid KIND_AUTO + gpus combination
    if grep -q "KIND_AUTO" "$config" && grep -q "gpus:" "$config"; then
      echo "❌ Invalid: KIND_AUTO with explicit gpus in $config"
      exit 1
    fi

    echo "  ✅ $model_name config valid"
  fi
done

echo "✅ All Triton configs are CI-valid"
