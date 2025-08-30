#!/usr/bin/env bash
set -euo pipefail

BASE_SHA="${1:-${GITHUB_BASE_SHA:-}}"
if [[ -z "${BASE_SHA}" ]]; then
  echo "BASE_SHA not provided; defaulting to main ancestry."
  BASE_SHA="origin/main"
fi

# Ensure we have base commit locally
git fetch --no-tags --depth=1 origin "${BASE_SHA}" || true

# Collect changed files only
if git rev-parse --quiet --verify "${BASE_SHA}" >/dev/null; then
  mapfile -t changed < <(git diff --name-only "${BASE_SHA}...HEAD")
else
  # Fallback: compare to main branch tip
  git fetch --no-tags --depth=1 origin main || true
  mapfile -t changed < <(git diff --name-only "origin/main...HEAD")
fi

if [[ ${#changed[@]} -eq 0 ]]; then
  echo "No changed files detected."
  exit 0
fi

regex='(^|/)docker-compose\.yml\.backup-|\.gguf$|\.safetensors$|\.pt$|\.onnx$|\.ckpt$|\.bin$|\.npz$|\.npy$|\.tar$|\.zst$|\.gz$|\.zip$|\.parquet$'
violations=()

for f in "${changed[@]}"; do
  if [[ "$f" =~ $regex ]]; then
    violations+=("$f")
  fi
done

if [[ ${#violations[@]} -gt 0 ]]; then
  echo "Forbidden artifacts detected in PR changes:"
  for v in "${violations[@]}"; do
    echo " - $v"
  done
  exit 1
fi

echo "No forbidden artifacts in PR changes."
