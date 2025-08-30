#!/usr/bin/env bash
set -euo pipefail
BASE_SHA="${1:-origin/main}"
CHANGED_FILES=$(git diff --name-only "${BASE_SHA}...HEAD" 2> /dev/null || git diff --name-only "${BASE_SHA}" 2> /dev/null || echo "")
[ -z "$CHANGED_FILES" ] && {
  echo "No changed files detected"
  exit 0
}

# Block classic model/backup/binary blobs often seen in LLaVA/Triton repos
FORBIDDEN_REGEX='((^|/)docker-compose\.yml\.backup-|\.(gguf|safetensors|pt|onnx|engine|plan|ckpt|bin|npz|npy|tar|zst|gz|zip|parquet)$)'

BAD=""
while IFS= read -r f; do
  echo "$f" | grep -qE "$FORBIDDEN_REGEX" && BAD+="$f"$'\n'
done <<< "$CHANGED_FILES"

if [ -n "$BAD" ]; then
  echo "❌ Forbidden artifacts detected:"
  printf "%b" "$BAD"
  exit 1
else
  echo "✅ No forbidden artifacts detected"
fi
