#!/bin/bash
set -euo pipefail

# Model structure validation script
MODELS_DIR="${MODELS_DIR:-/mnt/models}"
ERRORS=0

echo "=== Model Directory Structure Validation ==="
echo "Checking models in: $MODELS_DIR"
echo

# Function to check model directory
check_model() {
  local name="$1"
  local dir="$2"
  shift 2
  local required_files=("$@")

  echo -n "Checking $name... "

  if [ ! -d "$dir" ]; then
    echo "❌ Directory missing: $dir"
    ((ERRORS++))
    return
  fi

  local missing=()
  for file in "${required_files[@]}"; do
    if [ ! -f "$dir/$file" ]; then
      missing+=("$file")
    fi
  done

  if [ ${#missing[@]} -eq 0 ]; then
    echo "✓ All files present"
  else
    echo "❌ Missing files: ${missing[*]}"
    ((ERRORS++))
  fi
}

# Check each model
check_model "Mistral AWQ" "$MODELS_DIR/mistral-awq" \
  "model.safetensors" "config.json" "tokenizer.json"

check_model "DeepSeek GPTQ" "$MODELS_DIR/deepseek-gptq" \
  "model.safetensors" "config.json" "tokenizer.json" "quantize_config.json"

check_model "LLaVA 7B" "$MODELS_DIR/llava-7b" \
  "config.json" "tokenizer.model" "tokenizer_config.json"

echo
if [ $ERRORS -eq 0 ]; then
  echo "✅ All model directories valid!"
  exit 0
else
  echo "❌ Found $ERRORS errors in model structure"
  exit 1
fi
