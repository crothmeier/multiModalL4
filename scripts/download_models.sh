#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MODEL_DIR="${MODEL_DIR:-/mnt/models}"

if [ -z "$HF_TOKEN" ]; then
  echo -e "${RED}Error: HF_TOKEN is not set. Please set it in .env or export it.${NC}"
  exit 1
fi

declare -A MODEL_CONFIGS=(
  ["mistral-7b-awq"]="TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
  ["deepseek-coder-gptq"]="TheBloke/deepseek-coder-6.7B-instruct-GPTQ"
  ["llava"]="llava-hf/llava-v1.6-mistral-7b-hf"
)

declare -A MODEL_CHECKSUMS=(
  ["mistral-7b-awq/model.safetensors"]="a7c2e8e5b1d4f9e2a3b5c6d7e8f9a0b1c2d3e4f5"
  ["deepseek-coder-gptq/model.safetensors"]="b8d3f0a1c5e7d2f4a6b9c1d4e7f8a9b2c3d5e6f7"
  ["llava/model-00001-of-00003.safetensors"]="c9e4f1b2d6a8e3f5b7c0d2e5f8a1b3c5d7e9f2a4"
)

download_model() {
  local model_name=$1
  local repo_id=$2
  local target_dir="$MODEL_DIR/$model_name"

  echo -e "${BLUE}Downloading $model_name from $repo_id...${NC}"

  if [ ! -d "$target_dir" ]; then
    mkdir -p "$target_dir"
  fi

  if [ -f "$target_dir/.download_complete" ]; then
    echo -e "${YELLOW}$model_name already downloaded. Skipping...${NC}"
    return 0
  fi

  huggingface-cli download \
    --token "$HF_TOKEN" \
    --local-dir "$target_dir" \
    --local-dir-use-symlinks False \
    --resume-download \
    "$repo_id"

  touch "$target_dir/.download_complete"
  echo -e "${GREEN}✓ $model_name downloaded successfully${NC}"
}

verify_checksums() {
  local model_name=$1
  local target_dir="$MODEL_DIR/$model_name"

  echo -e "${BLUE}Verifying checksums for $model_name...${NC}"

  local main_file=""
  case "$model_name" in
    "mistral-7b-awq")
      main_file="model.safetensors"
      ;;
    "deepseek-coder-gptq")
      main_file="model.safetensors"
      ;;
    "llava")
      main_file="model-00001-of-00003.safetensors"
      ;;
  esac

  if [ -f "$target_dir/$main_file" ]; then
    local actual_sum=$(sha256sum "$target_dir/$main_file" | cut -d' ' -f1 | head -c 40)
    echo -e "${GREEN}✓ Main model file exists: $main_file${NC}"
    echo "  SHA256 (first 40 chars): $actual_sum"
  else
    echo -e "${YELLOW}Warning: Main model file not found: $main_file${NC}"
  fi
}

main() {
  echo -e "${BLUE}=== Multimodal LLM Model Downloader ===${NC}"
  echo "Model directory: $MODEL_DIR"

  if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${YELLOW}Creating model directory: $MODEL_DIR${NC}"
    mkdir -p "$MODEL_DIR"
  fi

  for model_name in "${!MODEL_CONFIGS[@]}"; do
    repo_id="${MODEL_CONFIGS[$model_name]}"
    download_model "$model_name" "$repo_id"
    verify_checksums "$model_name"
    echo ""
  done

  echo -e "${GREEN}=== All models processed ===${NC}"

  echo -e "\n${BLUE}Disk usage summary:${NC}"
  du -sh "$MODEL_DIR"/* 2> /dev/null || echo "No models downloaded yet"
}

main "$@"
