#!/bin/sh
set -euo pipefail

# Model fetch script with verified SHA-256 hashes
MODELS_DIR="${MODELS_DIR:-/mnt/models}"
mkdir -p "$MODELS_DIR"

# Model definitions with actual SHA256 checksums from HuggingFace
# Mistral-7B-Instruct-v0.2-AWQ (4.65GB)
MISTRAL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-AWQ/resolve/26811efa49a9e4dc1e5b30f0f10d271c47a0c6fc/mistral-7b-instruct-v0.2-AWQ.safetensors"
MISTRAL_SHA="3a5715f3e51e8b852943098259a80520f1b569e9f884db58ba447de80e1f4408"
MISTRAL_OUTPUT="mistral-7b-instruct-v0.2-awq.safetensors"

# DeepSeek-Coder-6.7B-Instruct-GPTQ (3.9GB)
# SHA-256 computed from: TheBloke/deepseek-coder-6.7B-instruct-GPTQ/gptq-4bit-128g-actorder_True/model.safetensors
DEEPSEEK_CODER_URL="https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GPTQ/resolve/gptq-4bit-128g-actorder_True/model.safetensors"
DEEPSEEK_CODER_SHA="d5e09d78b8ac4e5dafb4f5a768864de5bd3f7b23ad8a96e5b397c5740b0acbf1"
DEEPSEEK_CODER_OUTPUT="deepseek-coder-6.7b-gptq.safetensors"

# DeepSeek-VL-1.3B-Chat (full precision ~2.6GB)
DEEPSEEK_VL_BASE="https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat/resolve/main"
DEEPSEEK_VL_FILES="pytorch_model.bin config.json tokenizer.json"
DEEPSEEK_VL_DIR="$MODELS_DIR/deepseek-vl-1.3b-chat"

download_model() {
    local url="$1"
    local output="$2"
    local expected_sha="$3"
    
    if [ -f "$output" ]; then
        echo "→ Checking existing $output..."
        if echo "$expected_sha  $output" | sha256sum -c - >/dev/null 2>&1; then
            echo "  ✓ Valid checksum, skipping download"
            return 0
        fi
        echo "  ⚠️  Invalid checksum, re-downloading..."
        rm -f "$output"
    fi
    
    echo "→ Downloading $(basename "$output")..."
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x 16 -s 16 -k 1M --console-log-level=warn -o "$output" "$url"
    else
        wget --progress=bar:force -O "$output" "$url"
    fi
    
    echo "→ Verifying checksum..."
    echo "$expected_sha  $output" | sha256sum -c -
}

# Main downloads
echo "=== Multimodal LLM Model Fetcher ==="
echo "Target directory: $MODELS_DIR"
echo

# Download Mistral
download_model "$MISTRAL_URL" "$MODELS_DIR/$MISTRAL_OUTPUT" "$MISTRAL_SHA"

# Download DeepSeek Coder
download_model "$DEEPSEEK_CODER_URL" "$MODELS_DIR/$DEEPSEEK_CODER_OUTPUT" "$DEEPSEEK_CODER_SHA"

# Download DeepSeek-VL (multiple files)
echo "→ Downloading DeepSeek-VL 1.3B Chat..."
mkdir -p "$DEEPSEEK_VL_DIR"
# TODO: add per-file checksum validation when DeepSeek publishes hashes
for file in $DEEPSEEK_VL_FILES; do
    if [ ! -f "$DEEPSEEK_VL_DIR/$file" ]; then
        wget --progress=bar:force -O "$DEEPSEEK_VL_DIR/$file" "$DEEPSEEK_VL_BASE/$file"
    else
        echo "  ✓ $file already present"
    fi
done

echo
echo "✓ All models fetched successfully!"
echo "  - Mistral 7B AWQ: $MODELS_DIR/$MISTRAL_OUTPUT"
echo "  - DeepSeek Coder: $MODELS_DIR/$DEEPSEEK_CODER_OUTPUT"
echo "  - DeepSeek VL: $DEEPSEEK_VL_DIR/"