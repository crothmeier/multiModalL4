#!/bin/bash
set -euo pipefail

# Model fetch script with verified SHA-256 hashes
MODELS_DIR="${MODELS_DIR:-/mnt/models}"
mkdir -p "$MODELS_DIR"

# Model definitions with actual SHA256 checksums from HuggingFace
# Mistral-7B-Instruct-v0.2-AWQ (4.65GB)
MISTRAL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-AWQ/resolve/26811efa49a9e4dc1e5b30f0f10d271c47a0c6fc/mistral-7b-instruct-v0.2-AWQ.safetensors"
MISTRAL_SHA="3a5715f3e51e8b852943098259a80520f1b569e9f884db58ba447de80e1f4408"
MISTRAL_DIR="$MODELS_DIR/mistral-awq"

# DeepSeek-Coder-6.7B-Instruct-GPTQ (3.9GB)
DEEPSEEK_CODER_URL="https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GPTQ/resolve/gptq-4bit-128g-actorder_True/model.safetensors"
DEEPSEEK_CODER_SHA="d5e09d78b8ac4e5dafb4f5a768864de5bd3f7b23ad8a96e5b397c5740b0acbf1"
DEEPSEEK_CODER_DIR="$MODELS_DIR/deepseek-gptq"

# LLaVA-v1.5-7B (13GB) - Compatible with vLLM
LLAVA_BASE="https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main"
LLAVA_DIR="$MODELS_DIR/llava-7b"
LLAVA_FILES=(
    "config.json"
    "generation_config.json"
    "model-00001-of-00003.safetensors"
    "model-00002-of-00003.safetensors"
    "model-00003-of-00003.safetensors"
    "model.safetensors.index.json"
    "tokenizer.model"
    "tokenizer_config.json"
    "special_tokens_map.json"
)

download_model_file() {
    local url="$1"
    local output="$2"
    local expected_sha="${3:-}"

    if [ -f "$output" ] && [ -n "$expected_sha" ]; then
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

    if [ -n "$expected_sha" ]; then
        echo "→ Verifying checksum..."
        echo "$expected_sha  $output" | sha256sum -c -
    fi
}

# Main downloads
echo "=== Multimodal LLM Model Fetcher ==="
echo "Target directory: $MODELS_DIR"
echo

# Mistral AWQ
echo "→ Setting up Mistral AWQ..."
mkdir -p "$MISTRAL_DIR"
download_model_file "$MISTRAL_URL" "$MISTRAL_DIR/model.safetensors" "$MISTRAL_SHA"
# Download tokenizer files
for file in config.json tokenizer.json tokenizer_config.json; do
    if [ ! -f "$MISTRAL_DIR/$file" ]; then
        wget -q -O "$MISTRAL_DIR/$file" \
            "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/$file"
    fi
done

# DeepSeek Coder GPTQ
echo "→ Setting up DeepSeek Coder GPTQ..."
mkdir -p "$DEEPSEEK_CODER_DIR"
download_model_file "$DEEPSEEK_CODER_URL" "$DEEPSEEK_CODER_DIR/model.safetensors" "$DEEPSEEK_CODER_SHA"
# Download config files
for file in config.json tokenizer.json tokenizer_config.json quantize_config.json; do
    if [ ! -f "$DEEPSEEK_CODER_DIR/$file" ]; then
        wget -q -O "$DEEPSEEK_CODER_DIR/$file" \
            "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GPTQ/resolve/main/$file"
    fi
done

# LLaVA-7B
echo "→ Setting up LLaVA v1.5 7B..."
mkdir -p "$LLAVA_DIR"
for file in "${LLAVA_FILES[@]}"; do
    if [ ! -f "$LLAVA_DIR/$file" ]; then
        echo "  Downloading $file..."
        wget --progress=bar:force -O "$LLAVA_DIR/$file" "$LLAVA_BASE/$file"
    else
        echo "  ✓ $file already present"
    fi
done

echo
echo "✓ All models fetched successfully!"
echo "  - Mistral 7B AWQ: $MISTRAL_DIR/"
echo "  - DeepSeek Coder: $DEEPSEEK_CODER_DIR/"
echo "  - LLaVA 7B: $LLAVA_DIR/"
