#!/bin/bash
set -euo pipefail

# Model fetch script with correct URLs
MODELS_DIR="${MODELS_DIR:-/mnt/models}"
mkdir -p "$MODELS_DIR"

# Model definitions with correct URLs
# Mistral-7B-Instruct-v0.2-AWQ
MISTRAL_AWQ_FILES=(
    "model-00001-of-00002.safetensors"
    "model-00002-of-00002.safetensors"
)
MISTRAL_BASE="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-AWQ/resolve/main"
MISTRAL_DIR="$MODELS_DIR/mistral-awq"

# DeepSeek-Coder-6.7B-Instruct-GPTQ
DEEPSEEK_URL="https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GPTQ/resolve/main/model.safetensors"
DEEPSEEK_DIR="$MODELS_DIR/deepseek-gptq"

# LLaVA-v1.5-7B
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

download_file() {
    local url="$1"
    local output="$2"

    if [ -f "$output" ] && [ -s "$output" ]; then
        echo "  ✓ $(basename "$output") already exists"
        return 0
    fi

    echo "→ Downloading $(basename "$output")..."
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x 16 -s 16 -k 1M --console-log-level=warn -o "$output" "$url"
    else
        wget --progress=bar:force -O "$output" "$url" || {
            echo "  ❌ Failed to download $(basename "$output")"
            return 1
        }
    fi
}

# Main downloads
echo "=== Multimodal LLM Model Fetcher ==="
echo "Target directory: $MODELS_DIR"
echo

# Mistral AWQ
echo "1. Setting up Mistral AWQ..."
mkdir -p "$MISTRAL_DIR"

# Download model shards
for shard in "${MISTRAL_AWQ_FILES[@]}"; do
    download_file "$MISTRAL_BASE/$shard" "$MISTRAL_DIR/$shard"
done

# Download config files
for file in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
    download_file "$MISTRAL_BASE/$file" "$MISTRAL_DIR/$file"
done

# Create combined safetensors file if shards exist
if [ -f "$MISTRAL_DIR/model-00001-of-00002.safetensors" ] && [ -f "$MISTRAL_DIR/model-00002-of-00002.safetensors" ]; then
    echo "  ℹ️  Note: Mistral AWQ uses sharded model files"
fi

# DeepSeek Coder GPTQ
echo -e "\n2. Setting up DeepSeek Coder GPTQ..."
mkdir -p "$DEEPSEEK_DIR"
download_file "$DEEPSEEK_URL" "$DEEPSEEK_DIR/model.safetensors"

# Download config files
for file in config.json tokenizer.json tokenizer_config.json quantize_config.json special_tokens_map.json; do
    download_file "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GPTQ/resolve/main/$file" "$DEEPSEEK_DIR/$file"
done

# LLaVA-7B
echo -e "\n3. Setting up LLaVA v1.5 7B..."
mkdir -p "$LLAVA_DIR"
for file in "${LLAVA_FILES[@]}"; do
    download_file "$LLAVA_BASE/$file" "$LLAVA_DIR/$file"
done

# Also get the preprocessor config for vision
download_file "$LLAVA_BASE/preprocessor_config.json" "$LLAVA_DIR/preprocessor_config.json"

echo -e "\n=== Download Summary ==="
echo "Checking model directories..."

# Verify downloads
for dir in "$MISTRAL_DIR" "$DEEPSEEK_DIR" "$LLAVA_DIR"; do
    if [ -d "$dir" ]; then
        echo -e "\n📁 $dir:"
        find "$dir" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.json" -o -name "*.model" \) -exec ls -lah {} \; | head -5
        total_size=$(du -sh "$dir" | cut -f1)
        echo "   Total size: $total_size"
    fi
done

echo -e "\n✓ Model download complete!"
echo "  Note: Some models use sharded files which vLLM will handle automatically."
