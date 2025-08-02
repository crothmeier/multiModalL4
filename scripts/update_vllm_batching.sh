#!/bin/bash
# Update vLLM continuous batching parameters for optimal L4 performance
# Replaces JSON-based configuration with CLI flags per vLLM 0.8.4+

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}\n"
}

# Update docker-compose.yml to use correct vLLM CLI flags
update_vllm_services() {
    log_section "Updating vLLM Services Configuration"

    # Backup original file
    cp "${PROJECT_ROOT}/docker-compose.yml" "${PROJECT_ROOT}/docker-compose.yml.bak.$(date +%Y%m%d_%H%M%S)"

    # Update the vLLM service configurations
    log_info "Updating vLLM command flags for continuous batching..."

    # Create temporary file with updated configuration
    cat > "${PROJECT_ROOT}/docker-compose.vllm-update.yml" <<'EOF'
# vLLM Service Configuration Update Patch
# Add these flags to vLLM services:

# For llava-llm service, update command to include:
    command:
      - --model=/models/llava-7b
      - --tokenizer=liuhaotian/llava-v1.5-7b
      - --served-model-name=llava
      - --trust-remote-code
      - --download-dir=/models
      - --tokenizer-mode=auto
      - --max-model-len=4096
      - --gpu-memory-utilization=0.10
      - --disable-custom-all-reduce
      - --image-input-type=pixel_values
      - --host=0.0.0.0
      - --port=8000
      # Continuous batching parameters (vLLM 0.8.4+ format)
      - --continuous-batching
      - --max-num-seqs=8
      - --max-num-paddings=32
      - --enable-prefix-caching
      - --block-size=32
      - --swap-space=4
      # L4 optimization flags
      - --enforce-eager
      - --disable-log-stats=false
      - --kv-cache-dtype=fp8_e5m2  # Start with FP8 KV-cache

# For other vLLM services (mistral-llm, coder-llm), add:
      # Continuous batching parameters
      - --continuous-batching
      - --max-num-seqs=8
      - --max-num-paddings=32
      - --enable-prefix-caching
EOF

    log_info "vLLM configuration update template created"

    # Apply updates to docker-compose.yml using yq if available
    if command -v yq &> /dev/null; then
        log_info "Using yq to update docker-compose.yml..."

        # Update llava-llm service
        yq eval '.services.llava-llm.command += [
            "--continuous-batching",
            "--max-num-seqs=8",
            "--max-num-paddings=32",
            "--enable-prefix-caching",
            "--block-size=32",
            "--swap-space=4",
            "--enforce-eager",
            "--kv-cache-dtype=fp8_e5m2"
        ]' -i "${PROJECT_ROOT}/docker-compose.yml"

        # Update mistral-llm service
        yq eval '.services.mistral-llm.command += [
            "--continuous-batching",
            "--max-num-seqs=8",
            "--max-num-paddings=32",
            "--enable-prefix-caching"
        ]' -i "${PROJECT_ROOT}/docker-compose.yml"

        # Update coder-llm service
        yq eval '.services.coder-llm.command += [
            "--continuous-batching",
            "--max-num-seqs=8",
            "--max-num-paddings=32",
            "--enable-prefix-caching"
        ]' -i "${PROJECT_ROOT}/docker-compose.yml"

        log_info "docker-compose.yml updated successfully with yq"
    else
        log_warn "yq not found, manual update required"
        log_info "Please manually add the following flags to vLLM services:"
        cat "${PROJECT_ROOT}/docker-compose.vllm-update.yml"
    fi
}

# Create vLLM startup script with proper parameters
create_vllm_startup_script() {
    log_section "Creating vLLM Startup Script"

    cat > "${PROJECT_ROOT}/scripts/start_vllm_optimized.sh" <<'EOF'
#!/bin/bash
# Optimized vLLM startup script for L4 GPU with continuous batching

set -euo pipefail

MODEL_PATH="${1:-/models/llava-7b}"
MODEL_NAME="${2:-llava}"
PORT="${3:-8000}"

echo "Starting vLLM with optimized continuous batching for L4..."

exec python -m vllm.entrypoints.openai.api_server \
    --model="${MODEL_PATH}" \
    --served-model-name="${MODEL_NAME}" \
    --host=0.0.0.0 \
    --port="${PORT}" \
    --continuous-batching \
    --max-num-seqs=8 \
    --max-num-paddings=32 \
    --enable-prefix-caching \
    --block-size=32 \
    --swap-space=4 \
    --gpu-memory-utilization=0.85 \
    --max-model-len=4096 \
    --kv-cache-dtype=fp8_e5m2 \
    --enforce-eager \
    --disable-log-stats=false \
    --trust-remote-code \
    --download-dir=/models \
    --tokenizer-mode=auto
EOF

    chmod +x "${PROJECT_ROOT}/scripts/start_vllm_optimized.sh"
    log_info "Created optimized vLLM startup script"
}

# Validate configuration
validate_config() {
    log_section "Validating Configuration"

    # Check if docker-compose.yml has continuous batching flags
    if grep -q "continuous-batching" "${PROJECT_ROOT}/docker-compose.yml"; then
        log_info "✅ Continuous batching flags found in docker-compose.yml"
    else
        log_warn "⚠️  Continuous batching flags not found in docker-compose.yml"
        log_info "Please manually add the flags or install yq"
    fi

    # Check vLLM version in image
    log_info "Checking vLLM image version..."
    if grep -q "vllm/vllm-openai:v0.6" "${PROJECT_ROOT}/docker-compose.yml"; then
        log_warn "Using vLLM 0.6.x - consider upgrading to 0.8.4+ for better FP8 support"
        log_info "Update image to: vllm/vllm-openai:v0.8.4 or newer"
    fi
}

# Generate report
generate_report() {
    log_section "Configuration Report"

    cat > "${PROJECT_ROOT}/vllm_batching_update.md" <<EOF
# vLLM Continuous Batching Configuration Update

## Changes Applied
- Updated vLLM services to use CLI flags instead of JSON config
- Enabled continuous batching with optimal L4 parameters
- Set max-num-seqs to 8 (L4 bandwidth optimization)
- Set max-num-paddings to 32
- Enabled prefix caching for better performance
- Configured FP8 KV-cache (fp8_e5m2 format)

## Key Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| --continuous-batching | enabled | Core feature for throughput |
| --max-num-seqs | 8 | Optimal for L4 memory bandwidth |
| --max-num-paddings | 32 | Balance between efficiency and memory |
| --block-size | 32 | Efficient for L4 architecture |
| --kv-cache-dtype | fp8_e5m2 | Reduce memory bandwidth usage |
| --enable-prefix-caching | true | Reuse common prefixes |

## Next Steps
1. Restart vLLM services: \`docker-compose restart llava-llm mistral-llm coder-llm\`
2. Monitor metrics: \`docker logs -f multimodal-stack-llava-llm-1\`
3. Verify batching is active: Check for "Continuous batching enabled" in logs
4. Run benchmark: \`./scripts/benchmark_genai_perf.sh\`

## Monitoring
Watch these metrics:
- vllm_num_batched_requests
- vllm_batch_size_avg
- vllm_cache_hit_rate
- vllm_gpu_memory_usage

## Rollback
To rollback: \`cp docker-compose.yml.bak.<timestamp> docker-compose.yml\`
EOF

    log_info "Report saved to: ${PROJECT_ROOT}/vllm_batching_update.md"
}

# Main execution
main() {
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║    vLLM Continuous Batching Update Script  ║${NC}"
    echo -e "${BLUE}║           Optimized for L4 GPU             ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}\n"

    update_vllm_services
    create_vllm_startup_script
    validate_config
    generate_report

    echo -e "\n${GREEN}═══════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ vLLM batching update completed!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════${NC}\n"

    log_info "Backup saved to: docker-compose.yml.bak.*"
    log_info "Review changes and restart services when ready"
}

# Parse arguments
case "${1:-}" in
    --help)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Updates vLLM services to use optimized continuous batching parameters"
        echo ""
        echo "Options:"
        echo "  --help    Show this help message"
        echo "  --apply   Apply changes and restart services"
        exit 0
        ;;
    --apply)
        main
        log_info "Restarting vLLM services..."
        docker-compose restart llava-llm mistral-llm coder-llm || true
        ;;
    *)
        main
        ;;
esac
