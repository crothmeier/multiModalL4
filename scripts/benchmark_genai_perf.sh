#!/bin/bash
# GenAI-Perf Benchmark Script for FP8 LLaVA Model
# Using NVIDIA's official GenAI-Perf container (24.06)

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
RESULTS_DIR="${PROJECT_ROOT}/benchmarks/$(date +%Y%m%d_%H%M%S)"
TRITON_ENDPOINT="${TRITON_ENDPOINT:-http://localhost:8002}"

# Default parameters
MODEL_NAME="${MODEL_NAME:-llava_fp8}"
BACKEND="${BACKEND:-tensorrtllm}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 4 8 16}"
INPUT_TOKENS_MEAN="${INPUT_TOKENS_MEAN:-1024}"
INPUT_TOKENS_STDDEV="${INPUT_TOKENS_STDDEV:-256}"
OUTPUT_TOKENS_MEAN="${OUTPUT_TOKENS_MEAN:-128}"
OUTPUT_TOKENS_STDDEV="${OUTPUT_TOKENS_STDDEV:-32}"
DURATION="${DURATION:-60}"  # seconds

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

# Check prerequisites
check_prerequisites() {
    log_section "Checking Prerequisites"

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check if Triton is running
    if ! curl -s "${TRITON_ENDPOINT}/v2/health/ready" > /dev/null 2>&1; then
        log_warn "Triton server not responding at ${TRITON_ENDPOINT}"
        log_info "Attempting to check container status..."
        docker ps | grep triton || true
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_info "Triton server is ready at ${TRITON_ENDPOINT}"
    fi

    # Create results directory
    mkdir -p "${RESULTS_DIR}"
    log_info "Results will be saved to: ${RESULTS_DIR}"
}

# Run GenAI-Perf benchmark
run_benchmark() {
    local concurrency=$1
    local output_file="${RESULTS_DIR}/fp8_profile_c${concurrency}.json"
    local csv_file="${RESULTS_DIR}/fp8_results_c${concurrency}.csv"

    log_info "Running benchmark with concurrency=${concurrency}..."

    # Build the GenAI-Perf command
    local cmd="docker run --rm --gpus all --net host \
        -v ${RESULTS_DIR}:/results \
        nvcr.io/nvidia/genai-perf:24.06.0 \
        genai-perf \
            --endpoint ${TRITON_ENDPOINT}/v2/models/${MODEL_NAME}/generate \
            --service-kind triton \
            --model ${MODEL_NAME} \
            --backend ${BACKEND} \
            --profile-export-file /results/fp8_profile_c${concurrency}.json \
            --concurrency ${concurrency} \
            --synthetic-input-tokens-mean ${INPUT_TOKENS_MEAN} \
            --synthetic-input-tokens-stddev ${INPUT_TOKENS_STDDEV} \
            --output-tokens-mean ${OUTPUT_TOKENS_MEAN} \
            --output-tokens-stddev ${OUTPUT_TOKENS_STDDEV} \
            --measurement-interval ${DURATION} \
            --stability-percentage 999 \
            --percentile 50 90 95 99 99.9"

    log_info "Executing: ${cmd}"

    # Run the benchmark
    if eval "${cmd}" 2>&1 | tee "${RESULTS_DIR}/genai_perf_c${concurrency}.log"; then
        log_info "Benchmark completed for concurrency=${concurrency}"

        # Extract key metrics from the log
        extract_metrics "${RESULTS_DIR}/genai_perf_c${concurrency}.log" "${csv_file}"
    else
        log_error "Benchmark failed for concurrency=${concurrency}"
        return 1
    fi
}

# Extract metrics from GenAI-Perf output
extract_metrics() {
    local log_file=$1
    local csv_file=$2

    log_info "Extracting metrics from ${log_file}..."

    # Parse the output and create CSV
    {
        echo "Metric,Value,Unit"
        grep -E "Throughput|Latency|Time to First Token" "${log_file}" | \
            sed 's/│//g' | sed 's/║//g' | \
            awk -F'[[:space:]]{2,}' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $1","$2}'
    } > "${csv_file}"

    log_info "Metrics saved to ${csv_file}"
}

# Run warmup
run_warmup() {
    log_section "Running Warmup"

    log_info "Sending warmup requests to ensure model is loaded..."

    docker run --rm --gpus all --net host \
        nvcr.io/nvidia/genai-perf:24.06.0 \
        genai-perf \
            --endpoint ${TRITON_ENDPOINT}/v2/models/${MODEL_NAME}/generate \
            --service-kind triton \
            --model ${MODEL_NAME} \
            --backend ${BACKEND} \
            --concurrency 1 \
            --synthetic-input-tokens-mean 128 \
            --output-tokens-mean 32 \
            --measurement-interval 10 \
            --warmup || log_warn "Warmup failed, continuing anyway"

    log_info "Warmup complete"
}

# Generate comparison report
generate_report() {
    log_section "Generating Benchmark Report"

    local report_file="${RESULTS_DIR}/benchmark_report.md"

    cat > "${report_file}" <<EOF
# GenAI-Perf Benchmark Report

## Configuration
- **Model**: ${MODEL_NAME}
- **Backend**: ${BACKEND}
- **Endpoint**: ${TRITON_ENDPOINT}
- **Date**: $(date)
- **Input Tokens**: ${INPUT_TOKENS_MEAN}±${INPUT_TOKENS_STDDEV}
- **Output Tokens**: ${OUTPUT_TOKENS_MEAN}±${OUTPUT_TOKENS_STDDEV}
- **Duration**: ${DURATION}s

## Results by Concurrency

EOF

    # Add results for each concurrency level
    for c in ${CONCURRENCY_LEVELS}; do
        if [[ -f "${RESULTS_DIR}/fp8_results_c${c}.csv" ]]; then
            echo "### Concurrency = ${c}" >> "${report_file}"
            echo '```csv' >> "${report_file}"
            cat "${RESULTS_DIR}/fp8_results_c${c}.csv" >> "${report_file}"
            echo '```' >> "${report_file}"
            echo "" >> "${report_file}"
        fi
    done

    # Add green-light criteria check
    cat >> "${report_file}" <<EOF

## Green-Light Criteria Check

| Metric | Target | Status |
|--------|--------|--------|
| P99 Latency | ≤ 500ms | $(check_p99_latency) |
| Throughput | ≥ 100 RPS | $(check_throughput) |
| GPU Mem BW | ≤ 85% of 300 GB/s | $(check_bandwidth) |

## Recommendations

EOF

    # Add recommendations based on results
    if [[ $(check_p99_latency) == "❌ FAIL" ]]; then
        echo "- P99 latency exceeds target. Consider reducing batch size or enabling more aggressive optimization." >> "${report_file}"
    fi

    if [[ $(check_throughput) == "❌ FAIL" ]]; then
        echo "- Throughput below target. Check for preprocessing bottlenecks or increase concurrency." >> "${report_file}"
    fi

    echo "- Monitor vllm_spec_decode_efficiency metric for Arctic speculation performance" >> "${report_file}"
    echo "- Watch for FP8 accuracy degradation in production traffic" >> "${report_file}"

    log_info "Report generated: ${report_file}"
}

# Check P99 latency (placeholder - parse from actual results)
check_p99_latency() {
    # This would parse actual results from GenAI-Perf output
    echo "⏳ PENDING"
}

# Check throughput (placeholder - parse from actual results)
check_throughput() {
    # This would parse actual results from GenAI-Perf output
    echo "⏳ PENDING"
}

# Check bandwidth utilization
check_bandwidth() {
    if command -v nvidia-smi &> /dev/null; then
        local util=$(nvidia-smi dmon -c 1 | tail -1 | awk '{print $9}')
        if [[ -n "$util" ]] && [[ "$util" -lt 85 ]]; then
            echo "✅ PASS (${util}%)"
        else
            echo "❌ FAIL (${util}%)"
        fi
    else
        echo "⏳ PENDING"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       GenAI-Perf FP8 Benchmark Script      ║${NC}"
    echo -e "${BLUE}║              NVIDIA SDK 24.06              ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}\n"

    check_prerequisites
    run_warmup

    log_section "Running Benchmarks"

    # Run benchmarks for each concurrency level
    for concurrency in ${CONCURRENCY_LEVELS}; do
        if ! run_benchmark "${concurrency}"; then
            log_warn "Skipping remaining benchmarks due to failure"
            break
        fi

        # Add delay between runs
        if [[ "${concurrency}" != "${CONCURRENCY_LEVELS##* }" ]]; then
            log_info "Waiting 10 seconds before next run..."
            sleep 10
        fi
    done

    generate_report

    echo -e "\n${GREEN}═══════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ Benchmark completed successfully!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════${NC}\n"

    log_info "Results directory: ${RESULTS_DIR}"
    log_info "View report: cat ${RESULTS_DIR}/benchmark_report.md"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --endpoint)
            TRITON_ENDPOINT="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY_LEVELS="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model NAME         Model name (default: llava_fp8)"
            echo "  --endpoint URL       Triton endpoint (default: http://localhost:8002)"
            echo "  --concurrency LEVELS Space-separated concurrency levels (default: 1 4 8 16)"
            echo "  --duration SECONDS   Test duration in seconds (default: 60)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main
main
