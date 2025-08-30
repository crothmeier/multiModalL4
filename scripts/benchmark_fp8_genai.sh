#!/bin/bash
# GenAI-Perf benchmarking script for LLaVA FP8
# Replaces homegrown benchmarking with NVIDIA's official tool

set -euo pipefail

# Configuration
MODEL_NAME="${MODEL_NAME:-llava_fp8}"
TRITON_URL="${TRITON_URL:-localhost:8000}"
RESULTS_DIR="$(pwd)/benchmark_results/$(date +%Y%m%d_%H%M%S)"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "Starting GenAI-Perf benchmark for $MODEL_NAME"
echo "Results will be saved to: $RESULTS_DIR"

# Run comprehensive benchmark with various concurrency levels
docker run --rm \
  --gpus all \
  --network host \
  -v "$RESULTS_DIR:/results" \
  nvcr.io/nvidia/tritonserver:24.06-py3-sdk \
  genai-perf \
  --model "$MODEL_NAME" \
  --backend tensorrtllm \
  --endpoint "v2/models/$MODEL_NAME/generate" \
  --endpoint-type triton \
  --service-kind triton \
  --url "$TRITON_URL" \
  --num-prompts "$NUM_PROMPTS" \
  --random-seed 42 \
  --synthetic-input-tokens-mean 1024 \
  --synthetic-input-tokens-stddev 256 \
  --output-tokens-mean 128 \
  --output-tokens-stddev 32 \
  --concurrency 1 2 4 8 16 \
  --percentile 50 90 95 99 \
  --measurement-interval 10000 \
  --profile-export-file "/results/fp8_profile.json" \
  --generate-plots \
  --verbose

# Run memory bandwidth specific test for L4
echo "Running L4-specific memory bandwidth test..."
docker run --rm \
  --gpus all \
  --network host \
  -v "$RESULTS_DIR:/results" \
  nvcr.io/nvidia/tritonserver:24.06-py3-sdk \
  genai-perf \
  --model "$MODEL_NAME" \
  --backend tensorrtllm \
  --endpoint "v2/models/$MODEL_NAME/generate" \
  --endpoint-type triton \
  --service-kind triton \
  --url "$TRITON_URL" \
  --num-prompts 100 \
  --batch-size 1 2 4 8 \
  --synthetic-input-tokens-mean 2048 \
  --output-tokens-mean 256 \
  --concurrency 1 \
  --profile-export-file "/results/fp8_bandwidth_test.json" \
  --verbose

# Compare with FP16 if available
if curl -s "http://$TRITON_URL/v2/models/llava_fp16/ready" | grep -q "true"; then
  echo "Running comparison benchmark against FP16 model..."

  docker run --rm \
    --gpus all \
    --network host \
    -v "$RESULTS_DIR:/results" \
    nvcr.io/nvidia/tritonserver:24.06-py3-sdk \
    genai-perf \
    --model llava_fp16 \
    --backend tensorrtllm \
    --endpoint v2/models/llava_fp16/generate \
    --endpoint-type triton \
    --service-kind triton \
    --url "$TRITON_URL" \
    --num-prompts 500 \
    --random-seed 42 \
    --synthetic-input-tokens-mean 1024 \
    --output-tokens-mean 128 \
    --concurrency 4 \
    --percentile 50 90 95 99 \
    --profile-export-file "/results/fp16_profile.json" \
    --verbose
fi

# Parse and summarize results
echo "Parsing benchmark results..."
python3 - "$RESULTS_DIR" << 'EOF'
import json
import sys
import os

results_dir = sys.argv[1]

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def print_summary(name, data):
    print(f"\n=== {name} Results ===")
    if 'latency' in data:
        lat = data['latency']
        print(f"Latency P50: {lat.get('p50', 'N/A')}ms")
        print(f"Latency P95: {lat.get('p95', 'N/A')}ms")
        print(f"Latency P99: {lat.get('p99', 'N/A')}ms")

    if 'throughput' in data:
        print(f"Throughput: {data['throughput']} tokens/sec")

    if 'memory' in data:
        print(f"GPU Memory: {data['memory'].get('used_gb', 'N/A')}GB")

# Load FP8 results
fp8_path = os.path.join(results_dir, 'fp8_profile.json')
if os.path.exists(fp8_path):
    fp8_data = load_results(fp8_path)
    print_summary("FP8 Model", fp8_data)

# Load FP16 results if available
fp16_path = os.path.join(results_dir, 'fp16_profile.json')
if os.path.exists(fp16_path):
    fp16_data = load_results(fp16_path)
    print_summary("FP16 Model", fp16_data)

    # Calculate improvements
    print("\n=== FP8 vs FP16 Comparison ===")
    if 'latency' in fp8_data and 'latency' in fp16_data:
        fp8_p99 = fp8_data['latency']['p99']
        fp16_p99 = fp16_data['latency']['p99']
        improvement = ((fp16_p99 - fp8_p99) / fp16_p99) * 100
        print(f"Latency improvement: {improvement:.1f}%")

    if 'memory' in fp8_data and 'memory' in fp16_data:
        fp8_mem = fp8_data['memory']['used_gb']
        fp16_mem = fp16_data['memory']['used_gb']
        reduction = ((fp16_mem - fp8_mem) / fp16_mem) * 100
        print(f"Memory reduction: {reduction:.1f}%")

print(f"\nFull results saved in: {results_dir}")
EOF

echo "Benchmark complete! Results saved in: $RESULTS_DIR"

# Generate performance report
cat > "$RESULTS_DIR/performance_report.md" << EOF
# LLaVA FP8 Performance Report

## Test Configuration
- Model: $MODEL_NAME
- GPU: NVIDIA L4 (24GB)
- Triton Version: 24.06
- Test Date: $(date)
- Number of Prompts: $NUM_PROMPTS

## Key Metrics
- Target Throughput: >100 img/sec
- Target Latency P99: <500ms
- Memory Usage: <20GB

## Results
See JSON files for detailed metrics:
- fp8_profile.json: Main FP8 benchmark
- fp8_bandwidth_test.json: L4 bandwidth optimization test
- fp16_profile.json: FP16 comparison (if available)

## Recommendations
1. Monitor memory bandwidth utilization on L4
2. Consider reducing batch size if bandwidth saturates
3. Test FP8 KV-cache carefully before production
EOF

echo "Performance report generated: $RESULTS_DIR/performance_report.md"
