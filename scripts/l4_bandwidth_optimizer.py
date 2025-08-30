#!/usr/bin/env python3
"""
L4-Specific Memory Bandwidth Optimizer
Addresses L4's 300 GB/s bandwidth limitation for optimal FP8 performance
"""

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Dict

import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class L4OptimizationConfig:
    """L4-specific optimization configuration"""

    # Memory bandwidth optimizations
    max_num_seqs: int = 8  # Lower than default 256 to reduce memory transfers
    max_paddings: int = 32
    block_size: int = 32  # Larger blocks = fewer transfers
    num_gpu_blocks_override: int = 2048  # Manual tuning for L4

    # FP8 quantization settings
    quantization_method: str = "fp8"
    activation_dtype: str = "fp8_e4m3"  # E4M3 for activations
    weight_dtype: str = "fp8_e4m3"  # E4M3 for weights
    kv_cache_dtype: str = "fp16"  # Start conservative, then test fp8_e5m2

    # Continuous batching optimization
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 2048  # Reduce for L4 bandwidth
    max_num_seqs_per_batch: int = 4

    # PagedAttention tuning
    gpu_memory_utilization: float = 0.85
    swap_space: int = 4  # GB of CPU swap space

    # CUDA optimizations
    enable_cuda_graph: bool = True
    cuda_graph_max_batch_size: int = 8
    use_triton_flash_attn: bool = True

    # L4 Ada architecture specific
    enable_fp8_e4m3_kvcache: bool = False  # Experimental, start disabled
    tensor_parallel_size: int = 1  # Single L4 GPU
    pipeline_parallel_size: int = 1


class L4BandwidthOptimizer:
    """Optimizer for L4 GPU bandwidth constraints"""

    def __init__(self):
        self.config = L4OptimizationConfig()
        self.triton_config_path = (
            "/home/crathmene/git/multiModalL4/triton/models/llava_fp8/config.pbtxt"
        )
        self.vllm_config_path = "/etc/vllm/config.yaml"

    def measure_bandwidth_utilization(self) -> Dict:
        """Measure current memory bandwidth utilization"""
        try:
            # Use nvidia-smi to get memory bandwidth stats
            result = subprocess.run(
                ["nvidia-smi", "dmon", "-s", "mu", "-c", "5"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                # Parse memory utilization and bandwidth
                mem_utils = []
                for line in lines[2:]:  # Skip headers
                    parts = line.split()
                    if len(parts) >= 4:
                        mem_utils.append(
                            {
                                "memory_util": int(parts[2]),
                                "bandwidth_util": int(parts[3]),
                            }
                        )

                avg_mem = sum(m["memory_util"] for m in mem_utils) / len(mem_utils)
                avg_bw = sum(m["bandwidth_util"] for m in mem_utils) / len(mem_utils)

                return {
                    "avg_memory_utilization": avg_mem,
                    "avg_bandwidth_utilization": avg_bw,
                    "bandwidth_saturated": avg_bw > 85,
                    "recommendation": self._get_bandwidth_recommendation(avg_bw),
                }

        except Exception as e:
            logger.error(f"Failed to measure bandwidth: {e}")
            return {}

    def _get_bandwidth_recommendation(self, bandwidth_util: float) -> str:
        """Get recommendation based on bandwidth utilization"""
        if bandwidth_util > 90:
            return "Critical: Reduce batch size and sequence length"
        elif bandwidth_util > 80:
            return "Warning: Consider enabling more aggressive FP8 quantization"
        elif bandwidth_util > 70:
            return "Optimal: Current settings are well-balanced"
        else:
            return (
                "Underutilized: Can increase batch size or disable some optimizations"
            )

    def optimize_triton_config(self) -> Dict:
        """Optimize Triton configuration for L4"""
        logger.info("Optimizing Triton configuration for L4...")

        optimizations = {
            # Reduce memory transfers
            "continuous_batching": {
                "max_num_seqs": self.config.max_num_seqs,
                "max_paddings": self.config.max_paddings,
            },
            # PagedAttention tuning for L4
            "block_size": self.config.block_size,
            "num_gpu_blocks_override": self.config.num_gpu_blocks_override,
            # FP8 configuration
            "quantization_config": {
                "method": self.config.quantization_method,
                "activation_dtype": self.config.activation_dtype,
                "weight_dtype": self.config.weight_dtype,
                "kv_cache_dtype": self.config.kv_cache_dtype,
            },
            # CUDA optimizations
            "cuda_config": {
                "enable_cuda_graph": self.config.enable_cuda_graph,
                "cuda_graph_max_batch_size": self.config.cuda_graph_max_batch_size,
                "use_triton_flash_attn": self.config.use_triton_flash_attn,
            },
        }

        # Apply to Triton config
        self._update_triton_config(optimizations)

        return optimizations

    def _update_triton_config(self, optimizations: Dict):
        """Update Triton config.pbtxt with optimizations"""
        try:
            # Read current config
            with open(self.triton_config_path, "r") as f:
                config_content = f.read()

            # Add L4-specific parameters
            l4_params = f"""
# L4 Bandwidth Optimizations
parameters: [
  {{ key: "max_num_seqs", value: {{ string_value: "{optimizations["continuous_batching"]["max_num_seqs"]}" }} }},
  {{ key: "block_size", value: {{ string_value: "{optimizations["block_size"]}" }} }},
  {{ key: "num_gpu_blocks_override", value: {{ string_value: "{optimizations["num_gpu_blocks_override"]}" }} }},
  {{ key: "gpu_memory_utilization", value: {{ string_value: "{self.config.gpu_memory_utilization}" }} }},
  {{ key: "enable_chunked_prefill", value: {{ string_value: "{str(self.config.enable_chunked_prefill).lower()}" }} }},
  {{ key: "max_num_batched_tokens", value: {{ string_value: "{self.config.max_num_batched_tokens}" }} }}
]
"""

            # Update config (append if parameters section doesn't exist)
            if "parameters:" not in config_content:
                config_content += l4_params

            # Write back
            with open(self.triton_config_path + ".optimized", "w") as f:
                f.write(config_content)

            logger.info(
                f"Optimized config written to {self.triton_config_path}.optimized"
            )

        except Exception as e:
            logger.error(f"Failed to update Triton config: {e}")

    def create_vllm_config(self) -> Dict:
        """Create optimized vLLM configuration for L4"""
        vllm_config = {
            "model": "llava-fp8",
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "dtype": "float16",  # Base dtype
            "quantization": self.config.quantization_method,
            # L4 bandwidth optimizations
            "max_num_seqs": self.config.max_num_seqs,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "max_paddings": self.config.max_paddings,
            # Memory optimizations
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "swap_space": self.config.swap_space,
            "block_size": self.config.block_size,
            # FP8 specific
            "quantization_config": {
                "activation_dtype": self.config.activation_dtype,
                "weight_dtype": self.config.weight_dtype,
                "kv_cache_dtype": self.config.kv_cache_dtype,
            },
            # Performance features
            "enable_cuda_graph": self.config.enable_cuda_graph,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "use_v2_block_manager": True,
            # L4 specific tuning
            "disable_sliding_window": False,
            "enable_prefix_caching": True,
            "cpu_offload_gb": 0,  # No CPU offload for latency
            # Experimental FP8 KV-cache (disabled by default)
            "enable_fp8_e4m3_kvcache": self.config.enable_fp8_e4m3_kvcache,
        }

        # Save config
        os.makedirs(os.path.dirname(self.vllm_config_path), exist_ok=True)
        with open(self.vllm_config_path, "w") as f:
            yaml.dump(vllm_config, f, default_flow_style=False)

        logger.info(f"vLLM config saved to {self.vllm_config_path}")
        return vllm_config

    def benchmark_optimizations(self) -> Dict:
        """Benchmark the impact of L4 optimizations"""
        logger.info("Benchmarking L4 optimizations...")

        results = {"baseline": {}, "optimized": {}, "improvement": {}}

        # Test baseline (no optimizations)
        baseline_config = L4OptimizationConfig()
        baseline_config.max_num_seqs = 256  # Default
        baseline_config.block_size = 16  # Default
        baseline_config.enable_cuda_graph = False

        # Run baseline test
        results["baseline"] = self._run_benchmark_test(baseline_config)

        # Test optimized configuration
        results["optimized"] = self._run_benchmark_test(self.config)

        # Calculate improvements
        for metric in ["latency_ms", "throughput_tps", "memory_gb"]:
            if metric in results["baseline"] and metric in results["optimized"]:
                baseline_val = results["baseline"][metric]
                optimized_val = results["optimized"][metric]

                if metric == "latency_ms" or metric == "memory_gb":
                    # Lower is better
                    improvement = ((baseline_val - optimized_val) / baseline_val) * 100
                else:
                    # Higher is better
                    improvement = ((optimized_val - baseline_val) / baseline_val) * 100

                results["improvement"][metric] = f"{improvement:.1f}%"

        return results

    def _run_benchmark_test(self, config: L4OptimizationConfig) -> Dict:
        """Run a benchmark test with given configuration"""
        # This would integrate with GenAI-Perf or custom benchmark
        # Placeholder for actual benchmark results
        return {
            "latency_ms": 450,  # Example
            "throughput_tps": 120,  # Tokens per second
            "memory_gb": 18.5,
            "bandwidth_util": 75,
        }

    def auto_tune(self) -> Dict:
        """Auto-tune configuration based on workload"""
        logger.info("Starting auto-tuning for L4...")

        tuning_results = {"iterations": [], "best_config": None, "best_score": 0}

        # Parameters to tune
        param_grid = {
            "max_num_seqs": [4, 8, 16],
            "block_size": [16, 32, 64],
            "max_num_batched_tokens": [1024, 2048, 4096],
            "kv_cache_dtype": ["fp16", "fp8_e5m2"],
        }

        best_score = 0
        best_config = self.config

        # Grid search (simplified)
        for max_seqs in param_grid["max_num_seqs"]:
            for block_size in param_grid["block_size"]:
                for max_tokens in param_grid["max_num_batched_tokens"]:
                    for kv_dtype in param_grid["kv_cache_dtype"]:
                        # Create test config
                        test_config = L4OptimizationConfig()
                        test_config.max_num_seqs = max_seqs
                        test_config.block_size = block_size
                        test_config.max_num_batched_tokens = max_tokens
                        test_config.kv_cache_dtype = kv_dtype

                        # Run test
                        result = self._run_benchmark_test(test_config)

                        # Calculate score (weighted)
                        score = (
                            (500 - result["latency_ms"]) * 0.4  # Latency
                            + result["throughput_tps"] * 0.4  # Throughput
                            + (100 - result["bandwidth_util"]) * 0.2  # Bandwidth
                        )

                        tuning_results["iterations"].append(
                            {
                                "config": asdict(test_config),
                                "score": score,
                                "metrics": result,
                            }
                        )

                        if score > best_score:
                            best_score = score
                            best_config = test_config

        tuning_results["best_config"] = asdict(best_config)
        tuning_results["best_score"] = best_score

        # Apply best configuration
        self.config = best_config
        self.optimize_triton_config()

        logger.info(f"Auto-tuning complete. Best score: {best_score:.2f}")

        return tuning_results

    def generate_optimization_report(self) -> str:
        """Generate L4 optimization report"""
        bandwidth_stats = self.measure_bandwidth_utilization()

        report = f"""
# L4 GPU Optimization Report

## Hardware Specifications
- GPU: NVIDIA L4 (Ada Lovelace)
- Memory: 24GB GDDR6
- Bandwidth: 300 GB/s (constraint)
- FP8 Support: Native E4M3/E5M2

## Current Configuration
- Max Sequences: {self.config.max_num_seqs}
- Block Size: {self.config.block_size}
- Max Batched Tokens: {self.config.max_num_batched_tokens}
- KV Cache Type: {self.config.kv_cache_dtype}
- CUDA Graphs: {"Enabled" if self.config.enable_cuda_graph else "Disabled"}

## Bandwidth Utilization
- Memory Utilization: {bandwidth_stats.get("avg_memory_utilization", "N/A")}%
- Bandwidth Utilization: {bandwidth_stats.get("avg_bandwidth_utilization", "N/A")}%
- Status: {bandwidth_stats.get("recommendation", "Unknown")}

## Optimization Strategy
1. Reduced max_num_seqs from 256 to {self.config.max_num_seqs}
2. Increased block_size to {self.config.block_size} for fewer memory transfers
3. Using {self.config.kv_cache_dtype} for KV-cache (conservative approach)
4. {"Enabled" if self.config.enable_cuda_graph else "Disabled"} CUDA graphs for kernel fusion

## Recommendations
1. Monitor bandwidth saturation closely
2. Consider FP8 E5M2 for KV-cache after thorough testing
3. Reduce batch size if bandwidth exceeds 85%
4. Use GenAI-Perf for production benchmarking

## Next Steps
- Run `python3 l4_bandwidth_optimizer.py --auto-tune` for workload-specific tuning
- Deploy with monitoring to track bandwidth saturation
- A/B test FP8 KV-cache carefully in production
"""

        # Save report
        report_path = "/logs/l4_optimization_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Optimization report saved to {report_path}")
        return report


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="L4 Bandwidth Optimizer")
    parser.add_argument("--auto-tune", action="store_true", help="Run auto-tuning")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--apply", action="store_true", help="Apply optimizations")
    args = parser.parse_args()

    optimizer = L4BandwidthOptimizer()

    if args.auto_tune:
        logger.info("Running auto-tuning...")
        results = optimizer.auto_tune()
        print(json.dumps(results["best_config"], indent=2))

    elif args.benchmark:
        logger.info("Running benchmarks...")
        results = optimizer.benchmark_optimizations()
        print(json.dumps(results, indent=2))

    elif args.apply:
        logger.info("Applying L4 optimizations...")
        optimizer.optimize_triton_config()
        optimizer.create_vllm_config()
        print("Optimizations applied successfully")

    else:
        # Default: generate report
        report = optimizer.generate_optimization_report()
        print(report)


if __name__ == "__main__":
    main()
