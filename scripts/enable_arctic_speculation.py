#!/usr/bin/env python3
"""
Arctic Speculative Decoding Configuration for vLLM 0.8.4+
Replaces EAGLE with Arctic for better acceptance rates
"""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ArcticConfig:
    """Arctic speculative decoding configuration for vLLM 0.8.4+"""

    draft_model: str = "llava-v1.6-draft-fp8"  # Updated from 1.5b
    speculator: str = "arctic"  # Replaces EAGLE
    proposal_length: int = 4
    acceptance_rate_target: float = 0.75
    disable_on_low_acceptance: bool = True
    acceptance_window: int = 100
    max_speculation_tokens: int = 8
    min_speculation_tokens: int = 2
    adaptive_speculation: bool = True

    # L4-specific optimizations
    use_cuda_graph: bool = True
    cuda_graph_max_batch_size: int = 8

    # Memory optimization for L4
    share_kv_cache_with_draft: bool = True
    draft_model_tp: int = 1  # Tensor parallel for draft model


class ArcticSpeculationManager:
    """Manages Arctic speculative decoding for LLaVA FP8"""

    def __init__(self, vllm_endpoint: str = "http://localhost:8000"):
        self.vllm_endpoint = vllm_endpoint
        self.config = ArcticConfig()
        self.low_efficiency_start = None  # Track when low efficiency started
        self.speculation_enabled = True

    def check_vllm_version(self) -> bool:
        """Verify vLLM version supports Arctic"""
        try:
            response = requests.get(f"{self.vllm_endpoint}/version")
            if response.status_code == 200:
                version_info = response.json()
                version = version_info.get("version", "")

                # Check for vLLM 0.8.4+
                major, minor, patch = version.split(".")[:3]
                if int(major) == 0 and int(minor) == 8 and int(patch) >= 4:
                    logger.info(f"vLLM version {version} supports Arctic speculation")
                    return True
                else:
                    logger.warning(f"vLLM version {version} may not support Arctic")
                    return False
        except Exception as e:
            logger.error(f"Failed to check vLLM version: {e}")
            return False

    def download_draft_model(self) -> bool:
        """Download and prepare draft model for speculation"""
        draft_model_path = f"/mnt/models/{self.config.draft_model}"

        if os.path.exists(draft_model_path):
            logger.info(f"Draft model already exists at {draft_model_path}")
            return True

        logger.info(f"Downloading draft model: {self.config.draft_model}")

        # Download command for draft model
        download_cmd = f"""
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='liuhaotian/{self.config.draft_model}',
    local_dir='{draft_model_path}',
    local_dir_use_symlinks=False
)
"
        """

        try:
            os.system(download_cmd)
            logger.info("Draft model downloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to download draft model: {e}")
            return False

    def quantize_draft_model(self) -> bool:
        """Quantize draft model to FP8 for faster speculation"""
        logger.info("Quantizing draft model to FP8...")

        quantization_config = {
            "method": "fp8",
            "activation_dtype": "fp8_e4m3",
            "weight_dtype": "fp8_e4m3",
            "calibration_samples": 256,  # Fewer samples for draft model
            "calibration_dataset": "coco",
        }

        # Send quantization request
        try:
            response = requests.post(
                f"{self.vllm_endpoint}/v1/models/quantize",
                json={
                    "model": self.config.draft_model,
                    "quantization_config": quantization_config,
                },
            )

            if response.status_code == 200:
                logger.info("Draft model quantized successfully")
                return True
            else:
                logger.error(f"Quantization failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to quantize draft model: {e}")
            return False

    def enable_speculation(self) -> bool:
        """Enable Arctic speculation on main model"""
        logger.info("Enabling Arctic speculative decoding...")

        # Prepare configuration
        speculation_config = {
            "model": "llava_fp8",
            "speculative_config": asdict(self.config),
            "engine_config": {
                "max_num_seqs": 8,  # L4 optimization
                "max_paddings": 32,
                "block_size": 32,
                "num_gpu_blocks_override": 2048,
                "enable_prefix_caching": True,
                "disable_log_stats": False,
            },
        }

        # Send configuration to vLLM
        try:
            response = requests.post(
                f"{self.vllm_endpoint}/v1/models/update",
                json=speculation_config,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Arctic speculation enabled: {result}")
                return True
            else:
                logger.error(f"Failed to enable speculation: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to enable speculation: {e}")
            return False

    def monitor_acceptance_rate(self) -> Optional[float]:
        """Monitor speculation acceptance rate using correct vLLM 0.9 metric"""
        try:
            response = requests.get(f"{self.vllm_endpoint}/metrics")
            if response.status_code == 200:
                metrics_text = response.text

                # Parse Prometheus metrics for acceptance rate (vLLM 0.9 metric)
                for line in metrics_text.split("\n"):
                    if "vllm_spec_decode_efficiency" in line and not line.startswith(
                        "#"
                    ):
                        rate = float(line.split()[-1])
                        logger.info(f"Current speculation efficiency: {rate:.2%}")
                        return rate

        except Exception as e:
            logger.error(f"Failed to get acceptance rate: {e}")

        return None

    def auto_tune_speculation(self) -> Dict:
        """Auto-tune speculation parameters based on acceptance rate"""
        logger.info("Auto-tuning speculation parameters...")

        current_rate = self.monitor_acceptance_rate()
        if current_rate is None:
            return {"status": "error", "message": "Could not get acceptance rate"}

        tuning_result = {
            "current_rate": current_rate,
            "target_rate": self.config.acceptance_rate_target,
            "adjustments": [],
        }

        # Adjust proposal length based on acceptance rate
        if current_rate < 0.5:
            # Poor acceptance, reduce speculation
            self.config.proposal_length = max(2, self.config.proposal_length - 1)
            tuning_result["adjustments"].append(
                f"Reduced proposal_length to {self.config.proposal_length}"
            )

        elif current_rate > 0.85:
            # High acceptance, can be more aggressive
            self.config.proposal_length = min(8, self.config.proposal_length + 1)
            tuning_result["adjustments"].append(
                f"Increased proposal_length to {self.config.proposal_length}"
            )

        # Disable speculation if consistently poor
        if current_rate < 0.3:
            self.config.disable_on_low_acceptance = True
            tuning_result["adjustments"].append(
                "Disabling speculation due to low acceptance"
            )

            # Disable speculation
            response = requests.post(
                f"{self.vllm_endpoint}/v1/models/update",
                json={"model": "llava_fp8", "speculative_config": {"enabled": False}},
            )

        # Apply new configuration
        if tuning_result["adjustments"]:
            self.enable_speculation()

        return tuning_result

    def benchmark_with_speculation(self) -> Dict:
        """Benchmark performance with speculation enabled"""
        logger.info("Running benchmark with Arctic speculation...")

        test_prompts = [
            "Describe this image in detail.",
            "What objects can you see in this image?",
            "Explain what's happening in this scene.",
            "What is the main subject of this photograph?",
            "Describe the colors and composition.",
        ]

        results = {"with_speculation": {}, "without_speculation": {}}

        # Test with speculation
        self.enable_speculation()
        results["with_speculation"] = self._run_inference_test(test_prompts)

        # Test without speculation
        requests.post(
            f"{self.vllm_endpoint}/v1/models/update",
            json={"model": "llava_fp8", "speculative_config": {"enabled": False}},
        )
        results["without_speculation"] = self._run_inference_test(test_prompts)

        # Calculate improvements
        with_spec = results["with_speculation"]["avg_latency_ms"]
        without_spec = results["without_speculation"]["avg_latency_ms"]
        improvement = ((without_spec - with_spec) / without_spec) * 100

        results["improvement_percent"] = improvement
        results["recommendation"] = "keep_enabled" if improvement > 10 else "disable"

        logger.info(f"Speculation improvement: {improvement:.1f}%")

        return results

    def _run_inference_test(self, prompts: list) -> Dict:
        """Run inference test with given prompts"""
        import time

        latencies = []

        for prompt in prompts:
            start = time.time()

            response = requests.post(
                f"{self.vllm_endpoint}/v1/completions",
                json={
                    "model": "llava_fp8",
                    "prompt": prompt,
                    "max_tokens": 128,
                    "temperature": 0.7,
                },
            )

            latency = (time.time() - start) * 1000
            latencies.append(latency)

        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "num_samples": len(latencies),
        }


def main():
    """Main entry point"""
    manager = ArcticSpeculationManager()

    # Check vLLM version
    if not manager.check_vllm_version():
        logger.error("vLLM version does not support Arctic speculation")
        sys.exit(1)

    # Download and prepare draft model
    if not manager.download_draft_model():
        logger.error("Failed to prepare draft model")
        sys.exit(1)

    # Quantize draft model
    if not manager.quantize_draft_model():
        logger.warning("Draft model quantization failed, using FP16")

    # Enable speculation
    if not manager.enable_speculation():
        logger.error("Failed to enable Arctic speculation")
        sys.exit(1)

    # Run benchmark
    benchmark_results = manager.benchmark_with_speculation()

    # Save results
    with open("/logs/arctic_speculation_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info("Arctic speculation configuration complete")

    # Start monitoring loop with auto-toggle on low efficiency
    import asyncio
    import time

    async def monitor_loop():
        """Monitor loop with auto-toggle when efficiency < 0.55 for > 5 minutes"""
        while True:
            await asyncio.sleep(60)

            # Get current efficiency
            current_efficiency = manager.monitor_acceptance_rate()

            if current_efficiency is not None:
                # Check if efficiency is below threshold (0.55)
                if current_efficiency < 0.55:
                    if manager.low_efficiency_start is None:
                        # Start tracking low efficiency period
                        manager.low_efficiency_start = time.time()
                        logger.warning(
                            f"Speculation efficiency below 0.55 ({current_efficiency:.2%}), starting timer"
                        )
                    else:
                        # Check if it's been > 5 minutes
                        duration = time.time() - manager.low_efficiency_start
                        if duration > 300 and manager.speculation_enabled:  # 5 minutes
                            logger.error(
                                f"Speculation efficiency below 0.55 for {duration / 60:.1f} minutes, disabling speculation"
                            )

                            # Disable speculation
                            response = requests.post(
                                f"{manager.vllm_endpoint}/v1/models/update",
                                json={
                                    "model": "llava_fp8",
                                    "speculative_config": {"enabled": False},
                                },
                            )
                            manager.speculation_enabled = False
                            logger.info(
                                "Arctic speculation has been automatically disabled due to low efficiency"
                            )
                else:
                    # Efficiency is good, reset timer
                    if manager.low_efficiency_start is not None:
                        logger.info(
                            f"Speculation efficiency recovered to {current_efficiency:.2%}"
                        )
                        manager.low_efficiency_start = None

            # Regular auto-tuning if speculation is still enabled
            if manager.speculation_enabled:
                tuning_result = manager.auto_tune_speculation()
                logger.info(f"Auto-tuning result: {tuning_result}")

    if os.getenv("ENABLE_AUTO_TUNING", "true").lower() == "true":
        asyncio.run(monitor_loop())


if __name__ == "__main__":
    main()
