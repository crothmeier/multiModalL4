"""Integration test for GPU memory allocation."""

import subprocess
import time

import pytest


@pytest.mark.integration
class TestGPUMemory:
    """Test GPU memory utilization stays under 90%."""

    def test_gpu_memory_headroom(self):
        """Verify GPU memory usage leaves headroom."""
        # Start all services
        subprocess.run(["docker-compose", "up", "-d"], check=True)

        # Wait for services to stabilize
        time.sleep(30)

        # Check GPU memory usage
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in result.stdout.strip().split("\n"):
            used, total = map(int, line.split(", "))
            utilization = used / total

            assert utilization < 0.90, (
                f"GPU memory usage {utilization:.1%} exceeds 90% threshold"
            )
            print(f"✓ GPU memory usage: {utilization:.1%} ({used}/{total} MB)")
