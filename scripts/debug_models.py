#!/usr/bin/env python3
"""Debug script to test each model individually with vLLM."""

import json
import os
import subprocess
from pathlib import Path

# Models to test
MODELS = {
    "mistral-awq": {
        "path": "/mnt/models/mistral-awq",
        "tokenizer": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.50,
        "quantization": "awq",
        "expected_issues": None,
    },
    "deepseek-gptq": {
        "path": "/mnt/models/deepseek-gptq",
        "tokenizer": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "max_model_len": 4096,  # Reduced for testing
        "gpu_memory_utilization": 0.35,
        "quantization": "gptq",
        "trust_remote_code": True,
        "expected_issues": "May need quantization config adjustments",
    },
    "deepseek-vl": {
        "path": "/mnt/models/deepseek-vl-1.3b-chat",
        "tokenizer": None,  # Use model's tokenizer
        "max_model_len": 2048,  # Conservative for multimodal
        "gpu_memory_utilization": 0.25,
        "trust_remote_code": True,
        "expected_issues": "Multimodal support may require vLLM upgrade",
    },
}


def test_model_loading(model_name, config):
    """Test if a model can be loaded with vLLM."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")

    # Build vLLM command
    cmd = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config["path"],
        "--max-model-len",
        str(config["max_model_len"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--host",
        "0.0.0.0",  # nosec B104 - local testing only
        "--port",
        "8001",  # Different port for testing
        "--disable-log-requests",
    ]

    # Add optional parameters
    if config.get("tokenizer"):
        cmd.extend(["--tokenizer", config["tokenizer"]])

    if config.get("quantization"):
        cmd.extend(["--quantization", config["quantization"]])

    if config.get("trust_remote_code"):
        cmd.append("--trust-remote-code")

    # Set environment
    env = os.environ.copy()
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    env[key] = value

    print(f"Command: {' '.join(cmd)}")
    if config.get("expected_issues"):
        print(f"⚠️  Expected issues: {config['expected_issues']}")

    # Run the command with timeout
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )

        # Wait for startup (10 seconds)
        import time

        print("\nWaiting for model to load...")
        for i in range(10):
            time.sleep(1)
            print(".", end="", flush=True)

            # Check if process crashed
            if process.poll() is not None:
                break

        print()

        if process.poll() is None:
            # Process is still running, try to get model info
            try:
                import requests

                response = requests.get("http://localhost:8001/v1/models", timeout=2)
                if response.status_code == 200:
                    models = response.json()
                    print("✅ SUCCESS: Model loaded successfully!")
                    print(f"Model info: {json.dumps(models, indent=2)}")
                else:
                    print(f"❌ API returned status {response.status_code}")
            except Exception as e:
                print(f"❌ Could not connect to API: {e}")

            # Terminate the process
            process.terminate()
            process.wait(timeout=5)
        else:
            # Process crashed, get output
            stdout, stderr = process.communicate()
            print(f"❌ FAILED: Process exited with code {process.returncode}")

            # Print relevant error lines
            print("\nError output:")
            error_lines = stderr.split("\n")
            for line in error_lines:
                if any(
                    keyword in line.lower()
                    for keyword in ["error", "exception", "failed", "traceback"]
                ):
                    print(f"  {line}")

            # Print last few lines if no specific errors found
            if not any("error" in line.lower() for line in error_lines):
                print("\nLast 10 lines of output:")
                for line in error_lines[-10:]:
                    if line.strip():
                        print(f"  {line}")

    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT: Process took too long")
        process.kill()
    except Exception as e:
        print(f"❌ ERROR: {e}")


def check_model_files(model_name, config):
    """Check if model files exist and are valid."""
    print(f"\nChecking files for {model_name}:")
    model_path = Path(config["path"])

    if not model_path.exists():
        print(f"  ❌ Model directory does not exist: {model_path}")
        return False

    # Check for essential files
    essential_files = ["config.json"]
    optional_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
        "*.bin",
    ]

    for file in essential_files:
        file_path = model_path / file
        if file_path.exists():
            print(f"  ✅ {file} exists")
        else:
            print(f"  ❌ {file} missing")

    for pattern in optional_files:
        if "*" in pattern:
            files = list(model_path.glob(pattern))
            if files:
                print(f"  ✅ Found {len(files)} {pattern} files")
        else:
            file_path = model_path / pattern
            if file_path.exists():
                print(f"  ✅ {pattern} exists")

    # Check config.json content
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                model_config = json.load(f)
                print(f"  Model type: {model_config.get('model_type', 'unknown')}")
                arch = (
                    model_config.get("architectures", ["unknown"])[0]
                    if "architectures" in model_config
                    else "unknown"
                )
                print(f"  Architecture: {arch}")

                # Check for quantization config
                if "quantization_config" in model_config:
                    print(f"  Quantization: {model_config['quantization_config']}")
        except Exception as e:
            print(f"  ⚠️  Could not parse config.json: {e}")

    return True


def suggest_alternatives():
    """Suggest alternative models if DeepSeek models don't work."""
    print("\n" + "=" * 60)
    print("Alternative Model Suggestions")
    print("=" * 60)

    alternatives = {
        "Code Models": [
            "TheBloke/CodeLlama-7B-Instruct-GPTQ (4-bit, ~4GB)",
            "TheBloke/WizardCoder-Python-7B-V1.0-GPTQ (4-bit, ~4GB)",
            "TheBloke/starcoder-GPTQ (4-bit, various sizes)",
        ],
        "Multimodal Models": [
            "liuhaotian/llava-v1.5-7b (requires vLLM 0.5.0+)",
            "THUDM/cogvlm-chat-hf (17B, might be too large)",
            "Salesforce/blip2-opt-2.7b (smaller, good for demos)",
        ],
        "General Purpose": [
            "TheBloke/Llama-2-7B-Chat-GPTQ (4-bit, ~4GB)",
            "TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ (4-bit, ~4GB)",
        ],
    }

    print("\nIf DeepSeek models continue to fail, consider these alternatives:")
    for category, models in alternatives.items():
        print(f"\n{category}:")
        for model in models:
            print(f"  - {model}")

    print("\nNote: Most TheBloke models have good vLLM compatibility.")


def main():
    """Main debug routine."""
    print("vLLM Model Debug Script")
    print("=" * 60)

    # Check vLLM version
    try:
        result = subprocess.run(
            [
                "python3",
                "-c",
                "import vllm; print(f'vLLM version: {vllm.__version__}')",
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip())
    except Exception:
        print("Could not determine vLLM version")

    # Test each model
    for model_name, config in MODELS.items():
        if check_model_files(model_name, config):
            test_model_loading(model_name, config)
        else:
            print(f"\nSkipping {model_name} due to missing files")

    # Suggest alternatives
    suggest_alternatives()


if __name__ == "__main__":
    main()
