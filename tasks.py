"""Invoke tasks for running integration tests with isolated Docker environment."""

import os
import sys
import time
from pathlib import Path

from invoke import Context, task

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_colored(message, color=GREEN):
    """Print message in color."""
    print(f"{color}{message}{RESET}")


def wait_for_health(ctx, service, port, path="/healthz", timeout=60):
    """Wait for a service to become healthy."""
    import requests

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"http://localhost:{port}{path}")
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


@task
def test_setup(ctx):
    """Set up test environment with isolated Docker network."""
    print_colored("=== Setting up test environment ===", BLUE)

    # Create test network
    print("→ Creating isolated test network...")
    ctx.run("docker network create multimodal-test-net || true", hide=True)

    # Create test .env file
    print("→ Creating test environment file...")
    test_env = """
HF_TOKEN=test-token
JWT_SECRET=test-secret-key-for-integration-tests
JWT_ALGORITHM=HS256
JWT_AUDIENCE=llm-platform
JWT_ISSUER=multimodal-stack
"""
    with open(".env.test", "w") as f:
        f.write(test_env.strip())

    # Start services with test config
    print("→ Starting services...")
    ctx.run("docker-compose --env-file .env.test -p multimodal-test up -d")

    # Wait for services to be ready
    print("→ Waiting for services to be healthy...")
    services = [
        ("API Gateway", 8080, "/healthz"),
        ("Model Orchestrator", 8888, "/health"),
    ]

    all_healthy = True
    for name, port, path in services:
        print(f"   Checking {name}...", end="")
        if wait_for_health(ctx, name, port, path):
            print_colored(" ✓", GREEN)
        else:
            print_colored(" ✗", RED)
            all_healthy = False

    if not all_healthy:
        print_colored("\n❌ Some services failed to start!", RED)
        return False

    print_colored("\n✅ Test environment ready!", GREEN)
    return True


@task
def test_teardown(ctx):
    """Tear down test environment."""
    print_colored("\n=== Tearing down test environment ===", BLUE)

    # Stop and remove containers
    print("→ Stopping test containers...")
    ctx.run("docker-compose -p multimodal-test down -v", hide=True)

    # Remove test network
    print("→ Removing test network...")
    ctx.run("docker network rm multimodal-test-net || true", hide=True)

    # Clean up test files
    print("→ Cleaning up test files...")
    if os.path.exists(".env.test"):
        os.remove(".env.test")

    print_colored("✅ Test environment cleaned up!", GREEN)


@task
def test_integration(ctx):
    """Run all integration tests with proper setup/teardown."""
    print_colored("=== Running Integration Tests ===", BLUE)

    # Setup
    if not test_setup(ctx):
        print_colored("Setup failed! Aborting tests.", RED)
        test_teardown(ctx)
        sys.exit(1)

    try:
        # Set test environment variables
        test_env = {
            "JWT_SECRET": "test-secret-key-for-integration-tests",
            "JWT_ALGORITHM": "HS256",
            "JWT_AUDIENCE": "llm-platform",
            "JWT_ISSUER": "multimodal-stack",
        }

        # Run pytest with coverage
        print_colored("\n→ Running pytest integration tests...", YELLOW)
        env_str = " ".join(f"{k}={v}" for k, v in test_env.items())
        result = ctx.run(
            f"{env_str} pytest -m integration -v --tb=short --cov=services --cov-report=term-missing",
            warn=True,
        )

        # Print results summary
        if result.ok:
            print_colored("\n✅ All integration tests passed!", GREEN)
        else:
            print_colored("\n❌ Some tests failed!", RED)

        return result.ok

    finally:
        # Always teardown
        test_teardown(ctx)


@task
def test_security(ctx):
    """Run only security-related tests."""
    print_colored("=== Running Security Tests ===", BLUE)

    # Setup
    if not test_setup(ctx):
        print_colored("Setup failed! Aborting tests.", RED)
        test_teardown(ctx)
        sys.exit(1)

    try:
        # Set test environment variables
        test_env = {
            "JWT_SECRET": "test-secret-key-for-integration-tests",
            "JWT_ALGORITHM": "HS256",
            "JWT_AUDIENCE": "llm-platform",
            "JWT_ISSUER": "multimodal-stack",
        }

        # Run security tests
        print_colored("\n→ Running security tests...", YELLOW)
        env_str = " ".join(f"{k}={v}" for k, v in test_env.items())

        # Run JWT auth tests
        print("\n→ Testing JWT authentication...")
        auth_result = ctx.run(
            f"{env_str} pytest tests/integration/test_auth.py -v", warn=True
        )

        # Run path validation tests
        print("\n→ Testing path traversal protection...")
        path_result = ctx.run(
            f"{env_str} pytest tests/integration/test_path_validation.py -v", warn=True
        )

        # Summary
        all_passed = auth_result.ok and path_result.ok
        if all_passed:
            print_colored("\n✅ All security tests passed!", GREEN)
        else:
            print_colored("\n❌ Some security tests failed!", RED)

        return all_passed

    finally:
        # Always teardown
        test_teardown(ctx)


@task
def test_cleanup(ctx):
    """Clean up any leftover test resources."""
    print_colored("=== Cleaning up test resources ===", BLUE)

    # Stop any running test containers
    print("→ Stopping any running test containers...")
    ctx.run("docker-compose -p multimodal-test down -v || true", hide=True)

    # Remove test network
    print("→ Removing test networks...")
    ctx.run("docker network rm multimodal-test-net || true", hide=True)

    # Remove test files
    print("→ Removing test files...")
    test_files = [".env.test", ".coverage", "htmlcov"]
    for file in test_files:
        if Path(file).exists():
            if Path(file).is_dir():
                ctx.run(f"rm -rf {file}")
            else:
                os.remove(file)

    print_colored("✅ Cleanup complete!", GREEN)


@task(pre=[test_integration, test_security])
def test_all(ctx):
    """Run all integration tests with proper setup/teardown."""
    print_colored("\n=== All Tests Complete ===", BLUE)

    # Generate coverage report
    if Path(".coverage").exists():
        print("\n→ Generating HTML coverage report...")
        ctx.run("coverage html", hide=True)
        print_colored("Coverage report available at: htmlcov/index.html", YELLOW)


@task
def test_gpu_memory(ctx):
    """Test GPU memory utilization."""
    print_colored("=== Testing GPU Memory Usage ===", BLUE)

    # Check if nvidia-smi is available
    result = ctx.run("which nvidia-smi", warn=True, hide=True)
    if not result.ok:
        print_colored("⚠️  nvidia-smi not found - skipping GPU tests", YELLOW)
        return

    # Run GPU memory test
    ctx.run("pytest tests/integration/test_gpu_memory.py -v")


# Add default task
@task(default=True)
def help(ctx):
    """Show available tasks."""
    print_colored("=== Available Test Tasks ===", BLUE)
    print(
        """
Tasks:
  test-integration  Run all integration tests with isolated Docker environment
  test-security     Run only security-related tests (JWT auth, path traversal)
  test-cleanup      Clean up any leftover test resources
  test-all          Run all tests (integration + security) with coverage
  test-gpu-memory   Test GPU memory utilization (requires nvidia-smi)

Usage:
  invoke test-integration
  invoke test-security
  invoke test-all
  invoke test-cleanup
"""
    )
