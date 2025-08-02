#!/usr/bin/env python3
"""Test script for RequestQueue functionality during model swaps"""

import asyncio
import time

import httpx


async def test_request(client, model_name, request_id):
    """Send a request to ensure a specific model"""
    start_time = time.time()
    try:
        response = await client.post(
            "http://localhost:8888/ensure_model",
            json={"model": model_name},
            timeout=60.0,
        )
        elapsed = time.time() - start_time
        result = response.json()
        print(
            f"Request {request_id} for {model_name}: "
            f"completed in {elapsed:.1f}s - {result}"
        )
        return result
    except httpx.ReadTimeout:
        elapsed = time.time() - start_time
        print(f"Request {request_id} for {model_name}: TIMEOUT after {elapsed:.1f}s")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        print(
            f"Request {request_id} for {model_name}: ERROR after {elapsed:.1f}s - {e}"
        )
        return None


async def main():
    """Test concurrent requests during model swaps"""
    print("Testing RequestQueue functionality...")

    async with httpx.AsyncClient() as client:
        # Check current status
        status = await client.get("http://localhost:8888/status")
        print(f"Initial status: {status.json()}")

        # Test 1: Multiple requests for the same model during swap
        print("\n--- Test 1: Multiple requests for same model during swap ---")
        target_model = "mistral"  # Pick a model that takes time to load

        # Send multiple concurrent requests for the same model
        tasks = []
        for i in range(3):
            task = asyncio.create_task(test_request(client, target_model, f"1-{i}"))
            tasks.append(task)
            await asyncio.sleep(0.1)  # Small delay between requests

        await asyncio.gather(*tasks)

        # Test 2: Requests for different models
        print("\n--- Test 2: Mixed model requests ---")
        await asyncio.sleep(2)  # Let the system stabilize

        tasks = [
            asyncio.create_task(test_request(client, "mistral", "2-0")),
            asyncio.create_task(test_request(client, "mistral", "2-1")),
            asyncio.create_task(test_request(client, "coder", "2-2")),
        ]

        await asyncio.gather(*tasks)

        # Test 3: Verify timeout behavior
        print("\n--- Test 3: Testing timeout behavior ---")
        # This would require mocking a slow swap, skipping for now

        # Final status
        status = await client.get("http://localhost:8888/status")
        print(f"\nFinal status: {status.json()}")


if __name__ == "__main__":
    asyncio.run(main())
