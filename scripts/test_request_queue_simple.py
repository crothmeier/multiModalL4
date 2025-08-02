#!/usr/bin/env python3
"""Simple test for RequestQueue functionality"""

import asyncio
import time

import httpx


async def test_simple():
    """Test basic request queue functionality"""
    async with httpx.AsyncClient() as client:
        # Check initial status
        status = await client.get("http://localhost:8888/status")
        print(f"Initial status: {status.json()}")

        # First ensure mistral is loaded
        print("\n--- Ensuring mistral is loaded ---")
        response = await client.post(
            "http://localhost:8888/ensure_model",
            json={"model": "mistral"},
            timeout=60.0,
        )
        print(f"Response: {response.json()}")

        # Now test concurrent requests during a swap
        print("\n--- Testing concurrent requests during swap to coder ---")

        async def request_coder(request_id):
            start = time.time()
            try:
                resp = await client.post(
                    "http://localhost:8888/ensure_model",
                    json={"model": "coder"},
                    timeout=60.0,
                )
                elapsed = time.time() - start
                print(
                    f"Request {request_id}: completed in {elapsed:.1f}s - {resp.json()}"
                )
            except Exception as e:
                elapsed = time.time() - start
                print(f"Request {request_id}: failed after {elapsed:.1f}s - {e}")

        # Send 3 concurrent requests
        tasks = [request_coder(i) for i in range(3)]
        await asyncio.gather(*tasks)

        # Final status
        status = await client.get("http://localhost:8888/status")
        print(f"\nFinal status: {status.json()}")


if __name__ == "__main__":
    asyncio.run(test_simple())
