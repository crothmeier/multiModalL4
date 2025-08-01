#!/usr/bin/env python3
"""Demonstrate RequestQueue functionality"""

import asyncio
import time

import httpx


async def main():
    """Test request queue with concurrent requests"""
    async with httpx.AsyncClient() as client:
        # Check current status
        status = await client.get("http://localhost:8888/status")
        print(f"Initial status: {status.json()}")

        if status.json()["current_model"] == "mistral-llm":
            # Switch to coder
            target_model = "coder"
        else:
            # Switch to mistral
            target_model = "mistral"

        print(f"\n--- Testing concurrent requests for {target_model} ---")

        async def request_model(request_id):
            start = time.time()
            try:
                print(f"Request {request_id}: Starting...")
                resp = await client.post(
                    "http://localhost:8888/ensure_model",
                    json={"model": target_model},
                    timeout=60.0,
                )
                elapsed = time.time() - start
                result = resp.json()
                print(
                    f"Request {request_id}: Completed in {elapsed:.1f}s - "
                    f"swapped={result['swapped']}"
                )
                return result
            except Exception as e:
                elapsed = time.time() - start
                print(f"Request {request_id}: Failed after {elapsed:.1f}s - {e}")
                return None

        # Send 3 concurrent requests
        print("\nSending 3 concurrent requests...")
        tasks = []
        for i in range(3):
            task = asyncio.create_task(request_model(i))
            tasks.append(task)
            await asyncio.sleep(0.1)  # Small delay between requests

        results = await asyncio.gather(*tasks)

        # Count how many actually swapped
        swap_count = sum(1 for r in results if r and r["swapped"])
        print(f"\nResults: {swap_count} swap(s) performed out of 3 requests")
        print(
            "This demonstrates that only one swap occurs while "
            "other requests wait and return quickly"
        )

        # Final status
        status = await client.get("http://localhost:8888/status")
        print(f"\nFinal status: {status.json()}")


if __name__ == "__main__":
    asyncio.run(main())
