#!/usr/bin/env python3
"""Simple demo showing RequestQueue behavior"""

import asyncio
import time

import httpx


async def main():
    """Show how requests queue during model swaps"""
    async with httpx.AsyncClient() as client:
        print("RequestQueue Demo - Showing queuing behavior during model swaps")
        print("=" * 60)

        # Check current status
        status = await client.get("http://localhost:8888/status")
        print(f"Initial status: {status.json()}")

        print("\n--- Simulating 3 concurrent requests for mistral ---")
        print("Expected behavior:")
        print("- Request 0 triggers the model swap")
        print("- Requests 1 and 2 queue and wait for the swap to complete")
        print("- All requests return after the swap completes\n")

        async def request_model(request_id):
            start = time.time()
            try:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Request {request_id}: "
                    "Sending request..."
                )
                resp = await client.post(
                    "http://localhost:8888/ensure_model",
                    json={"model": "mistral"},
                    timeout=60.0,
                )
                elapsed = time.time() - start
                result = resp.json()
                print(
                    f"[{time.strftime('%H:%M:%S')}] Request {request_id}: "
                    f"Completed in {elapsed:.1f}s - swapped={result['swapped']}"
                )
                return result
            except Exception as e:
                elapsed = time.time() - start
                print(
                    f"[{time.strftime('%H:%M:%S')}] Request {request_id}: "
                    f"Failed after {elapsed:.1f}s - {type(e).__name__}: {e}"
                )
                return None

        # Send 3 concurrent requests
        tasks = []
        for i in range(3):
            task = asyncio.create_task(request_model(i))
            tasks.append(task)
            await asyncio.sleep(0.5)  # Small delay to show queuing

        results = await asyncio.gather(*tasks)

        # Final status
        print("\n--- Final Results ---")
        status = await client.get("http://localhost:8888/status")
        print(f"Final status: {status.json()}")

        successful = sum(1 for r in results if r is not None)
        swapped = sum(1 for r in results if r and r.get("swapped", False))
        print(
            f"\nSummary: {successful}/3 requests succeeded, {swapped} triggered a swap"
        )

        if successful == 3 and swapped == 1:
            print(
                "✓ RequestQueue working correctly: "
                "Multiple requests queued, only one swap performed"
            )
        else:
            print("✗ Unexpected behavior - check orchestrator logs")


if __name__ == "__main__":
    asyncio.run(main())
