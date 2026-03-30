#!/usr/bin/env python3
"""
Test: does max_model_len affect concurrency ceiling?

Sends increasing waves of concurrent requests to a running vLLM server
and checks /metrics for num_requests_waiting to find the point where
the scheduler starts queuing. Run against the same server with different
--max-model-len values to see if it changes the ceiling.

Usage:
  # Terminal 1: start vLLM with --max-model-len 2048
  # Terminal 2:
  python test_concurrency_cap.py --model gpt-oss-20b --max-concurrency 50

  # Then restart with --max-model-len 32768 and run again to compare.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid

import aiohttp


async def _check_waiting(session: aiohttp.ClientSession, metrics_url: str) -> float:
    """Poll /metrics and return vllm:num_requests_waiting."""
    try:
        async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
            text = await resp.text()
            for line in text.splitlines():
                if line.startswith("vllm:num_requests_waiting{"):
                    return float(line.split()[-1])
    except Exception:
        pass
    return 0.0


async def _send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    uid = uuid.uuid4().hex[:20]
    prompt = (
        f"[{uid}] Write a detailed technical essay about distributed systems. "
        "Cover consensus algorithms, fault tolerance, replication strategies, "
        "and real-world deployment considerations."
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.perf_counter()
    async with semaphore:
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                data = await resp.json()
                usage = data.get("usage", {})
                return {
                    "success": True,
                    "output_tokens": usage.get("completion_tokens", 0),
                    "latency": time.perf_counter() - t0,
                }
        except Exception as e:
            return {"success": False, "error": str(e), "latency": time.perf_counter() - t0}


async def test_concurrency_level(
    base_url: str,
    model: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
) -> dict:
    """Fire num_requests at given concurrency and monitor waiting queue."""
    completions_url = f"{base_url.rstrip('/')}/v1/completions"
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 16)

    max_waiting = 0.0
    waiting_samples = []

    async with aiohttp.ClientSession(connector=connector) as session:
        # Monitor task: poll /metrics while requests run
        stop = asyncio.Event()

        async def monitor():
            nonlocal max_waiting
            while not stop.is_set():
                w = await _check_waiting(session, metrics_url)
                waiting_samples.append(w)
                max_waiting = max(max_waiting, w)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.3)
                    break
                except asyncio.TimeoutError:
                    pass

        mon = asyncio.create_task(monitor())

        t0 = time.perf_counter()
        tasks = [
            _send_request(session, completions_url, model, max_tokens, semaphore)
            for _ in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

        stop.set()
        await mon

    good = [r for r in results if r["success"]]
    total_out = sum(r["output_tokens"] for r in good)
    avg_waiting = sum(waiting_samples) / len(waiting_samples) if waiting_samples else 0

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "successful": len(good),
        "failed": len(results) - len(good),
        "elapsed_s": round(elapsed, 2),
        "output_tps": round(total_out / elapsed, 1) if elapsed > 0 else 0,
        "max_waiting": max_waiting,
        "avg_waiting": round(avg_waiting, 1),
        "queued": max_waiting > 0,
    }


async def main(args: argparse.Namespace):
    concurrency_levels = [int(x) for x in args.levels.split(",")]
    # Scale num_requests to 2x concurrency (min 10)
    max_tokens = args.max_tokens

    print(f"Model: {args.model}")
    print(f"Testing concurrency levels: {concurrency_levels}")
    print(f"Max tokens per request: {max_tokens}")
    print()

    header = f"{'Conc':>6}  {'Reqs':>6}  {'OK':>4}  {'Fail':>4}  {'Out TPS':>9}  {'MaxWait':>8}  {'AvgWait':>8}  {'Queued':>6}"
    print(header)
    print("-" * len(header))

    for conc in concurrency_levels:
        n = max(10, conc * 2)
        result = await test_concurrency_level(
            args.base_url, args.model, conc, n, max_tokens,
        )
        queued = "YES" if result["queued"] else "-"
        print(
            f"{result['concurrency']:>6}  "
            f"{result['num_requests']:>6}  "
            f"{result['successful']:>4}  "
            f"{result['failed']:>4}  "
            f"{result['output_tps']:>9.1f}  "
            f"{result['max_waiting']:>8.0f}  "
            f"{result['avg_waiting']:>8.1f}  "
            f"{queued:>6}"
        )

        if result["queued"]:
            print(f"\n  Queuing detected at concurrency={conc} (max_waiting={result['max_waiting']:.0f})")
            # Continue to show how it degrades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test concurrency ceiling for vLLM")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--levels", default="1,5,10,20,30,40,50",
                        help="Comma-separated concurrency levels to test")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Output tokens per request (keep short for fast iteration)")
    args = parser.parse_args()
    asyncio.run(main(args))
