#!/usr/bin/env python3
"""
Standalone TPS sweep benchmark for vLLM (OpenAI-compatible completions API).

Measures output tokens/sec at varying concurrency levels (batch sizes) WITHOUT
restarting the server.  Batch size is controlled via asyncio.Semaphore on the
client; the server must be started with --max-num-seqs >= max(BATCH_SIZES).

Results are saved per batch size as JSON files and printed as a summary table.

Usage:
  python tps_sweep_minimal.py \\
    --model qwen35 \\
    --base-url http://localhost:8000 \\
    --batch-sizes 1,2,4,8,16,32,64,128 \\
    --input-len 512 \\
    --output-len 512 \\
    --num-prompts 200 \\
    --num-warmup 20 \\
    --results-dir ../results/qwen35
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import aiohttp

# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

_BASE_WORDS = (
    "the quick brown fox jumps over the lazy dog in a bustling city where "
    "many different large language models are deployed on powerful compute "
    "clusters running parallel inference workloads generating text at scale "
).split()


def make_prompt(target_tokens: int) -> str:
    """Approximate a prompt with *target_tokens* tokens.
    Uses rough heuristic: 1 token ≈ 0.75 words (for English prose).
    """
    words_needed = max(1, int(target_tokens * 0.75))
    repeats = (words_needed // len(_BASE_WORDS)) + 2
    words = (_BASE_WORDS * repeats)[:words_needed]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Request data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    output_tokens: int
    input_tokens: int
    latency_s: float
    ttft_s: Optional[float]
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Single async request (streaming, extracts usage from final chunk)
# ---------------------------------------------------------------------------


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        # include_usage causes vLLM to emit token counts in the final SSE chunk
        "stream_options": {"include_usage": True},
    }
    t_start = time.perf_counter()
    ttft: Optional[float] = None
    output_tokens = 0
    input_tokens = 0

    async with semaphore:
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=600)
            ) as resp:
                resp.raise_for_status()
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Usage appears in the last non-DONE chunk when stream_options used
                    usage = chunk.get("usage")
                    if usage:
                        output_tokens = usage.get("completion_tokens", 0)
                        input_tokens = usage.get("prompt_tokens", 0)

                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("text") and ttft is None:
                        ttft = time.perf_counter() - t_start

            return RequestResult(
                output_tokens=output_tokens,
                input_tokens=input_tokens,
                latency_s=time.perf_counter() - t_start,
                ttft_s=ttft,
                success=True,
            )
        except Exception as exc:
            return RequestResult(
                output_tokens=0,
                input_tokens=0,
                latency_s=time.perf_counter() - t_start,
                ttft_s=None,
                success=False,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Batch measurement
# ---------------------------------------------------------------------------


def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    idx = max(0, min(len(data) - 1, int(len(data) * p / 100)))
    return data[idx]


async def measure_batch_size(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    batch_size: int,
    num_warmup: int,
) -> dict:
    """Send all prompts with at most batch_size concurrent requests.
    Returns a dict of throughput and latency metrics.
    """
    url = f"{base_url.rstrip('/')}/v1/completions"
    semaphore = asyncio.Semaphore(batch_size)
    # Allow enough TCP connections for peak concurrency + a small buffer
    connector = aiohttp.TCPConnector(limit=batch_size + 16)

    async with aiohttp.ClientSession(connector=connector) as session:
        # --- Warmup (not timed) ---
        if num_warmup > 0:
            warmup_sem = asyncio.Semaphore(min(num_warmup, batch_size))
            warmup_tasks = [
                send_request(session, url, model, prompts[i % len(prompts)],
                             max_tokens, warmup_sem)
                for i in range(num_warmup)
            ]
            await asyncio.gather(*warmup_tasks)

        # --- Timed benchmark ---
        t_start = time.perf_counter()
        tasks = [
            send_request(session, url, model, p, max_tokens, semaphore)
            for p in prompts
        ]
        results: list[RequestResult] = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t_start

    good = [r for r in results if r.success]
    failed = len(results) - len(good)

    total_out = sum(r.output_tokens for r in good)
    total_in = sum(r.input_tokens for r in good)

    latencies = [r.latency_s for r in good]
    ttfts = [r.ttft_s for r in good if r.ttft_s is not None]

    output_tps = total_out / elapsed if elapsed > 0 else 0.0
    total_tps = (total_in + total_out) / elapsed if elapsed > 0 else 0.0

    return {
        "batch_size": batch_size,
        "num_prompts": len(prompts),
        "num_successful": len(good),
        "num_failed": failed,
        "elapsed_s": round(elapsed, 3),
        "output_tps": round(output_tps, 2),
        "total_tps": round(total_tps, 2),
        "request_tps": round(len(good) / elapsed, 3) if elapsed > 0 else 0.0,
        "latency_mean_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "latency_p50_s": round(_percentile(latencies, 50), 3),
        "latency_p90_s": round(_percentile(latencies, 90), 3),
        "latency_p99_s": round(_percentile(latencies, 99), 3),
        "ttft_mean_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else 0.0,
        "ttft_p50_s": round(_percentile(ttfts, 50), 3),
        "ttft_p99_s": round(_percentile(ttfts, 99), 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(results: list[dict]) -> None:
    header = f"{'Batch':>6}  {'Out TPS':>9}  {'Tot TPS':>9}  {'Lat P50':>8}  {'Lat P99':>8}  {'TTFT P99':>9}  {'Failed':>6}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['batch_size']:>6}  "
            f"{r['output_tps']:>9.1f}  "
            f"{r['total_tps']:>9.1f}  "
            f"{r['latency_p50_s']:>8.2f}  "
            f"{r['latency_p99_s']:>8.2f}  "
            f"{r['ttft_p99_s']:>9.2f}  "
            f"{r['num_failed']:>6}"
        )


async def _run(args: argparse.Namespace) -> None:
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    prompt = make_prompt(args.input_len)
    prompts = [prompt] * args.num_prompts

    all_results: list[dict] = []

    for bs in batch_sizes:
        print(f"\n[batch_size={bs}]  sending {args.num_prompts} prompts "
              f"(warmup={args.num_warmup}) ...", flush=True)
        result = await measure_batch_size(
            base_url=args.base_url,
            model=args.model,
            prompts=prompts,
            max_tokens=args.output_len,
            batch_size=bs,
            num_warmup=args.num_warmup,
        )
        all_results.append(result)
        print(
            f"  output_tps={result['output_tps']:.1f}  "
            f"total_tps={result['total_tps']:.1f}  "
            f"lat_p99={result['latency_p99_s']:.2f}s  "
            f"ttft_p99={result['ttft_p99_s']:.2f}s  "
            f"failed={result['num_failed']}"
        )

        out_file = results_dir / f"bs{bs:04d}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

    # Combined summary
    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "results": all_results,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_table(all_results)
    print(f"\nResults saved to {results_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TPS sweep: vary concurrency without restarting vLLM"
    )
    parser.add_argument("--model", required=True,
                        help="Model name as registered in vLLM (--served-model-name)")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128",
                        help="Comma-separated concurrency levels to sweep")
    parser.add_argument("--input-len", type=int, default=512,
                        help="Approximate prompt length in tokens")
    parser.add_argument("--output-len", type=int, default=512,
                        help="max_tokens per request")
    parser.add_argument("--num-prompts", type=int, default=200,
                        help="Requests per batch-size point")
    parser.add_argument("--num-warmup", type=int, default=20,
                        help="Warmup requests before timing (not counted)")
    parser.add_argument("--results-dir", default="results",
                        help="Directory to write per-batch-size JSON files")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
