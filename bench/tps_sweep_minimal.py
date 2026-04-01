#!/usr/bin/env python3
"""
Standalone TPS sweep benchmark for vLLM (OpenAI-compatible completions API).

Two test modes:

  decode  — Isolate decode throughput. All requests share the same prompt so
            prefix caching makes prefill essentially free after the first hit.
            Sweeps batch sizes.

  prefill — Isolate prefill throughput. Each request has a unique prompt (no
            caching), output is capped at a few tokens. Sweeps input lengths
            and optionally batch sizes.

Key design choices:
  - No server restarts: batch size is controlled via asyncio.Semaphore on the
    client; the server must be started with --max-num-seqs >= max(BATCH_SIZES).
  - Saturation detection: polls /metrics for vllm:num_requests_waiting; stops
    the sweep when the KV cache fills (3+ consecutive non-zero readings).

Usage:
  # Decode sweep (batch sizes)
  python tps_sweep_minimal.py decode \\
    --model qwen35 --base-url http://localhost:8000 \\
    --batch-sizes 1,64,128,256,512 \\
    --input-len 512 --output-len 1024 \\
    --results-dir ../results/qwen35/decode

  # Prefill sweep (input lengths × batch sizes)
  python tps_sweep_minimal.py prefill \\
    --model qwen35 --base-url http://localhost:8000 \\
    --input-lens 4096,16384,32768,65536 \\
    --batch-sizes 1,16,64 \\
    --output-len 8 \\
    --results-dir ../results/qwen35/prefill

  # Legacy (no subcommand) — behaves like old combined test
  python tps_sweep_minimal.py \\
    --model qwen35 ...
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp

# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

# Diverse technical topics — rotated across requests so adjacent prompts differ
_TOPICS = [
    "KV cache management and memory allocation in LLM inference engines",
    "distributed consensus beyond Raft: EPaxos, Hermes, and Flexible Paxos",
    "memory allocator design for latency-sensitive systems (jemalloc, mimalloc, tcmalloc)",
    "tensor and expert parallelism strategies in large model training and inference",
    "lock-free and wait-free data structures: algorithms, correctness proofs, pitfalls",
    "storage engine design trade-offs: B-tree vs LSM-tree vs COLA",
    "speculative decoding: draft model selection, verification batching, acceptance rates",
    "NUMA-aware scheduling and memory placement in multi-socket servers",
    "RDMA and kernel-bypass networking for high-throughput distributed systems",
    "continuous batching in LLM serving: iteration-level scheduling and preemption",
    "write-optimized databases: compaction strategies, read amplification, space overhead",
    "GPU memory hierarchy: L1/L2 caches, shared memory, register files, coalescing",
    "disaggregated prefill and decode in LLM inference: latency and throughput trade-offs",
    "network function virtualization: DPDK, XDP, eBPF packet processing pipelines",
    "compiler optimizations for heterogeneous compute: auto-vectorization, loop tiling, fusion",
    "CRDTs and eventual consistency models in distributed databases",
    "persistent memory (Optane) programming models and failure-atomicity primitives",
    "tail latency in distributed systems: causes, measurement, and mitigation strategies",
    "flash attention and memory-efficient attention variants: FlashAttention-2, PagedAttention",
    "prefix trees (tries) in IP routing, autocompletion, and key-value index structures",
]

# Padding passages — varied so long prompts don't repeat the same paragraph
_PASSAGES = [
    "Modern computing systems face fundamental trade-offs between throughput, latency, "
    "energy efficiency, and fault tolerance. Every design decision carries implications "
    "across multiple dimensions that must be evaluated against specific workload "
    "characteristics, hardware capabilities, and operational constraints.",

    "Tail latency — particularly at the 99th and 99.9th percentiles — often determines "
    "practical system usability in production environments where SLA violations carry "
    "financial and reputational consequences. Understanding the sources of latency variance "
    "is prerequisite to any effective mitigation strategy.",

    "The memory hierarchy of modern processors spans registers, L1/L2/L3 caches, main "
    "memory, and persistent storage, each differing by orders of magnitude in bandwidth, "
    "latency, and capacity. Algorithms and data structures must be designed with explicit "
    "awareness of these tiers to achieve competitive performance.",

    "Concurrency control in multi-core systems requires careful reasoning about cache "
    "coherence protocols, memory ordering, and the interaction between hardware and "
    "compiler optimizations. Lock-free algorithms offer scalability advantages but "
    "demand rigorous correctness arguments under all observable execution interleavings.",

    "Distributed systems are defined by partial failure: individual components may fail "
    "independently while others continue operating. Building reliable abstractions atop "
    "unreliable hardware requires explicit reasoning about failure modes, network "
    "partitions, and the consistency guarantees achievable under each scenario.",

    "Inference efficiency for large language models is constrained by memory bandwidth "
    "rather than compute in the memory-bound decode phase. Arithmetic intensity — the "
    "ratio of floating-point operations to memory bytes accessed — is the key metric "
    "distinguishing compute-bound from memory-bound workloads.",

    "Compiler optimization passes operate on intermediate representations that abstract "
    "away machine-specific details while preserving semantic equivalence. Transformations "
    "such as loop unrolling, vectorization, and instruction scheduling must respect "
    "data dependencies and maintain observable behavior at optimization boundaries.",

    "Network protocols balance reliability, ordering, and efficiency across diverse "
    "link characteristics. Congestion control algorithms must infer network state from "
    "observable signals such as round-trip time variation and packet loss, adapting "
    "transmission rates to maximize utilization without inducing persistent queuing.",

    "Database query optimizers generate execution plans by estimating the cost of "
    "alternative operator orderings, join strategies, and index access paths. Cardinality "
    "estimation errors propagate through plan trees and can cause orders-of-magnitude "
    "performance regressions in complex multi-table queries.",

    "Hardware accelerators achieve efficiency by exploiting data parallelism, operation "
    "fusion, and reduced-precision arithmetic. Programming models for heterogeneous "
    "systems must expose enough structure for the compiler and runtime to map "
    "computations onto specialized execution units without sacrificing generality.",
]

_CHARS_PER_TOKEN = 3.8  # conservative estimate for mixed technical/code text


def _pad_to_chars(target_chars: int, start_idx: int) -> str:
    """Repeat passages in rotation until we reach approximately target_chars."""
    parts: list[str] = []
    total = 0
    i = start_idx
    while total < target_chars:
        p = _PASSAGES[i % len(_PASSAGES)]
        parts.append(p)
        total += len(p) + 1
        i += 1
    text = " ".join(parts)
    return text[:target_chars]


def make_prompt(
    target_input_tokens: int,
    target_output_tokens: int,
    request_idx: int,
    *,
    shared_prefix: bool = False,
) -> str:
    """
    Build a prompt that:
      1. Starts with a UUID prefix (unique per request, or shared for cache testing).
      2. Contains enough context to fill target_input_tokens.
      3. Asks for a thorough essay to encourage target_output_tokens of output.

    When shared_prefix=True, all requests use the same prefix so prefix caching
    makes prefill essentially free after the first request. Used for decode tests.
    """
    if shared_prefix:
        uid = "shared-bench-prefix-00"    # identical across requests — enables prefix caching
        request_idx = 0                   # same topic + padding for all — fully shared prompt
    else:
        uid = uuid.uuid4().hex[:20]       # unique per request — defeats prefix caching
    topic = _TOPICS[request_idx % len(_TOPICS)]

    if target_input_tokens <= 2048:
        # Short-to-medium ISL: direct essay request, padded with context
        instruction = (
            f"[{uid}]\n\n"
            f"Write a thorough technical essay on: {topic}\n\n"
            f"Your essay must cover all of the following:\n"
            f"  1. Motivation and the problem being solved\n"
            f"  2. Theoretical foundations and core concepts\n"
            f"  3. Key algorithms, data structures, and mechanisms\n"
            f"  4. Implementation details and engineering trade-offs\n"
            f"  5. Performance characteristics, bottlenecks, and optimization strategies\n"
            f"  6. Comparison with alternative approaches\n"
            f"  7. Real-world deployment experience and case studies\n"
            f"  8. Current limitations and open research questions\n"
            f"  9. Future directions and emerging work\n\n"
            f"Background context:\n"
        )
        suffix = "\n\nEssay:\n"
        reserved_chars = int((len(instruction) + len(suffix)) * 1.1)
        padding_chars = max(0, int(target_input_tokens * _CHARS_PER_TOKEN) - reserved_chars)
        return instruction + _pad_to_chars(padding_chars, request_idx) + suffix

    else:
        # Long ISL: document analysis format (~8k tokens)
        header = f"[{uid}]\n\nReference Document:\n\n"
        question = (
            f"\n\nTask: Based on the reference document above, write a comprehensive "
            f"technical analysis of {topic}. Cover design principles, algorithms, "
            f"performance characteristics, implementation trade-offs, real-world "
            f"applications, known limitations, and future research directions. "
            f"Your analysis should be detailed and well-structured.\n\nAnalysis:\n"
        )
        reserved_chars = int((len(header) + len(question)) * 1.1)
        doc_chars = max(0, int(target_input_tokens * _CHARS_PER_TOKEN) - reserved_chars)
        return header + _pad_to_chars(doc_chars, request_idx) + question


# ---------------------------------------------------------------------------
# Saturation monitor
# ---------------------------------------------------------------------------


KV_CACHE_SATURATION_THRESHOLD = 0.95


async def _monitor_saturation(
    metrics_url: str,
    stop_event: asyncio.Event,
) -> dict:
    """
    Poll vLLM /metrics every 0.5s for gpu_cache_usage_perc.
    Saturation is declared when KV cache usage exceeds the threshold for
    3+ consecutive polls (~1.5s), indicating the cache is genuinely full
    (not just transient request queuing).
    """
    result: dict = {
        "max_kv_cache_pct": 0.0,
        "max_waiting": 0.0,
        "saturated": False,
        "peak_consecutive": 0,
    }
    consecutive = 0

    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        while not stop_event.is_set():
            try:
                async with session.get(
                    metrics_url, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    text = await resp.text()
                    kv_pct = 0.0
                    waiting = 0.0
                    for line in text.splitlines():
                        if line.startswith("vllm:gpu_cache_usage_perc"):
                            try:
                                kv_pct = max(kv_pct, float(line.split()[-1]))
                            except (ValueError, IndexError):
                                pass
                        elif line.startswith("vllm:num_requests_waiting{"):
                            try:
                                waiting += float(line.split()[-1])
                            except (ValueError, IndexError):
                                pass

                    result["max_kv_cache_pct"] = max(result["max_kv_cache_pct"], kv_pct)
                    result["max_waiting"] = max(result["max_waiting"], waiting)

                    if kv_pct >= KV_CACHE_SATURATION_THRESHOLD:
                        consecutive += 1
                        result["peak_consecutive"] = max(
                            result["peak_consecutive"], consecutive
                        )
                        if consecutive >= 3:
                            result["saturated"] = True
                    else:
                        consecutive = 0

            except asyncio.CancelledError:
                break
            except Exception:
                pass

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.5)
                break  # stop_event was set — exit loop
            except asyncio.TimeoutError:
                pass  # normal: poll again

    return result


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    output_tokens: int
    input_tokens: int
    latency_s: float
    ttft_s: Optional[float]
    success: bool
    error: Optional[str] = None
    output_hash: Optional[str] = None


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    *,
    greedy: bool = False,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0 if greedy else 0.6,
        "stream": True,
        # Instructs vLLM to include token counts in the final SSE chunk
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
    input_len: int,
    output_len: int,
    num_prompts: int,
    batch_size: int,
    num_warmup: int,
    *,
    shared_prefix: bool = False,
    greedy: bool = False,
    test_mode: str = "combined",
) -> dict:
    """
    Send num_prompts requests with at most batch_size concurrent, measure TPS.
    Concurrently polls /metrics to detect KV cache saturation.

    shared_prefix: if True, all prompts share the same prefix (for decode tests).
    greedy: if True, use temperature=0 (identical outputs, maximizes KV sharing).
    test_mode: label stored in results ("decode", "prefill", or "combined").
    """
    completions_url = f"{base_url.rstrip('/')}/v1/completions"
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    semaphore = asyncio.Semaphore(batch_size)
    connector = aiohttp.TCPConnector(limit=batch_size + 16)

    # Build prompts
    prompts = [
        make_prompt(input_len, output_len, i, shared_prefix=shared_prefix)
        for i in range(max(num_prompts, num_warmup))
    ]

    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup (not timed, no saturation check)
        if num_warmup > 0:
            wu_sem = asyncio.Semaphore(min(num_warmup, batch_size))
            wu_tasks = [
                send_request(session, completions_url, model, prompts[i], output_len, wu_sem, greedy=greedy)
                for i in range(num_warmup)
            ]
            await asyncio.gather(*wu_tasks)

        # Start saturation monitor
        stop_monitor = asyncio.Event()
        monitor_task = asyncio.create_task(
            _monitor_saturation(metrics_url, stop_monitor)
        )

        # Timed benchmark
        t_start = time.perf_counter()
        tasks = [
            send_request(
                session, completions_url, model,
                prompts[i % len(prompts)],
                output_len, semaphore,
                greedy=greedy,
            )
            for i in range(num_prompts)
        ]
        results: list[RequestResult] = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t_start

        # Stop monitor and collect saturation data
        stop_monitor.set()
        saturation = await monitor_task

    good = [r for r in results if r.success]
    failed = len(results) - len(good)
    total_out = sum(r.output_tokens for r in good)
    total_in = sum(r.input_tokens for r in good)
    latencies = [r.latency_s for r in good]
    ttfts = [r.ttft_s for r in good if r.ttft_s is not None]

    return {
        "test_mode": test_mode,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "num_prompts": num_prompts,
        "num_successful": len(good),
        "num_failed": failed,
        "elapsed_s": round(elapsed, 3),
        "output_tps": round(total_out / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tps": round((total_in + total_out) / elapsed, 2) if elapsed > 0 else 0.0,
        "request_tps": round(len(good) / elapsed, 3) if elapsed > 0 else 0.0,
        "latency_mean_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "latency_p50_s": round(_percentile(latencies, 50), 3),
        "latency_p90_s": round(_percentile(latencies, 90), 3),
        "latency_p99_s": round(_percentile(latencies, 99), 3),
        "ttft_mean_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else 0.0,
        "ttft_p50_s": round(_percentile(ttfts, 50), 3),
        "ttft_p99_s": round(_percentile(ttfts, 99), 3),
        "max_kv_cache_pct": saturation["max_kv_cache_pct"],
        "max_requests_waiting": saturation["max_waiting"],
        "kv_saturated": saturation["saturated"],
        "peak_consecutive_saturation_polls": saturation["peak_consecutive"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


MIN_PROMPTS = 20
MAX_PROMPTS = 1000


def _scale_prompts(base: int, batch_size: int) -> int:
    """Scale request count with batch size.

    At batch_size=1, sequential requests are slow — use fewer.
    At large batch sizes, need enough requests to fill the batch and
    get stable measurements.
    Ensures at least 2× the batch size so the pipeline stays full.
    """
    scaled = max(MIN_PROMPTS, min(batch_size * 3, base, MAX_PROMPTS))
    return max(scaled, batch_size * 2)


def _print_table(results: list[dict]) -> None:
    header = (
        f"{'Batch':>6}  {'ISL':>6}  {'OSL':>6}  {'N':>5}  {'Out TPS':>9}  {'Tot TPS':>9}  "
        f"{'Lat P50':>8}  {'Lat P99':>8}  {'TTFT P99':>9}  "
        f"{'KV%':>5}  {'Sat':>4}  {'Fail':>5}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        sat = "YES" if r.get("kv_saturated") else "-"
        kv_pct = r.get("max_kv_cache_pct", 0) * 100
        print(
            f"{r['batch_size']:>6}  "
            f"{r.get('input_len', '?'):>6}  "
            f"{r.get('output_len', '?'):>6}  "
            f"{r['num_prompts']:>5}  "
            f"{r['output_tps']:>9.1f}  "
            f"{r['total_tps']:>9.1f}  "
            f"{r['latency_p50_s']:>8.2f}  "
            f"{r['latency_p99_s']:>8.2f}  "
            f"{r['ttft_p99_s']:>9.2f}  "
            f"{kv_pct:>5.1f}  "
            f"{sat:>4}  "
            f"{r['num_failed']:>5}"
        )


def _log_result(result: dict) -> None:
    sat_str = "  *** KV SATURATED ***" if result["kv_saturated"] else ""
    print(
        f"  output_tps={result['output_tps']:.1f}  "
        f"total_tps={result['total_tps']:.1f}  "
        f"lat_p99={result['latency_p99_s']:.2f}s  "
        f"ttft_p99={result['ttft_p99_s']:.2f}s  "
        f"max_waiting={result['max_requests_waiting']:.0f}  "
        f"failed={result['num_failed']}"
        f"{sat_str}"
    )


def _save_summary(results_dir: Path, args: argparse.Namespace,
                   all_results: list[dict], test_mode: str) -> None:
    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "test_mode": test_mode,
        "num_prompts": args.num_prompts,
        "results": all_results,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _print_table(all_results)
    print(f"\nResults saved to {results_dir}/")


# ── Decode sweep ──────────────────────────────────────────────────────

async def _run_decode(args: argparse.Namespace) -> None:
    """Sweep batch sizes with shared prefix (isolate decode throughput)."""
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for bs in batch_sizes:
        n_prompts = _scale_prompts(args.num_prompts, bs)
        print(
            f"\n[decode] batch_size={bs}  ISL={args.input_len}  OSL={args.output_len}  "
            f"prompts={n_prompts}  warmup={args.num_warmup}",
            flush=True,
        )
        result = await measure_batch_size(
            base_url=args.base_url,
            model=args.model,
            input_len=args.input_len,
            output_len=args.output_len,
            num_prompts=n_prompts,
            batch_size=bs,
            num_warmup=args.num_warmup,
            shared_prefix=True,
            greedy=True,
            test_mode="decode",
        )
        all_results.append(result)
        _log_result(result)

        out_file = results_dir / f"decode_bs{bs:06d}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        if result["kv_saturated"]:
            print(f"\nKV cache saturated — stopping sweep at batch_size={bs}.")
            break

    _save_summary(results_dir, args, all_results, "decode")


# ── Prefill sweep ─────────────────────────────────────────────────────

async def _run_prefill(args: argparse.Namespace) -> None:
    """Sweep input lengths (and optionally batch sizes) with unique prompts."""
    input_lens = [int(x) for x in args.input_lens.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for isl in input_lens:
        for bs in batch_sizes:
            n_prompts = _scale_prompts(args.num_prompts, bs)
            print(
                f"\n[prefill] ISL={isl}  batch_size={bs}  OSL={args.output_len}  "
                f"prompts={n_prompts}  warmup={args.num_warmup}",
                flush=True,
            )
            result = await measure_batch_size(
                base_url=args.base_url,
                model=args.model,
                input_len=isl,
                output_len=args.output_len,
                num_prompts=n_prompts,
                batch_size=bs,
                num_warmup=args.num_warmup,
                shared_prefix=False,
                test_mode="prefill",
            )
            all_results.append(result)
            _log_result(result)

            out_file = results_dir / f"prefill_isl{isl:06d}_bs{bs:06d}.json"
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)

            if result["kv_saturated"]:
                print(f"\nKV cache saturated — stopping at ISL={isl} batch_size={bs}.")
                break
        else:
            continue
        break  # also break outer loop on saturation

    _save_summary(results_dir, args, all_results, "prefill")


# ── Legacy (combined) ─────────────────────────────────────────────────

async def _run_combined(args: argparse.Namespace) -> None:
    """Original sweep: unique prompts, batch size sweep (no mode isolation)."""
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for bs in batch_sizes:
        n_prompts = _scale_prompts(args.num_prompts, bs)
        print(
            f"\n[batch_size={bs}]  prompts={n_prompts}  "
            f"ISL={args.input_len}  OSL={args.output_len}  "
            f"warmup={args.num_warmup}",
            flush=True,
        )
        result = await measure_batch_size(
            base_url=args.base_url,
            model=args.model,
            input_len=args.input_len,
            output_len=args.output_len,
            num_prompts=n_prompts,
            batch_size=bs,
            num_warmup=args.num_warmup,
        )
        all_results.append(result)
        _log_result(result)

        out_file = results_dir / f"bs{bs:06d}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        if result["kv_saturated"]:
            print(f"\nKV cache saturated — stopping sweep at batch_size={bs}.")
            break

    _save_summary(results_dir, args, all_results, "combined")


# ── CLI ───────────────────────────────────────────────────────────────

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True,
                        help="--served-model-name value used when launching vLLM")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--num-prompts", type=int, default=200,
                        help="Requests per sweep point")
    parser.add_argument("--num-warmup", type=int, default=20,
                        help="Warmup requests before timing (not counted)")
    parser.add_argument("--results-dir", default="results")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TPS sweep: isolate decode or prefill throughput"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -- decode --
    p_dec = subparsers.add_parser("decode",
        help="Isolate decode throughput (shared prefix, sweep batch sizes)")
    _add_common_args(p_dec)
    p_dec.add_argument("--batch-sizes", default="1,64,128,256,512",
                        help="Comma-separated concurrency levels to sweep")
    p_dec.add_argument("--input-len", type=int, default=512,
                        help="Prompt length in tokens (shared across requests)")
    p_dec.add_argument("--output-len", type=int, default=1024,
                        help="max_tokens per request")

    # -- prefill --
    p_pre = subparsers.add_parser("prefill",
        help="Isolate prefill throughput (unique prompts, sweep input lengths)")
    _add_common_args(p_pre)
    p_pre.add_argument("--input-lens", default="4096,16384,32768,65536",
                        help="Comma-separated input lengths to sweep")
    p_pre.add_argument("--batch-sizes", default="1,16,64",
                        help="Concurrency levels per input length")
    p_pre.add_argument("--output-len", type=int, default=8,
                        help="max_tokens per request (keep small)")

    # -- combined (legacy) --
    p_comb = subparsers.add_parser("combined",
        help="Original combined test (unique prompts, sweep batch sizes)")
    _add_common_args(p_comb)
    p_comb.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096",
                        help="Comma-separated concurrency levels to sweep")
    p_comb.add_argument("--input-len", type=int, default=1024,
                        help="Target prompt length in tokens (ISL)")
    p_comb.add_argument("--output-len", type=int, default=1024,
                        help="max_tokens per request (OSL cap)")

    args = parser.parse_args()

    if args.mode == "decode":
        asyncio.run(_run_decode(args))
    elif args.mode == "prefill":
        asyncio.run(_run_prefill(args))
    else:
        asyncio.run(_run_combined(args))


if __name__ == "__main__":
    main()
