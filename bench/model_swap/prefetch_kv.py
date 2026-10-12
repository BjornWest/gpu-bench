#!/usr/bin/env python3
"""
prefetch_kv.py — Identify and preload LMCache KV chunk files for specific conversations.

Reproduces LMCache's sha256_cbor rolling prefix hash to determine which .pt files
on disk correspond to given conversation texts, then optionally loads them into
memory (warming the OS page cache / keeping tensors in system RAM).

Requires PYTHONHASHSEED to be set and match what the vLLM server used.
Requires LMCACHE_PRE_CACHING_HASH_ALGORITHM=sha256_cbor on the vLLM server.

Usage:
  PYTHONHASHSEED=0 python prefetch_kv.py \\
      --model-name <served_model_name> \\
      --model-path <tokenizer_path> \\
      --kv-dir <lmcache_disk_dir> \\
      [--load] \\
      <conv1_text> [<conv2_text> ...]

Exits 0 if all chunk files were found, 1 if any are missing.
JSON output to stdout: { all_found, conversations: [{token_count, total_chunks, found_chunks, ...}] }
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cbor2
import torch
from transformers import AutoTokenizer

CHUNK_SIZE = 256  # Must match LMCACHE_CHUNK_SIZE (default 256)


def sha256_cbor(obj) -> bytes:
    """Reproduce vLLM's sha256_cbor: CBOR-canonical-serialize then SHA-256."""
    return hashlib.sha256(cbor2.dumps(obj, canonical=True)).digest()


def get_none_hash() -> bytes:
    """
    Reproduce LMCache's NONE_HASH (the initial prefix hash before any tokens).

    vLLM's init_none_hash sets NONE_HASH = hash_fn(PYTHONHASHSEED) when the env var
    is present; LMCache copies it. With PYTHONHASHSEED=0 and sha256_cbor, this is
    sha256_cbor("0") — deterministic and cross-process reproducible.
    """
    seed = os.environ.get("PYTHONHASHSEED")
    if seed is None:
        print(
            "ERROR: PYTHONHASHSEED not set. Cannot reproduce chunk hashes.\n"
            "Set PYTHONHASHSEED=0 matching the vLLM server's environment.",
            file=sys.stderr,
        )
        sys.exit(2)
    return sha256_cbor(seed)


def compute_chunk_hashes(token_ids: list) -> list:
    """
    Compute rolling prefix chunk hashes for a token sequence.
    Mirrors ChunkedTokenDatabase._prefix_hash() + _hash_tokens() in LMCache.

    With save_unfull_chunk=True (LMCache default), the final partial chunk
    (<256 tokens) is also included.

    Hash chain:
        h_0 = sha256_cbor((NONE_HASH, chunk_0_tokens_tuple, ()))
        h_i = sha256_cbor((h_{i-1}, chunk_i_tokens_tuple, ()))
    """
    prefix_hash = get_none_hash()
    hashes = []
    for i in range(0, len(token_ids), CHUNK_SIZE):
        chunk = tuple(token_ids[i : i + CHUNK_SIZE])
        # Canonical hash input: (prev_prefix_hash, tokens_tuple, extra_keys_tuple)
        # extra_keys is always () for standard text inference (no LoRA, no multimodal)
        prefix_hash = sha256_cbor((prefix_hash, chunk, ()))
        hashes.append(prefix_hash)
    return hashes


def scan_kv_dir(kv_dir: Path, model_name: str) -> dict:
    """
    Scan the KV cache directory and build a lookup: chunk_hash_hex -> file_path.

    File format: {model_name}@{world_size}@{worker_id}@{chunk_hash_hex}@{dtype}.pt
    '/' in model_name is replaced with '-' in the filename.

    Filters to only files matching the given model_name so cross-model collisions
    are impossible (different models use different KV dirs anyway, but this is safe).
    """
    safe_name = model_name.replace("/", "-")
    hash_to_file = {}
    for f in kv_dir.glob("*.pt"):
        parts = f.stem.split("@")
        # Expected: model@world_size@worker_id@hash@dtype (5 parts minimum)
        if len(parts) < 5:
            continue
        if parts[0] != safe_name:
            continue
        chunk_hash_hex = parts[3]
        hash_to_file[chunk_hash_hex] = f
    return hash_to_file


def main():
    parser = argparse.ArgumentParser(
        description="Identify and optionally preload LMCache KV chunk files for conversations."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Served model name (must match --served-model-name used with vLLM)",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the model directory (used for the tokenizer)",
    )
    parser.add_argument(
        "--kv-dir",
        required=True,
        help="LMCache disk KV cache directory (LMCACHE_LOCAL_DISK path, without file:// prefix)",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help=(
            "Load identified .pt files into memory via torch.load. "
            "Warms the OS page cache so subsequent disk reads are served from RAM. "
            "Note: for durable prefetch across swap gaps, hold the returned tensors "
            "in a long-lived process."
        ),
    )
    parser.add_argument(
        "conversations",
        nargs="+",
        help=(
            "User-message texts of conversations to prefetch. "
            "In production this would be reconstructed from transcript IDs."
        ),
    )
    args = parser.parse_args()

    kv_dir = Path(args.kv_dir)
    if not kv_dir.is_dir():
        print(f"ERROR: KV cache directory not found: {kv_dir}", file=sys.stderr)
        sys.exit(2)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    # Scan disk once; build hash→file lookup (avoids needing to know dtype)
    existing_chunks = scan_kv_dir(kv_dir, args.model_name)

    result = {
        "model_name": args.model_name,
        "kv_dir": str(kv_dir),
        "total_disk_chunks": len(existing_chunks),
        "all_found": True,
        "total_loaded_bytes": 0,
        "conversations": [],
    }

    # Keep loaded tensors alive so OS doesn't reclaim pages before the swap completes
    loaded_cache = {}

    for i, conv_text in enumerate(args.conversations):
        # Tokenize exactly as vLLM does: apply chat template with generation prompt
        messages = [{"role": "user", "content": conv_text}]
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )

        chunk_hashes = compute_chunk_hashes(token_ids)

        found_files = []
        missing_hashes = []
        for h in chunk_hashes:
            hex_h = h.hex()
            if hex_h in existing_chunks:
                found_files.append(str(existing_chunks[hex_h]))
            else:
                missing_hashes.append(hex_h)

        conv_result = {
            "conv_index": i,
            "token_count": len(token_ids),
            "total_chunks": len(chunk_hashes),
            "found_chunks": len(found_files),
            "missing_chunks": len(missing_hashes),
        }

        if missing_hashes:
            # Show first few missing hashes to help diagnose hash mismatches
            conv_result["sample_missing_hashes"] = missing_hashes[:3]
            result["all_found"] = False

        if args.load and found_files:
            nbytes = 0
            for fp in found_files:
                tensor = torch.load(fp, map_location="cpu", weights_only=True)
                loaded_cache[fp] = tensor
                nbytes += Path(fp).stat().st_size
            conv_result["loaded_bytes"] = nbytes
            result["total_loaded_bytes"] += nbytes

        result["conversations"].append(conv_result)

    print(json.dumps(result, indent=2))

    if args.load and loaded_cache:
        total_mib = result["total_loaded_bytes"] / 1024 / 1024
        print(
            f"Loaded {len(loaded_cache)} tensors ({total_mib:.1f} MiB) into memory. "
            "OS page cache is warm.",
            file=sys.stderr,
        )

    sys.exit(0 if result["all_found"] else 1)


if __name__ == "__main__":
    main()
