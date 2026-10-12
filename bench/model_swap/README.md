# Model-swapping experiments

Historical GPU experiments from the unused audit model-swapping workflow.
These are separate from the fixed-endpoint evaluation reported in the thesis.

| File | Purpose |
| --- | --- |
| `test_sleep_checkpoint.sh` | Boot vLLM, sleep/checkpoint/restore/reload, with optional RAM-backed weights and two-model cycles |
| `test_tmpfs_vs_disk.sh` | Compare disk and RAM-backed weight reloads |
| `test_lmcache.sh` | Check KV-cache persistence and prefetching across model swaps |
| `prefetch_kv.py` | Identify LMCache chunk files and optionally load them; used by `test_lmcache.sh` |

These scripts launch GPU workloads and retain their original vLLM, LMCache,
model-path and, in places, NixOS assumptions. Configure the appropriate
environment and supply explicit model paths before using them. The scripts
resolve the benchmark repository root from their location and expect the
`cuda-checkpoint` binary there. Generated logs, caches and temporary model
links stay local.

The separate [`../test_sleep_checkpoint.sh`](../test_sleep_checkpoint.sh)
checks a server that is already running. These larger experiments start
servers and exercise additional swap configurations.
