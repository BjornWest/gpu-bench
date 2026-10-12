# GPU benchmarks

Inference throughput and concurrency experiments used for hardware selection
and evaluation planning. Model launch scripts live in `models/`, sweep clients
and summary tools in `bench/`, and sweep settings in `sweep.conf`. Generated
measurements go in the ignored `results/` directory.

## Model swapping experiments

`model_swap.sh` is a suspend/resume prototype from the work on swapping models
between audit turns, alongside the LMCache/KV-cache design work. It operates
on an existing vLLM server using sleep mode, process signals and CUDA
checkpointing. The related round-trip experiment is
`bench/test_sleep_checkpoint.sh`.

The additional [model-swapping experiments](bench/model_swap/README.md) cover
server boot/sleep/checkpoint cycles, disk versus RAM-backed weight reloads,
and LMCache KV persistence/prefetching. This experimental workflow was not
used for the main thesis evaluation.

From this repository, its commands are:

```sh
./model_swap.sh status 8000
./model_swap.sh suspend 8000
./model_swap.sh resume 8000
```

The script expects `cuda-checkpoint` in the `gpu-bench/` root and a server
started with `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`. Its assumptions
were tailored to gpt-oss-20b. It provides the suspend/resume mechanism; LMCache
KV retention and model selection belong to the surrounding system design.

## InferenceX reference data

`reference/inferencex/` contains five reference CSVs used to inform throughput
assumptions for DeepSeek-R1, DeepSeek-V4, GLM-5, Kimi-K2.5 and MiniMax-M2.5.
These reference measurements are separate from locally generated sweep results.

From this repository, run:

```sh
python3 bench/parse_inferencex.py
```

The parser uses the Python standard library and resolves the bundled CSVs
relative to its own location. It prints single-node B200/B300 throughput
comparisons and configuration snippets, including estimates adjusted for
concurrency and context length. Estimated values and model proxies are
identified in the output; they are planning assumptions, not study outcomes.

The audit schedule and token-budget calculator remains in the parent audit-lab
repository at `petri_tests/eval_calculator/`. Its schedule-specific logic depends
on the evaluation system; parsing these benchmark references does not.
