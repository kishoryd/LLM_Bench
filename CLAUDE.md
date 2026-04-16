# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

LLM Bench is a multi-backend GPU inference benchmarking suite. It sweeps batch sizes across context length tiers (short/medium/long/near-limit), measures TTFT, end-to-end latency, and token throughput, and saves results as JSON/CSV.

## Running Benchmarks

```bash
# Single backend
python scripts/run_bench.py --backend native --model param2_17b

# Multiple backends
python scripts/run_bench.py --backend native vllm --model param2_17b

# All backends, multiple GPU counts, with quantization and profiler
python scripts/run_bench.py --backend all --model param2_17b --gpus 1 2 --quantize int4 --profiler pytorch

# Custom batch sizes and output
python scripts/run_bench.py --backend native --model param2_17b --batch-sizes 1,4,16 --output results/my_run.json --csv

# List available models/backends
python scripts/run_bench.py --list-models
python scripts/run_bench.py --list-backends
```

## Running Tests

```bash
# All tests
python -m unittest discover tests/

# Single test file
python -m unittest tests/test_backends.py

# Single test case
python -m unittest tests.test_backends.TestBackendRegistry.test_all_backends_listed
```

## Architecture

The data flow is: CLI (`scripts/run_bench.py`) → `BenchmarkRunner` (`src/runner.py`) → Backend → results.

**`src/runner.py` — `BenchmarkRunner`**: The core orchestrator. Loads configs, instantiates the backend, optionally wraps runs with a profiler, sweeps batch sizes across all context tiers, aggregates timing stats (avg/std), handles OOM gracefully, and returns structured result dicts.

**`src/backends/`**: Each backend implements `BaseBackend` (setup / generate / teardown). `generate()` must return a dict with keys `ttft_s`, `decode_s`, `e2e_s`, `output_tokens`, `tok_per_s`, `gpu_memory`. Backends are lazy-imported to avoid pulling in optional dependencies. Supported: `native`, `vllm`, `sglang`, `triton`, `ollama`.

**`src/models/`**: `ModelRegistry` resolves model names to YAML configs (from `configs/models/`). `build_context_tiers()` builds prompts at each target token length using the model's tokenizer.

**`src/profilers/`**: Optional wrappers (PyTorch profiler, Nsight). The profiler is only active for `batch_size=1`, first run of each tier.

**`src/gpu/`**: `GPUStatsSampler` polls GPU utilisation in a background thread during each generate call. `gpu_memory_snapshot()` captures per-device VRAM usage.

**`src/utils/`**: Config loaders (all YAML files are read from `configs/`), result reporter (`print_summary`, `save_json`, `save_csv`).

## Adding a New Backend

1. Create `src/backends/your_backend.py` inheriting from `BaseBackend`
2. Add it to `BACKEND_REGISTRY` in `src/backends/__init__.py`
3. Add `configs/backends/your_backend.yaml`

## Adding a New Model

Create `configs/models/your_model.yaml` — the registry picks it up automatically. Required fields: `name`, `path`, `max_context`, `default_dtype`, `trust_remote_code`.

## Config Files

- `configs/bench.yaml` — warmup/measure runs, batch sizes, context tiers, output format
- `configs/models/*.yaml` — model path, dtype, max context, prompt seeds
- `configs/backends/*.yaml` — backend-specific server/engine settings
- `configs/profiles/*.yaml` — profiler settings

Server-based backends (vllm, sglang, triton, ollama) expect an already-running inference server. Use the helper scripts `scripts/start_vllm.sh`, `scripts/start_sglang.sh`, `scripts/start_triton.sh` to launch them.
