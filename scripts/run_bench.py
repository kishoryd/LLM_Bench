#!/usr/bin/env python3
"""
CLI entry point for the LLM Bench suite.

Usage:
  python scripts/run_bench.py --backend native --model param2_17b
  python scripts/run_bench.py --backend native vllm --model param2_17b --profiler pytorch
  python scripts/run_bench.py --backend all --model param2_17b --gpus 1 2 --quantize int4
"""

import sys
import os
import argparse

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.runner import BenchmarkRunner
from src.backends import ALL_BACKENDS
from src.utils.config import list_available_models, list_available_backends
from src.utils.report import print_summary, save_json, save_csv


def main():
    parser = argparse.ArgumentParser(
        description="LLM Bench — Multi-backend GPU inference benchmark suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--backend", nargs="+", default=["native"],
        help=(
            "Backends to benchmark. Options:\n"
            "  native, vllm, sglang, triton, ollama, all\n"
            "  E.g. --backend native vllm"
        ),
    )
    parser.add_argument(
        "--model", required=True,
        help="Model config name (filename without .yaml in configs/models/)",
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=[1],
        help="GPU counts to test. E.g. --gpus 1 2",
    )
    parser.add_argument(
        "--quantize", choices=["none", "int8", "int4"], default="none",
        help="Quantization mode (native backend only)",
    )
    parser.add_argument(
        "--flash-attn", action="store_true",
        help="Enable Flash Attention 2 (native backend)",
    )
    parser.add_argument(
        "--profiler", choices=["none", "pytorch", "nsight"], default="none",
        help="Profiler to use",
    )
    parser.add_argument(
        "--batch-sizes", type=str, default=None,
        help="Comma-separated batch sizes. E.g. --batch-sizes 1,2,4,8",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to main bench config YAML (default: configs/bench.yaml)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Also export results as CSV",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available model configs and exit",
    )
    parser.add_argument(
        "--list-backends", action="store_true",
        help="List available backend configs and exit",
    )
    args = parser.parse_args()

    # List commands
    if args.list_models:
        print(f"\n  Available models: {list_available_models()}\n")
        return
    if args.list_backends:
        print(f"\n  Available backends: {list_available_backends()}\n")
        return

    # Resolve backends
    backends = ALL_BACKENDS if "all" in args.backend else args.backend

    # Resolve batch sizes
    batch_sizes = None
    if args.batch_sizes:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    # Detect GPUs
    available = torch.cuda.device_count()
    print(f"\n  Detected {available} GPU(s)")

    # Run benchmarks for each GPU count
    all_results = []
    for num_gpus in args.gpus:
        if num_gpus > available:
            print(f"  ⚠️  Skipping {num_gpus}-GPU: only {available} available.")
            continue

        runner = BenchmarkRunner(
            model_name=args.model,
            backend_names=backends,
            profiler_name=args.profiler if args.profiler != "none" else None,
            num_gpus=num_gpus,
            quantize=args.quantize,
            flash_attn=args.flash_attn,
            batch_sizes=batch_sizes,
            bench_config_path=args.config,
        )

        results = runner.run()
        all_results.extend(results)

    if not all_results:
        print("  No results collected.")
        return

    # Print summary
    print_summary(all_results)

    # Save results
    output_path = args.output or f"results/{args.model}_{'_'.join(backends)}.json"
    save_json(all_results, output_path)

    if args.csv:
        csv_path = output_path.replace(".json", ".csv")
        save_csv(all_results, csv_path)

    # Nsight command hints
    if args.profiler == "nsight":
        from src.profilers.nsight_profiler import NsightProfiler
        from src.utils.config import load_profiler_config
        nsight_cfg = load_profiler_config("nsight")
        nsight = NsightProfiler(nsight_cfg)
        script_args = f"--backend {' '.join(backends)} --model {args.model} --gpus {args.gpus[0]}"
        nsight.print_commands("scripts/run_bench.py", script_args)


if __name__ == "__main__":
    main()
