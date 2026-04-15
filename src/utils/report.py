"""
Reporting: pretty-print summary tables, JSON export, CSV export.
"""

import json
import csv
import os
from pathlib import Path


def print_summary(all_results: list[dict]):
    """Print a formatted summary table to stdout."""
    print("\n")
    print("╔" + "═" * 94 + "╗")
    print("║" + "  LLM BENCH — BENCHMARK SUMMARY".center(94) + "║")
    print("╚" + "═" * 94 + "╝")

    for res in all_results:
        backend = res.get("backend", "?")
        model = res.get("model", "?")
        n = res.get("num_gpus", "?")
        cfg = res.get("load_config", {})

        print(f"\n{'─' * 94}")
        print(f"  Backend: {backend}  |  Model: {model}  |  GPUs: {n}")
        print(
            f"  attn={cfg.get('attn_impl', '?')}  "
            f"dtype={cfg.get('dtype', '?')}  "
            f"quantize={cfg.get('quantize', '?')}  "
            f"load_time={cfg.get('load_time_s', '?')}s"
        )
        print(f"{'─' * 94}")

        for tier in res.get("tiers", []):
            print(f"\n  📊 {tier['label']}")
            print(
                f"  {'Batch':>6} {'Input':>8} {'TTFT (s)':>12} {'E2E (s)':>12} "
                f"{'±StdDev':>9} {'Throughput':>13} {'GPU Mem':>10} {'Util%':>7}"
            )
            print(
                f"  {'─' * 6} {'─' * 8} {'─' * 12} {'─' * 12} "
                f"{'─' * 9} {'─' * 13} {'─' * 10} {'─' * 7}"
            )

            for sw in tier.get("batch_sweeps", []):
                bs = sw["batch_size"]
                if sw.get("oom"):
                    print(f"  {bs:>6}  {'⛔ OOM':>60}")
                    continue

                mem0 = sw.get("gpu_memory", {}).get("GPU:0", {}).get("allocated_GB", 0)
                util0 = sw.get("gpu_util", {}).get("GPU:0", {}).get("avg_util_pct", 0)

                print(
                    f"  {bs:>6} "
                    f"{sw.get('input_tokens', 0):>8} "
                    f"{sw.get('ttft_avg_s', 0):>11.3f}s "
                    f"{sw.get('e2e_avg_s', 0):>11.3f}s "
                    f"±{sw.get('e2e_std_s', 0):>7.3f} "
                    f"{sw.get('tok_per_s_avg', 0):>11.1f} t/s "
                    f"{mem0:>8.2f} GB "
                    f"{util0:>6.1f}%"
                )

    print(f"\n{'─' * 94}\n")


def save_json(all_results: list[dict], output_path: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✅ JSON saved to {output_path}")


def save_csv(all_results: list[dict], output_path: str):
    """Flatten results and save to CSV for spreadsheet analysis."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = []
    for res in all_results:
        base = {
            "backend": res.get("backend", ""),
            "model": res.get("model", ""),
            "num_gpus": res.get("num_gpus", ""),
            "quantize": res.get("load_config", {}).get("quantize", ""),
            "dtype": res.get("load_config", {}).get("dtype", ""),
            "attn_impl": res.get("load_config", {}).get("attn_impl", ""),
        }
        for tier in res.get("tiers", []):
            for sw in tier.get("batch_sweeps", []):
                row = {
                    **base,
                    "tier": tier["label"],
                    "batch_size": sw["batch_size"],
                    "oom": sw.get("oom", False),
                }
                if not sw.get("oom"):
                    row.update({
                        "input_tokens": sw.get("input_tokens", 0),
                        "ttft_avg_s": sw.get("ttft_avg_s", 0),
                        "e2e_avg_s": sw.get("e2e_avg_s", 0),
                        "tok_per_s_avg": sw.get("tok_per_s_avg", 0),
                        "gpu_mem_GB": sw.get("gpu_memory", {}).get("GPU:0", {}).get("allocated_GB", 0),
                    })
                rows.append(row)

    if not rows:
        print("  ⚠️  No data to write to CSV")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✅ CSV saved to {output_path}")
