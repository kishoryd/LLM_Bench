"""
Throughput & Latency Benchmark for Param2-17B-A2.4B-Thinking
Tests: 1 GPU vs 2 GPU configurations (no model download — uses local path)

Metrics captured:
  - Time to First Token (TTFT)
  - Total generation time
  - Tokens/sec (throughput)
  - Memory usage per GPU

Usage:
  python benchmark_param2.py --gpus 1        # test on 1 GPU
  python benchmark_param2.py --gpus 2        # test on 2 GPUs
  python benchmark_param2.py --gpus 1 2      # test both and compare
"""

import sys
import time
import argparse
import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append("/home/kishoryd/LLM_Bench/data/Param2-17B")

MODEL_PATH = "/home/kishoryd/LLM_Bench/data/Param2-17B"  # local path, no download

# ── Benchmark config ──────────────────────────────────────────────────────────
PROMPTS = [
    "What is the BharatGen Mission?",
    "Explain the architecture of a mixture-of-experts language model.",
    "Summarize the history of artificial intelligence in 200 words.",
    "Write a Python function to compute the nth Fibonacci number iteratively.",
    "What are the key differences between supervised and unsupervised learning?",
]

GEN_CONFIGS = [
    {"max_new_tokens": 128,  "label": "short  (128 tok)"},
    {"max_new_tokens": 256,  "label": "medium (256 tok)"},
    {"max_new_tokens": 512,  "label": "long   (512 tok)"},
]

WARMUP_RUNS   = 1   # runs discarded before measuring
MEASURE_RUNS  = 3   # runs averaged for each config
TEMPERATURE   = 0.7
TOP_K         = 50
TOP_P         = 0.9
# ─────────────────────────────────────────────────────────────────────────────


def gpu_memory_summary(device_ids: list[int]) -> dict:
    info = {}
    for i in device_ids:
        alloc  = torch.cuda.memory_allocated(i)  / 1024**3
        reserv = torch.cuda.memory_reserved(i)   / 1024**3
        total  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        info[f"GPU:{i}"] = {
            "allocated_GB":  round(alloc,  2),
            "reserved_GB":   round(reserv, 2),
            "total_GB":      round(total,  2),
        }
    return info


def load_model(num_gpus: int):
    """Load model onto `num_gpus` GPUs using device_map='auto'."""
    print(f"\n{'='*60}")
    print(f"  Loading model on {num_gpus} GPU(s) ...")
    print(f"{'='*60}")

    # Restrict visible GPUs so device_map='auto' spreads only across them
    visible = ",".join(str(i) for i in range(num_gpus))
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = visible

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=False, local_files_only=True
    )

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",          # spreads across all visible GPUs
        torch_dtype=torch.float16,  # fp16 — needed for multi-GPU shard
        local_files_only=True,      # never re-download
    )
    load_time = time.perf_counter() - t0
    model.eval()

    device_ids = list(range(num_gpus))
    mem = gpu_memory_summary(device_ids)
    print(f"  Model loaded in {load_time:.1f}s")
    for k, v in mem.items():
        print(f"  {k}: {v['allocated_GB']:.2f} GB allocated / {v['total_GB']:.2f} GB total")

    return model, tokenizer, device_ids


def run_single(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    """Run one inference and return timing + token count."""
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        conversation=conversation,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    # Move to the first parameter's device (handles multi-GPU transparently)
    first_device = next(model.parameters()).device
    inputs = inputs.to(first_device)

    input_len = inputs.shape[-1]

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=TOP_K,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,   # cache ON for realistic inference benchmark
        )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    output_len   = output.shape[-1] - input_len
    elapsed      = t_end - t_start
    tokens_per_s = output_len / elapsed if elapsed > 0 else 0

    return {
        "input_tokens":    input_len,
        "output_tokens":   output_len,
        "elapsed_s":       round(elapsed, 4),
        "tokens_per_sec":  round(tokens_per_s, 2),
    }


def benchmark(num_gpus: int) -> dict:
    model, tokenizer, device_ids = load_model(num_gpus)
    results = {"num_gpus": num_gpus, "configs": []}

    for gen_cfg in GEN_CONFIGS:
        max_new = gen_cfg["max_new_tokens"]
        label   = gen_cfg["label"]
        print(f"\n  ── {label} {'─'*40}")

        cfg_result = {
            "label":          label,
            "max_new_tokens": max_new,
            "runs":           [],
        }

        for i, prompt in enumerate(PROMPTS):
            run_times   = []
            run_tok_s   = []
            run_out_tok = []

            # Warmup
            for _ in range(WARMUP_RUNS):
                run_single(model, tokenizer, prompt, max_new)

            # Measure
            for _ in range(MEASURE_RUNS):
                r = run_single(model, tokenizer, prompt, max_new)
                run_times.append(r["elapsed_s"])
                run_tok_s.append(r["tokens_per_sec"])
                run_out_tok.append(r["output_tokens"])

            avg_time  = round(sum(run_times)   / MEASURE_RUNS, 4)
            avg_tok_s = round(sum(run_tok_s)   / MEASURE_RUNS, 2)
            avg_out   = round(sum(run_out_tok) / MEASURE_RUNS, 1)

            print(f"    [{i+1}/{len(PROMPTS)}] prompt={prompt[:40]!r}...")
            print(f"          avg latency={avg_time:.3f}s  |  throughput={avg_tok_s:.1f} tok/s  |  output≈{avg_out:.0f} tok")

            cfg_result["runs"].append({
                "prompt_snippet":  prompt[:60],
                "avg_latency_s":   avg_time,
                "avg_tok_per_s":   avg_tok_s,
                "avg_output_toks": avg_out,
            })

        results["configs"].append(cfg_result)

    # Final GPU memory snapshot
    results["gpu_memory"] = gpu_memory_summary(device_ids)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_summary(all_results: list[dict]):
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + "  BENCHMARK SUMMARY".center(70) + "║")
    print("╚" + "═"*70 + "╝")

    for gen_cfg in GEN_CONFIGS:
        label = gen_cfg["label"]
        print(f"\n  📊 {label}")
        print(f"  {'GPU Config':<12} {'Avg Latency':>14} {'Avg Throughput':>16} {'Speedup':>10}")
        print(f"  {'─'*12} {'─'*14} {'─'*16} {'─'*10}")

        base_lat = None
        for res in all_results:
            cfg = next(c for c in res["configs"] if c["label"] == label)
            runs = cfg["runs"]
            avg_lat  = sum(r["avg_latency_s"] for r in runs) / len(runs)
            avg_toks = sum(r["avg_tok_per_s"] for r in runs) / len(runs)

            if base_lat is None:
                base_lat = avg_lat
                speedup  = "baseline"
            else:
                speedup = f"{base_lat / avg_lat:.2f}x"

            gpu_label = f"{res['num_gpus']} GPU{'s' if res['num_gpus'] > 1 else ' '}"
            print(f"  {gpu_label:<12} {avg_lat:>12.3f}s {avg_toks:>14.1f} t/s {speedup:>10}")

    print("\n  💾 GPU Memory at end of run:")
    for res in all_results:
        print(f"\n    {res['num_gpus']} GPU config:")
        for gpu, mem in res["gpu_memory"].items():
            print(f"      {gpu}: {mem['allocated_GB']} GB allocated / {mem['total_GB']} GB total")


def main():
    parser = argparse.ArgumentParser(description="Param2-17B GPU benchmark")
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=[1, 2],
        help="Number of GPUs to test, e.g. --gpus 1 2"
    )
    parser.add_argument(
        "--output", default="benchmark_results.json",
        help="Path to save JSON results"
    )
    args = parser.parse_args()

    available = torch.cuda.device_count()
    print(f"\n  Detected {available} GPU(s) on this machine.")

    all_results = []
    for n in args.gpus:
        if n > available:
            print(f"  ⚠️  Skipping {n}-GPU test: only {available} GPU(s) available.")
            continue
        result = benchmark(n)
        all_results.append(result)

    if all_results:
        print_summary(all_results)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  ✅ Full results saved to {args.output}\n")


if __name__ == "__main__":
    main()
