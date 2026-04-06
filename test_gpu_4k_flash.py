"""
Comprehensive Benchmark for Param2-17B-A2.4B-Thinking
======================================================
- 1 GPU vs 2 GPU (PCIe)
- Batch size sweep 1 → OOM
- 4 context tiers tuned for Thinking model (max_new_tokens = 1024+)
- KV cache fully cleared between every run
- Reasoning vs answer token split (via parse_model_output)
- TTFT, decode throughput, e2e latency, std dev
- GPU utilisation %, memory allocated, memory delta per run
- MoE expert cache reset between runs
- Flash Attention 2 + INT8 optional flags

Usage:
python benchmark_param2.py --gpus 1 2                        # FP16 baseline
python benchmark_param2.py --gpus 1 2 --flash_attn           # FP16 + Flash Attention 2
python benchmark_param2.py --gpus 1 2 --int8                 # INT8
python benchmark_param2.py --gpus 1 2 --int8 --flash_attn    # INT8 + Flash Attention 2
"""

import sys, time, argparse, json, gc, os, threading, statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.append("/home/kishoryd/LLM_Bench/data/Param2-17B")
from parsers import parse_model_output

MODEL_PATH    = "/home/kishoryd/LLM_Bench/data/Param2-17B"
MAX_CTX       = 4096
WARMUP_RUNS   = 1
MEASURE_RUNS  = 3
TEMPERATURE   = 0.7
TOP_K         = 50
TOP_P         = 0.9
BATCH_SIZES   = [1, 2, 4, 8, 16, 32]

# Context tiers revised for Thinking model
# Input kept smaller to leave room for reasoning tokens
CONTEXT_TIERS = [
    {"label": "short  (~256 input)",  "target_input": 220,  "max_new_tokens": 1024},
    {"label": "medium (~512 input)",  "target_input": 480,  "max_new_tokens": 1024},
    {"label": "long   (~1024 input)", "target_input": 980,  "max_new_tokens": 1500},
    {"label": "near   (~2048 input)", "target_input": 1980, "max_new_tokens": 1500},
]


# ── Cache management ──────────────────────────────────────────────────────────

def clear_kv_cache(model):
    """
    Aggressively clear all KV cache and expert cache from model layers.
    Handles standard HF cache, MoE expert cache, and trust_remote_code customs.
    """
    for module in model.modules():
        for attr in ["past_key_values", "expert_cache", "_cache",
                     "kv_cache", "cache", "key_cache", "value_cache"]:
            if hasattr(module, attr):
                setattr(module, attr, None)

    # Clear PyTorch CUDA allocator
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


def memory_allocated_gb(device_id: int = 0) -> float:
    return round(torch.cuda.memory_allocated(device_id) / 1024**3, 3)


# ── GPU utilisation sampler ───────────────────────────────────────────────────

class GPUStatsSampler:
    def __init__(self, device_ids):
        self.device_ids = device_ids
        self._util      = {i: [] for i in device_ids}
        self._stop      = threading.Event()
        self._thread    = None

    def start(self):
        self._stop.clear()
        self._util   = {i: [] for i in self.device_ids}
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join()
        return {
            f"GPU:{i}": {
                "avg_util_pct": round(statistics.mean(v), 1) if v else 0,
                "max_util_pct": round(max(v), 1)             if v else 0,
            }
            for i, v in self._util.items()
        }

    def _poll(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.device_ids}
            while not self._stop.is_set():
                for i, h in handles.items():
                    self._util[i].append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                time.sleep(0.1)
        except Exception:
            while not self._stop.is_set():
                for i in self.device_ids:
                    self._util[i].append(0)
                time.sleep(0.1)


# ── GPU power + temperature ───────────────────────────────────────────────────

def gpu_hw_snapshot(device_ids: list) -> dict:
    snap = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in device_ids:
            h    = pynvml.nvmlDeviceGetHandleByIndex(i)
            snap[f"GPU:{i}"] = {
                "allocated_GB":  round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                "reserved_GB":   round(torch.cuda.memory_reserved(i)  / 1024**3, 2),
                "total_GB":      round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
                "power_W":       round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1),
                "temp_C":        pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            }
    except Exception:
        for i in device_ids:
            snap[f"GPU:{i}"] = {
                "allocated_GB":  round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                "reserved_GB":   round(torch.cuda.memory_reserved(i)  / 1024**3, 2),
                "total_GB":      round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
                "power_W":       0,
                "temp_C":        0,
            }
    return snap


# ── Prompt builder ────────────────────────────────────────────────────────────

def _repeat_to_tokens(tokenizer, seed: str, target: int) -> str:
    chunk  = seed.strip() + " "
    text   = chunk
    while len(tokenizer.encode(text)) < target:
        text += chunk
    return tokenizer.decode(tokenizer.encode(text)[:target], skip_special_tokens=True)


def build_context_tiers(tokenizer) -> list:
    p_bharatgen = (
        "The BharatGen initiative is a government-backed research programme aimed at developing "
        "large-scale foundational AI models rooted in Indian languages, culture, and knowledge systems. "
        "It seeks to make AI accessible to over a billion people by training models on diverse Indic "
        "language corpora spanning Hindi, Tamil, Telugu, Bengali, Kannada, Malayalam, Marathi, Gujarati, "
        "and many other regional languages. The programme emphasises data sovereignty, ethical AI, and "
        "the democratisation of technology for underserved communities across rural and urban India. "
    )
    p_moe = (
        "Mixture-of-Experts (MoE) architectures improve the efficiency of large language models by "
        "activating only a subset of parameters for each input token. A gating network selects the "
        "top-k expert feed-forward networks to process each token, allowing total parameter count to "
        "scale without a proportional increase in compute per forward pass. Load balancing losses are "
        "added during training to prevent expert collapse. "
    )
    p_history = (
        "The history of artificial intelligence spans decades of research, beginning with the Dartmouth "
        "Conference of 1956. Early symbolic AI systems used hand-crafted rules. The field experienced "
        "AI winters before the rise of neural networks in the 1980s. The deep learning revolution of "
        "the 2010s, enabled by GPUs and large datasets, led to breakthroughs in vision, speech, and "
        "natural language processing. "
    )

    seeds = [p_bharatgen, p_moe, p_history, p_bharatgen + p_moe]
    questions = [
        "Summarise the BharatGen mission in 3 detailed bullet points.",
        "Explain how MoE models balance efficiency and scale with introduction, mechanism, and trade-offs.",
        "Write a detailed numbered timeline of major AI milestones from 1956 to the deep learning era.",
        "Write a detailed paragraph connecting BharatGen's goals with MoE architecture advantages.",
    ]

    tiers = []
    for cfg, seed, question in zip(CONTEXT_TIERS, seeds, questions):
        body = _repeat_to_tokens(tokenizer, seed, cfg["target_input"])
        # Verify total fits within MAX_CTX
        test_conv = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": f"{body}\n\n{question}"},
        ]
        input_len = len(tokenizer.apply_chat_template(
            test_conv, tokenize=True, add_generation_prompt=True
        ))
        max_new = min(cfg["max_new_tokens"], MAX_CTX - input_len - 10)

        tiers.append({
            **cfg,
            "prompt":       f"{body}\n\n{question}",
            "input_len":    input_len,
            "max_new_tokens": max_new,
            "safe":         (input_len + max_new) <= MAX_CTX,
        })

    return tiers


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(num_gpus: int, use_int8: bool = False, use_flash_attn: bool = False):
    print(f"\n{'='*66}")
    print(f"  Loading model | GPUs={num_gpus} | INT8={use_int8} | FlashAttn2={use_flash_attn}")
    print(f"{'='*66}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=False, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched decoder-only inference

    kwargs = {
        "trust_remote_code": True,
        "device_map":        "auto",
        "local_files_only":  True,
    }

    if use_int8:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=["lm_head", "gate", "router"],  # protect MoE routing
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    t0    = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    model.eval()
    load_time = time.perf_counter() - t0

    device_ids = list(range(num_gpus))
    print(f"  Loaded in {load_time:.1f}s")
    for i in device_ids:
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU:{i}  {alloc:.2f} / {total:.2f} GB")

    return model, tokenizer, device_ids


# ── Single batched inference run ──────────────────────────────────────────────

def run_batch(model, tokenizer, prompt: str, batch_size: int,
              max_new_tokens: int, device_ids: list) -> dict:

    conversations = [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]] * batch_size

    # Tokenise and left-pad
    encoded = [
        tokenizer.apply_chat_template(c, return_tensors="pt", add_generation_prompt=True)[0]
        for c in conversations
    ]
    max_len   = max(e.shape[0] for e in encoded)
    padded    = torch.stack([
        torch.cat([
            torch.full((max_len - e.shape[0],), tokenizer.pad_token_id, dtype=torch.long),
            e
        ]) for e in encoded
    ])
    attn_mask = (padded != tokenizer.pad_token_id).long()

    first_device = next(model.parameters()).device
    padded    = padded.to(first_device)
    attn_mask = attn_mask.to(first_device)
    input_len = padded.shape[1]

    mem_before = memory_allocated_gb(device_ids[0])

    # ── TTFT: prefill only ────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        _ = model(input_ids=padded, attention_mask=attn_mask)

    torch.cuda.synchronize()
    ttft_s = time.perf_counter() - t0

    # Clear prefill cache before full generation
    clear_kv_cache(model)

    # ── Full generation ───────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t_gen = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            padded,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=TOP_K,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    torch.cuda.synchronize()
    e2e_s = time.perf_counter() - t_gen

    mem_after = memory_allocated_gb(device_ids[0])

    # ── Parse reasoning vs answer (Thinking model) ────────────────────────────
    generated_tokens = output[0][input_len:]
    generated_text   = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    parsed           = parse_model_output(generated_text)

    reasoning_tokens = len(tokenizer.encode(parsed.get("reasoning", "") or ""))
    answer_tokens    = len(tokenizer.encode(parsed.get("final_answer", "") or ""))
    output_tokens    = output.shape[1] - input_len

    total_new    = output_tokens * batch_size
    decode_s     = max(e2e_s - ttft_s, 1e-6)
    tok_per_s    = total_new / decode_s

    # ── Cleanup — critical ────────────────────────────────────────────────────
    del output, padded, attn_mask, generated_tokens
    clear_kv_cache(model)

    return {
        "input_len":       input_len,
        "output_tokens":   output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "answer_tokens":   answer_tokens,
        "ttft_s":          round(ttft_s,   4),
        "decode_s":        round(decode_s, 4),
        "e2e_s":           round(e2e_s,    4),
        "tok_per_s":       round(tok_per_s, 2),
        "mem_before_GB":   mem_before,
        "mem_after_GB":    mem_after,
        "mem_delta_GB":    round(mem_after - mem_before, 3),
    }


# ── Batch sweep for one context tier ─────────────────────────────────────────

def sweep_batch_sizes(model, tokenizer, tier: dict, device_ids: list) -> list:
    label          = tier["label"]
    prompt         = tier["prompt"]
    max_new_tokens = tier["max_new_tokens"]
    results        = []
    sampler        = GPUStatsSampler(device_ids)

    status = "✅" if tier["safe"] else "⚠️  tight context"
    print(f"\n  ── {label}  input={tier['input_len']} tok  output_budget={max_new_tokens}  {status}")

    for bs in BATCH_SIZES:
        print(f"\n     batch_size={bs}")

        # ── Warmup ────────────────────────────────────────────────────────────
        try:
            for _ in range(WARMUP_RUNS):
                clear_kv_cache(model)                   # clean before warmup
                run_batch(model, tokenizer, prompt, bs, max_new_tokens, device_ids)
                clear_kv_cache(model)                   # clean after warmup
        except torch.cuda.OutOfMemoryError:
            print(f"     ⛔ OOM during warmup at batch_size={bs} — stopping sweep.")
            results.append({"batch_size": bs, "oom": True, "phase": "warmup"})
            clear_kv_cache(model)
            break

        # ── Measure runs ──────────────────────────────────────────────────────
        run_records = []
        oom_hit     = False

        for r_idx in range(MEASURE_RUNS):
            try:
                clear_kv_cache(model)                   # clean slate before each run
                sampler.start()
                r         = run_batch(model, tokenizer, prompt, bs, max_new_tokens, device_ids)
                gpu_stats = sampler.stop()
                hw_snap   = gpu_hw_snapshot(device_ids)
                clear_kv_cache(model)                   # clean after each run

                r["gpu_util"] = gpu_stats
                r["hw"]       = hw_snap
                run_records.append(r)

                mem_delta_str = f"{r['mem_delta_GB']:+.3f} GB"
                print(
                    f"       run {r_idx+1}: "
                    f"TTFT={r['ttft_s']:.3f}s  "
                    f"e2e={r['e2e_s']:.3f}s  "
                    f"throughput={r['tok_per_s']:.1f} tok/s  "
                    f"output={r['output_tokens']} tok "
                    f"(reasoning={r['reasoning_tokens']} / answer={r['answer_tokens']})  "
                    f"mem_delta={mem_delta_str}"
                )

            except torch.cuda.OutOfMemoryError:
                sampler.stop()
                clear_kv_cache(model)
                print(f"       ⛔ OOM at run {r_idx+1}, batch_size={bs} — stopping sweep.")
                results.append({"batch_size": bs, "oom": True, "phase": f"measure_run_{r_idx+1}"})
                oom_hit = True
                break

        if oom_hit:
            break

        # ── Aggregate ─────────────────────────────────────────────────────────
        n = len(run_records)

        def avg(key): return round(sum(r[key] for r in run_records) / n, 4)
        def std(key):
            vals = [r[key] for r in run_records]
            return round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)

        agg = {
            "batch_size":           bs,
            "oom":                  False,
            "input_tokens":         run_records[0]["input_len"],
            "output_tokens_avg":    avg("output_tokens"),
            "reasoning_tokens_avg": avg("reasoning_tokens"),
            "answer_tokens_avg":    avg("answer_tokens"),
            "ttft_avg_s":           avg("ttft_s"),
            "ttft_std_s":           std("ttft_s"),
            "e2e_avg_s":            avg("e2e_s"),
            "e2e_std_s":            std("e2e_s"),
            "decode_avg_s":         avg("decode_s"),
            "tok_per_s_avg":        avg("tok_per_s"),
            "tok_per_s_std":        std("tok_per_s"),
            "mem_delta_avg_GB":     avg("mem_delta_GB"),
            "gpu_memory":           run_records[-1]["hw"],
            "gpu_util":             run_records[-1]["gpu_util"],
        }

        print(
            f"     AVG → "
            f"TTFT={agg['ttft_avg_s']:.3f}s (±{agg['ttft_std_s']:.3f})  "
            f"e2e={agg['e2e_avg_s']:.3f}s (±{agg['e2e_std_s']:.3f})  "
            f"throughput={agg['tok_per_s_avg']:.1f} tok/s (±{agg['tok_per_s_std']:.1f})  "
            f"mem_delta={agg['mem_delta_avg_GB']:+.3f} GB"
        )

        results.append(agg)

    return results


# ── Top-level benchmark ───────────────────────────────────────────────────────

def benchmark(num_gpus: int, use_int8: bool, use_flash_attn: bool) -> dict:
    model, tokenizer, device_ids = load_model(num_gpus, use_int8, use_flash_attn)
    tiers   = build_context_tiers(tokenizer)
    results = {
        "num_gpus":      num_gpus,
        "use_int8":      use_int8,
        "use_flash_attn": use_flash_attn,
        "tiers":         [],
    }

    for tier in tiers:
        tier_result = {
            "label":          tier["label"],
            "input_tokens":   tier["input_len"],
            "max_new_tokens": tier["max_new_tokens"],
            "context_safe":   tier["safe"],
            "batch_sweeps":   sweep_batch_sizes(model, tokenizer, tier, device_ids),
        }
        results["tiers"].append(tier_result)
        # Full cache clear between tiers
        clear_kv_cache(model)

    results["gpu_memory_final"] = gpu_hw_snapshot(device_ids)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(all_results: list):
    W = 100
    print("\n")
    print("╔" + "═"*W + "╗")
    print("║" + "  BENCHMARK SUMMARY — Param2-17B Thinking  (A100 80GB PCIe, max ctx 4096)".center(W) + "║")
    print("╚" + "═"*W + "╝")

    for res in all_results:
        n     = res["num_gpus"]
        quant = "INT8" if res["use_int8"] else "FP16"
        attn  = "+FlashAttn2" if res["use_flash_attn"] else ""
        print(f"\n{'─'*W}")
        print(f"  {n} GPU{'s' if n>1 else ' '} | {quant}{attn}")
        print(f"{'─'*W}")

        for tier in res["tiers"]:
            print(f"\n  📊 {tier['label']}  (input={tier['input_tokens']} tok, output_budget={tier['max_new_tokens']} tok)")
            print(
                f"  {'Batch':>6} {'TTFT':>10} {'±':>7} {'E2E':>10} {'±':>7} "
                f"{'Tput(t/s)':>11} {'±':>7} {'Reasoning':>11} {'Answer':>9} "
                f"{'MemΔ':>8} {'Mem(GB)':>9} {'Util%':>7} {'Pwr(W)':>8} {'Temp°C':>8}"
            )
            print("  " + "─"*(W-2))

            for sw in tier["batch_sweeps"]:
                bs = sw["batch_size"]
                if sw["oom"]:
                    print(f"  {bs:>6}  ⛔ OOM ({sw['phase']})")
                    continue

                mem0  = sw["gpu_memory"].get("GPU:0", {}).get("allocated_GB", 0)
                util0 = sw["gpu_util"].get("GPU:0", {}).get("avg_util_pct", 0)
                pwr0  = sw["gpu_memory"].get("GPU:0", {}).get("power_W", 0)
                tmp0  = sw["gpu_memory"].get("GPU:0", {}).get("temp_C", 0)

                print(
                    f"  {bs:>6} "
                    f"{sw['ttft_avg_s']:>9.3f}s "
                    f"±{sw['ttft_std_s']:>5.3f} "
                    f"{sw['e2e_avg_s']:>9.3f}s "
                    f"±{sw['e2e_std_s']:>5.3f} "
                    f"{sw['tok_per_s_avg']:>10.1f} "
                    f"±{sw['tok_per_s_std']:>5.1f} "
                    f"{sw['reasoning_tokens_avg']:>11.1f} "
                    f"{sw['answer_tokens_avg']:>9.1f} "
                    f"{sw['mem_delta_avg_GB']:>+8.3f} "
                    f"{mem0:>8.2f}GB "
                    f"{util0:>6.1f}% "
                    f"{pwr0:>7.1f}W "
                    f"{tmp0:>7}°C"
                )

    print(f"\n{'─'*W}")
    print("  💾 Final GPU State:")
    for res in all_results:
        n = res["num_gpus"]
        print(f"\n    {n} GPU:")
        for gpu, mem in res["gpu_memory_final"].items():
            print(
                f"      {gpu}: {mem['allocated_GB']}GB allocated / {mem['total_GB']}GB total  "
                f"| {mem['power_W']}W | {mem['temp_C']}°C"
            )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Param2-17B Thinking — comprehensive benchmark")
    parser.add_argument("--gpus",       nargs="+", type=int, default=[1, 2])
    parser.add_argument("--output",     default="benchmark_results.json")
    parser.add_argument("--int8",       action="store_true", help="Load model in INT8")
    parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention 2")
    args = parser.parse_args()

    available = torch.cuda.device_count()
    print(f"\n  Detected {available} GPU(s) on this machine.")
    print(f"  INT8={args.int8}  FlashAttn2={args.flash_attn}")

    all_results = []
    for n in args.gpus:
        if n > available:
            print(f"  ⚠️  Skipping {n}-GPU: only {available} available.")
            continue
        all_results.append(benchmark(n, args.int8, args.flash_attn))

    if all_results:
        print_summary(all_results)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  ✅ Saved to {args.output}\n")


if __name__ == "__main__":
    main()
