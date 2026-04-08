"""
Comprehensive Benchmark for Param2-17B-A2.4B-Thinking  [PROFILED VERSION]
==========================================================================
Adds full profiling on top of the original benchmark:

  Timing / bottleneck profiling
    - cProfile over the entire run  →  benchmark_profile.prof  (view with snakeviz)
    - torch.profiler trace          →  ./torch_trace/           (view in chrome://tracing)
    - Per-phase wall-clock timers   →  printed + saved in JSON

  Memory / VRAM profiling
    - torch.cuda.memory_stats() snapshot at key checkpoints
    - Peak allocated / reserved per GPU printed per batch
    - VRAM timeline CSV (polled every 0.5 s)              →  vram_timeline.csv

  GPU utilisation & efficiency
    - pynvml utilisation % polled at 100 ms (already in original)
    - SM occupancy, memory BW utilisation via pynvml (if available)
    - Per-batch efficiency table in summary

Usage (mirrors original):
  python benchmark_param2_profiled.py --gpus 1
  python benchmark_param2_profiled.py --gpus 1 --quantize int4
  python benchmark_param2_profiled.py --gpus 1 --flash-attn --quantize int8

Extra flags:
  --torch-profiler     Enable torch.profiler trace (adds ~10 % overhead)
  --cprofile           Wrap entire script in cProfile (adds ~2 % overhead)
  --vram-poll-hz N     VRAM polling frequency in Hz (default 2)

Output artefacts:
  benchmark_results_profiled.json   full metrics
  benchmark_profile.prof            cProfile dump  (snakeviz benchmark_profile.prof)
  vram_timeline.csv                 VRAM over time
  torch_trace/                      torch.profiler traces (--torch-profiler only)
"""

import sys, time, argparse, json, gc, os, threading, statistics, csv, cProfile, pstats, io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from contextlib import contextmanager
from collections import defaultdict

sys.path.append("/home/kishor/LLM_Bench/data/Param2-17B")

MODEL_PATH   = "/home/kishor/LLM_Bench/data/Param2-17B"
MAX_CTX      = 4096
WARMUP_RUNS  = 1
MEASURE_RUNS = 3
TEMPERATURE  = 0.7
TOP_K        = 50
TOP_P        = 0.9

BATCH_SIZES  = [1, 2, 4, 8, 16, 32]

# ── GPU architecture detection ───────────────────────────────────────────────

ARCH_MAP = {
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
    (7, 0): "Volta",
    (7, 5): "Turing",
}
FA2_SUPPORTED_ARCHS  = {"Ampere", "Ada Lovelace", "Hopper"}
BF16_SUPPORTED_ARCHS = {"Ampere", "Ada Lovelace", "Hopper"}


def detect_gpu_info(device_id: int = 0) -> dict:
    props   = torch.cuda.get_device_properties(device_id)
    cc      = (props.major, props.minor)
    arch    = ARCH_MAP.get(cc, f"Unknown (sm_{props.major}{props.minor})")
    vram_gb = round(props.total_memory / 1024**3, 1)
    return {
        "name": props.name, "arch": arch,
        "compute_cap": f"sm_{props.major}{props.minor}",
        "vram_gb": vram_gb,
        "fa2_supported":  arch in FA2_SUPPORTED_ARCHS,
        "bf16_supported": arch in BF16_SUPPORTED_ARCHS,
    }


def print_gpu_info(num_gpus: int):
    print(f"\n{'─'*64}")
    print(f"  GPU Detection")
    print(f"{'─'*64}")
    for i in range(num_gpus):
        info = detect_gpu_info(i)
        print(f"  GPU:{i}  {info['name']}")
        print(f"         Architecture : {info['arch']} ({info['compute_cap']})")
        print(f"         VRAM         : {info['vram_gb']} GB")
        print(f"         Flash Attn 2 : {'✅' if info['fa2_supported'] else '❌ need sm_80+'}")
        print(f"         bfloat16     : {'✅' if info['bf16_supported'] else '⚠️  → float16'}")
    print()


def auto_select_dtype(num_gpus: int) -> torch.dtype:
    all_bf16 = all(detect_gpu_info(i)["bf16_supported"] for i in range(num_gpus))
    dtype = torch.bfloat16 if all_bf16 else torch.float16
    print(f"  dtype auto-selected : {'bfloat16' if all_bf16 else 'float16'}")
    return dtype


# ── Phase timer ──────────────────────────────────────────────────────────────

class PhaseTimer:
    """Accumulates named wall-clock phases with CUDA sync."""
    def __init__(self):
        self._log: list = []

    @contextmanager
    def phase(self, name: str):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        self._log.append({"phase": name, "elapsed_s": round(elapsed, 4)})
        print(f"    ⏱  [{name}] {elapsed*1000:.1f} ms")

    def summary(self) -> list:
        return self._log


_global_timer = PhaseTimer()


# ── VRAM timeline sampler ────────────────────────────────────────────────────

class VRAMTimelineSampler:
    """Polls VRAM allocated/reserved on all GPUs and writes a CSV."""
    def __init__(self, device_ids: list, poll_hz: float = 2.0, path: str = "vram_timeline.csv"):
        self.device_ids = device_ids
        self.interval   = 1.0 / poll_hz
        self.path       = path
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True)
        self._rows: list = []

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        self._flush()

    def _run(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            row = {"t_s": round(time.perf_counter() - t0, 3)}
            for i in self.device_ids:
                row[f"gpu{i}_alloc_gb"] = round(torch.cuda.memory_allocated(i) / 1024**3, 3)
                row[f"gpu{i}_reserv_gb"] = round(torch.cuda.memory_reserved(i)  / 1024**3, 3)
            self._rows.append(row)
            time.sleep(self.interval)

    def _flush(self):
        if not self._rows:
            return
        fieldnames = list(self._rows[0].keys())
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self._rows)
        print(f"  💾 VRAM timeline → {self.path}  ({len(self._rows)} samples)")


# ── GPU utilisation sampler (100 ms poll, same as original + extra metrics) ──

class GPUStatsSampler:
    def __init__(self, device_ids):
        self.device_ids = device_ids
        self._util      = {i: [] for i in device_ids}
        self._mem_bw    = {i: [] for i in device_ids}   # memory-bandwidth util % (if pynvml v2)
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._stop.clear()
        self._util   = {i: [] for i in self.device_ids}
        self._mem_bw = {i: [] for i in self.device_ids}
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        result = {}
        for i in self.device_ids:
            v = self._util[i]
            m = self._mem_bw[i]
            result[f"GPU:{i}"] = {
                "avg_util_pct": round(statistics.mean(v), 1) if v else 0,
                "max_util_pct": round(max(v), 1)             if v else 0,
                "avg_mem_bw_pct": round(statistics.mean(m), 1) if m else None,
            }
        return result

    def _poll(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.device_ids}
            while not self._stop.is_set():
                for i, h in handles.items():
                    rates = pynvml.nvmlDeviceGetUtilizationRates(h)
                    self._util[i].append(rates.gpu)
                    self._mem_bw[i].append(rates.memory)
                time.sleep(0.1)
        except Exception:
            while not self._stop.is_set():
                for i in self.device_ids:
                    self._util[i].append(0)
                    self._mem_bw[i].append(0)
                time.sleep(0.1)


# ── Quantization config ───────────────────────────────────────────────────────

def build_quant_config(quantize: str, dtype: torch.dtype):
    if quantize == "int4":
        print("  Quantization        : INT4 (NF4, double-quant)")
        cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return cfg, None
    elif quantize == "int8":
        print("  Quantization        : INT8 (LLM.int8)")
        return BitsAndBytesConfig(load_in_8bit=True), None
    else:
        print("  Quantization        : None (full precision)")
        return None, dtype


# ── GPU memory snapshot (extended) ───────────────────────────────────────────

def gpu_memory_snapshot(device_ids: list) -> dict:
    snap = {}
    for i in device_ids:
        stats = torch.cuda.memory_stats(i)
        snap[f"GPU:{i}"] = {
            "allocated_GB":     round(torch.cuda.memory_allocated(i) / 1024**3, 3),
            "reserved_GB":      round(torch.cuda.memory_reserved(i)  / 1024**3, 3),
            "peak_allocated_GB": round(stats.get("allocated_bytes.all.peak", 0) / 1024**3, 3),
            "peak_reserved_GB":  round(stats.get("reserved_bytes.all.peak",  0) / 1024**3, 3),
            "num_alloc_retries": stats.get("num_alloc_retries", 0),
            "total_GB":         round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
        }
    return snap


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _repeat_to_tokens(tokenizer, seed: str, target: int) -> str:
    chunk = seed.strip() + " "
    text  = chunk
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

    tiers = [
        {"label": "short  (~256 tok)",      "target_input": 220,  "max_new_tokens": 128,
         "seed": p_bharatgen,
         "question": "Summarise the BharatGen mission in 3 bullet points."},
        {"label": "medium (~1024 tok)",     "target_input": 980,  "max_new_tokens": 256,
         "seed": p_moe,
         "question": "Explain how MoE models balance efficiency and scale with intro, mechanism, and trade-offs."},
        {"label": "long   (~2048 tok)",     "target_input": 1980, "max_new_tokens": 256,
         "seed": p_history,
         "question": "Write a numbered timeline of major AI milestones from 1956 to the deep learning era."},
        {"label": "near-limit (~3584 tok)", "target_input": 3500, "max_new_tokens": 128,
         "seed": p_bharatgen + p_moe,
         "question": "Write two sentences connecting BharatGen and MoE architectures."},
    ]

    for t in tiers:
        body = _repeat_to_tokens(tokenizer, t["seed"], t["target_input"])
        t["prompt"] = f"{body}\n\n{t['question']}"

    return tiers


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(num_gpus: int, use_flash_attn: bool, quantize: str, use_torch_profiler: bool):
    print(f"\n{'='*64}")
    print(f"  Loading model on {num_gpus} GPU(s) ...")
    print(f"{'='*64}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    print_gpu_info(num_gpus)
    dtype     = auto_select_dtype(num_gpus)
    gpu0_info = detect_gpu_info(0)
    attn_impl = None

    if use_flash_attn:
        if gpu0_info["fa2_supported"]:
            try:
                import flash_attn  # noqa
                attn_impl = "flash_attention_2"
                print("  Flash Attention 2   : ✅ Enabled")
            except ImportError:
                attn_impl = "sdpa"
                print("  Flash Attention 2   : ⚠️  not installed → SDPA")
        else:
            attn_impl = "sdpa"
            print(f"  Flash Attention 2   : ❌ {gpu0_info['arch']} not sm_80+ → SDPA")
    else:
        attn_impl = "sdpa"
        print("  Flash Attention 2   : Not requested → SDPA")

    quant_cfg, load_dtype = build_quant_config(quantize, dtype)
    effective_dtype = load_dtype if load_dtype is not None else dtype

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=False, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = dict(
        trust_remote_code=True, device_map="auto",
        local_files_only=True, attn_implementation=attn_impl,
    )
    if quant_cfg is not None:
        load_kwargs["quantization_config"] = quant_cfg
        if quantize == "int8":
            load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = effective_dtype

    print(f"\n  Loading with settings:")
    print(f"    attn_implementation = {attn_impl}")
    print(f"    torch_dtype         = {load_kwargs.get('torch_dtype', 'managed by BnB')}")
    print(f"    quantize            = {quantize}")
    print()

    # ── Profile model load separately ───────────────────────────────────────
    with _global_timer.phase("model_load"):
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)
    model.eval()

    device_ids = list(range(num_gpus))
    for i in device_ids:
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU:{i}  {alloc:.2f} / {total:.2f} GB  (after model load)")

    load_time = next(e["elapsed_s"] for e in _global_timer.summary() if e["phase"] == "model_load")
    return model, tokenizer, device_ids, {
        "attn_impl":   attn_impl,
        "dtype":       str(effective_dtype),
        "quantize":    quantize,
        "load_time_s": round(load_time, 2),
    }


# ── Single batched inference run (with phase timing) ─────────────────────────

def run_batch(model, tokenizer, prompt: str, batch_size: int,
              max_new_tokens: int, device_ids: list,
              use_torch_profiler: bool = False,
              torch_trace_dir: str = "./torch_trace") -> dict:

    conversations = [
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user",   "content": prompt}]
    ] * batch_size

    # ── Tokenisation ──────────────────────────────────────────────────────────
    with _global_timer.phase(f"tokenise_bs{batch_size}"):
        encoded = [
            tokenizer.apply_chat_template(c, return_tensors="pt", add_generation_prompt=True)[0]
            for c in conversations
        ]
        max_len = max(e.shape[0] for e in encoded)
        padded  = torch.stack([
            torch.cat([
                torch.full((max_len - e.shape[0],), tokenizer.pad_token_id, dtype=torch.long), e
            ])
            for e in encoded
        ])
        attn_mask = (padded != tokenizer.pad_token_id).long()

    first_device = next(model.parameters()).device
    padded    = padded.to(first_device)
    attn_mask = attn_mask.to(first_device)
    input_len = padded.shape[1]

    # ── Prefill / TTFT ────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()          # reset peak counter before prefill
    torch.cuda.synchronize()
    t_prefill_start = time.perf_counter()

    with torch.no_grad():
        _ = model(input_ids=padded, attention_mask=attn_mask)

    torch.cuda.synchronize()
    ttft_s = time.perf_counter() - t_prefill_start

    mem_after_prefill = gpu_memory_snapshot(device_ids)

    # ── Full generation (optionally under torch.profiler) ────────────────────
    torch.cuda.synchronize()
    t_gen_start = time.perf_counter()

    if use_torch_profiler:
        os.makedirs(torch_trace_dir, exist_ok=True)
        trace_name = f"bs{batch_size}_inp{input_len}"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                torch_trace_dir, worker_name=trace_name
            ),
        ) as prof:
            with torch.no_grad():
                output = model.generate(
                    padded, attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens, do_sample=True,
                    top_k=TOP_K, top_p=TOP_P, temperature=TEMPERATURE,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )

        # Print top-10 CUDA ops by self CUDA time
        print("\n    ── torch.profiler top-10 CUDA ops (self time) ──")
        key_avgs = prof.key_averages()
        top_ops  = sorted(key_avgs, key=lambda e: e.self_cuda_time_total, reverse=True)[:10]
        for op in top_ops:
            print(f"      {op.key:<50s}  {op.self_cuda_time_total/1000:8.2f} ms  "
                  f"CUDA mem: {op.cuda_memory_usage/1024**2:.1f} MB")
        print(f"    ── trace saved to {torch_trace_dir}/{trace_name}* ──\n")

    else:
        with torch.no_grad():
            output = model.generate(
                padded, attention_mask=attn_mask,
                max_new_tokens=max_new_tokens, do_sample=True,
                top_k=TOP_K, top_p=TOP_P, temperature=TEMPERATURE,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    e2e_s       = t_end - t_gen_start
    output_toks = output.shape[1] - input_len
    total_new   = output_toks * batch_size
    decode_s    = max(e2e_s - ttft_s, 1e-6)
    tok_per_s   = total_new / decode_s

    mem_final = gpu_memory_snapshot(device_ids)

    return {
        "ttft_s":              round(ttft_s,    4),
        "decode_s":            round(decode_s,  4),
        "e2e_s":               round(e2e_s,     4),
        "output_tokens":       output_toks,
        "total_new_tokens":    total_new,
        "tok_per_s":           round(tok_per_s, 2),
        "input_len":           input_len,
        "gpu_memory":          mem_final,
        "gpu_memory_prefill":  mem_after_prefill,
        # compute efficiency proxy (output tok per GB per second)
        "efficiency_tok_per_GB_s": round(
            tok_per_s / max(mem_final.get("GPU:0", {}).get("allocated_GB", 1), 0.1), 2
        ),
    }


# ── Batch sweep for one context tier ─────────────────────────────────────────

def sweep_batch_sizes(model, tokenizer, tier: dict, device_ids: list,
                      use_torch_profiler: bool) -> list:
    label          = tier["label"]
    prompt         = tier["prompt"]
    max_new_tokens = tier["max_new_tokens"]
    results        = []
    sampler        = GPUStatsSampler(device_ids)

    print(f"\n  ── {label} {'─'*40}")

    for bs in BATCH_SIZES:
        print(f"\n     batch_size={bs}")

        # Warmup
        try:
            for _ in range(WARMUP_RUNS):
                run_batch(model, tokenizer, prompt, bs, max_new_tokens,
                          device_ids, use_torch_profiler=False)
        except torch.cuda.OutOfMemoryError:
            print(f"     ⛔ OOM during warmup at batch_size={bs}")
            results.append({"batch_size": bs, "oom": True, "phase": "warmup"})
            torch.cuda.empty_cache()
            break

        run_records = []
        oom_hit     = False

        for r_idx in range(MEASURE_RUNS):
            try:
                sampler.start()
                # Only use torch.profiler on first measure run of first batch to keep overhead low
                use_prof_this_run = use_torch_profiler and r_idx == 0
                r         = run_batch(model, tokenizer, prompt, bs, max_new_tokens,
                                      device_ids, use_torch_profiler=use_prof_this_run)
                gpu_stats = sampler.stop()
                r["gpu_util"] = gpu_stats
                run_records.append(r)

                mem0     = r["gpu_memory"].get("GPU:0", {})
                peak_mem = mem0.get("peak_allocated_GB", 0)
                util0    = gpu_stats.get("GPU:0", {}).get("avg_util_pct", 0)
                membw0   = gpu_stats.get("GPU:0", {}).get("avg_mem_bw_pct")
                membw_str = f"  memBW={membw0:.0f}%" if membw0 is not None else ""

                print(
                    f"       run {r_idx+1}: "
                    f"TTFT={r['ttft_s']:.3f}s  "
                    f"e2e={r['e2e_s']:.3f}s  "
                    f"tput={r['tok_per_s']:.1f} tok/s  "
                    f"out={r['output_tokens']} tok  "
                    f"alloc={mem0.get('allocated_GB', 0):.2f}GB  "
                    f"peak={peak_mem:.2f}GB  "
                    f"util={util0:.0f}%{membw_str}"
                )

            except torch.cuda.OutOfMemoryError:
                sampler.stop()
                print(f"       ⛔ OOM at run {r_idx+1}, batch_size={bs}")
                results.append({"batch_size": bs, "oom": True, "phase": f"measure_run_{r_idx+1}"})
                oom_hit = True
                torch.cuda.empty_cache()
                break

        if oom_hit:
            break

        n = len(run_records)

        def avg(key):
            return round(sum(r[key] for r in run_records) / n, 4)

        def std(key):
            vals = [r[key] for r in run_records]
            return round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)

        agg = {
            "batch_size":             bs,
            "oom":                    False,
            "input_tokens":           run_records[0]["input_len"],
            "output_tokens_avg":      avg("output_tokens"),
            "ttft_avg_s":             avg("ttft_s"),
            "ttft_std_s":             std("ttft_s"),
            "e2e_avg_s":              avg("e2e_s"),
            "e2e_std_s":              std("e2e_s"),
            "decode_avg_s":           avg("decode_s"),
            "tok_per_s_avg":          avg("tok_per_s"),
            "tok_per_s_std":          std("tok_per_s"),
            "efficiency_tok_per_GB_s_avg": avg("efficiency_tok_per_GB_s"),
            "gpu_memory":             run_records[-1]["gpu_memory"],
            "gpu_memory_prefill":     run_records[-1]["gpu_memory_prefill"],
            "gpu_util":               run_records[-1].get("gpu_util", {}),
        }

        # Latency budget breakdown (% of e2e)
        agg["prefill_pct"]  = round(100 * agg["ttft_avg_s"]    / max(agg["e2e_avg_s"], 1e-6), 1)
        agg["decode_pct"]   = round(100 * agg["decode_avg_s"]  / max(agg["e2e_avg_s"], 1e-6), 1)

        print(
            f"     AVG → TTFT={agg['ttft_avg_s']:.3f}s ({agg['prefill_pct']}%)  "
            f"decode={agg['decode_avg_s']:.3f}s ({agg['decode_pct']}%)  "
            f"e2e={agg['e2e_avg_s']:.3f}s (±{agg['e2e_std_s']:.3f})  "
            f"tput={agg['tok_per_s_avg']:.1f} tok/s (±{agg['tok_per_s_std']:.1f})  "
            f"eff={agg['efficiency_tok_per_GB_s_avg']:.2f} tok/GB/s"
        )

        results.append(agg)

    return results


# ── Top-level benchmark ────────────────────────────────────────────────────────

def benchmark(num_gpus: int, use_flash_attn: bool, quantize: str,
              use_torch_profiler: bool, vram_poll_hz: float) -> dict:
    model, tokenizer, device_ids, load_meta = load_model(
        num_gpus, use_flash_attn, quantize, use_torch_profiler
    )

    # Start VRAM timeline for entire benchmark run
    vram_sampler = VRAMTimelineSampler(device_ids, poll_hz=vram_poll_hz,
                                       path=f"vram_timeline_gpu{num_gpus}_{quantize}.csv")
    vram_sampler.start()

    tiers   = build_context_tiers(tokenizer)
    results = {
        "num_gpus":    num_gpus,
        "load_config": load_meta,
        "tiers":       [],
    }

    for tier in tiers:
        with _global_timer.phase(f"tier_{tier['label'].strip()}"):
            tier_result = {
                "label":          tier["label"],
                "target_input":   tier["target_input"],
                "max_new_tokens": tier["max_new_tokens"],
                "batch_sweeps":   sweep_batch_sizes(model, tokenizer, tier, device_ids,
                                                    use_torch_profiler),
            }
        results["tiers"].append(tier_result)

    vram_sampler.stop()
    results["gpu_memory_final"] = gpu_memory_snapshot(device_ids)
    results["phase_timings"]    = _global_timer.summary()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ── Summary printer ────────────────────────────────────────────────────────────

def print_summary(all_results: list):
    print("\n")
    print("╔" + "═"*92 + "╗")
    print("║" + "  PROFILED BENCHMARK SUMMARY — Param2-17B".center(92) + "║")
    print("╚" + "═"*92 + "╝")

    for res in all_results:
        n   = res["num_gpus"]
        cfg = res.get("load_config", {})
        print(f"\n{'─'*92}")
        print(f"  {n} GPU — attn={cfg.get('attn_impl','?')}  dtype={cfg.get('dtype','?')}  "
              f"quantize={cfg.get('quantize','?')}  load_time={cfg.get('load_time_s','?')}s")
        print(f"{'─'*92}")

        # Phase timings table
        print(f"\n  ⏱  Phase Timings")
        print(f"  {'Phase':<45}  {'Time (s)':>10}")
        print(f"  {'─'*45}  {'─'*10}")
        for entry in res.get("phase_timings", []):
            print(f"  {entry['phase']:<45}  {entry['elapsed_s']:>10.3f}")

        for tier in res["tiers"]:
            print(f"\n  📊 {tier['label']}")
            print(f"  {'Batch':>6} {'InTok':>7} {'TTFT%':>6} {'DecPct':>7} "
                  f"{'TTFT(s)':>9} {'E2E(s)':>9} {'±Std':>7} "
                  f"{'tok/s':>8} {'PeakGB':>8} {'Util%':>7} {'MemBW%':>7} {'eff':>7}")
            print(f"  {'─'*6} {'─'*7} {'─'*6} {'─'*7} "
                  f"{'─'*9} {'─'*9} {'─'*7} "
                  f"{'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7}")

            for sw in tier["batch_sweeps"]:
                bs = sw["batch_size"]
                if sw["oom"]:
                    print(f"  {bs:>6}  {'⛔ OOM':>60}")
                    continue

                mem0     = sw["gpu_memory"].get("GPU:0", {})
                peak_mem = mem0.get("peak_allocated_GB", 0)
                util0    = sw.get("gpu_util", {}).get("GPU:0", {}).get("avg_util_pct", 0)
                membw0   = sw.get("gpu_util", {}).get("GPU:0", {}).get("avg_mem_bw_pct")
                membw_str = f"{membw0:>7.1f}" if membw0 is not None else "    n/a"
                eff      = sw.get("efficiency_tok_per_GB_s_avg", 0)

                print(
                    f"  {bs:>6} "
                    f"{sw['input_tokens']:>7} "
                    f"{sw.get('prefill_pct',0):>5.1f}% "
                    f"{sw.get('decode_pct',0):>6.1f}% "
                    f"{sw['ttft_avg_s']:>9.3f} "
                    f"{sw['e2e_avg_s']:>9.3f} "
                    f"±{sw['e2e_std_s']:>6.3f} "
                    f"{sw['tok_per_s_avg']:>8.1f} "
                    f"{peak_mem:>8.2f} "
                    f"{util0:>7.1f} "
                    f"{membw_str} "
                    f"{eff:>7.2f}"
                )

    print(f"\n{'─'*92}")
    print("  💾 Final GPU Memory:")
    for res in all_results:
        print(f"\n    {res['num_gpus']} GPU ({res['load_config'].get('quantize','none')}):")
        for gpu, mem in res["gpu_memory_final"].items():
            print(f"      {gpu}: alloc={mem['allocated_GB']} GB  "
                  f"peak={mem['peak_allocated_GB']} GB  "
                  f"retries={mem['num_alloc_retries']}  "
                  f"/ {mem['total_GB']} GB total")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Param2-17B profiled benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--gpus",          nargs="+", type=int, default=[1])
    parser.add_argument("--flash-attn",    action="store_true")
    parser.add_argument("--quantize",      choices=["none","int8","int4"], default="none")
    parser.add_argument("--output",        default="benchmark_results_profiled.json")
    parser.add_argument("--torch-profiler",action="store_true",
                        help="Enable torch.profiler trace (saved to ./torch_trace/)")
    parser.add_argument("--cprofile",      action="store_true",
                        help="Wrap entire run in cProfile → benchmark_profile.prof")
    parser.add_argument("--vram-poll-hz",  type=float, default=2.0,
                        help="VRAM polling frequency in Hz (default 2)")
    args = parser.parse_args()

    available = torch.cuda.device_count()
    print(f"\n  Detected {available} GPU(s) on this machine.")

    if args.quantize != "none":
        try:
            import bitsandbytes  # noqa
        except ImportError:
            print("  ❌ bitsandbytes not installed.  pip install bitsandbytes")
            sys.exit(1)

    def _run():
        all_results = []
        for n in args.gpus:
            if n > available:
                print(f"  ⚠️  Skipping {n}-GPU run: only {available} GPU(s) available.")
                continue
            all_results.append(
                benchmark(n, args.flash_attn, args.quantize,
                          args.torch_profiler, args.vram_poll_hz)
            )
        if all_results:
            print_summary(all_results)
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  ✅ Results saved → {args.output}\n")

    if args.cprofile:
        print("  🔬 cProfile enabled — profiling entire run ...")
        pr = cProfile.Profile()
        pr.enable()
        _run()
        pr.disable()

        # Save raw profile
        prof_path = "benchmark_profile.prof"
        pr.dump_stats(prof_path)
        print(f"  💾 cProfile dump → {prof_path}")
        print(f"     View with: snakeviz {prof_path}")
        print(f"     Or:  python -m pstats {prof_path}")

        # Print top-30 by cumulative time
        s  = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(30)
        print("\n  ── cProfile top-30 functions (cumulative time) ──")
        # Filter to only our code + torch lines for readability
        for line in s.getvalue().splitlines():
            if any(kw in line for kw in ["benchmark", "run_batch", "generate", "forward",
                                          "torch", "model", "tokeniz", "cumtime", "ncalls",
                                          "tottime", "─", "Ordered"]):
                print("  " + line)
    else:
        _run()


if __name__ == "__main__":
    main()
