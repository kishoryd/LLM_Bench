"""
Comprehensive Benchmark for Param2-17B-A2.4B-Thinking
======================================================
Tests: 1 GPU vs 2 GPU
Sweeps: batch size 1 → OOM across 4 context tiers

Metrics per run:
  - Time to First Token (TTFT)         — prefill speed
  - Decode throughput (tok/s)          — generation speed
  - End-to-end latency                 — total wall time
  - Output token count                 — did model stop early?
  - Latency std dev                    — stability across runs
  - GPU utilisation %                  — compute efficiency
  - GPU memory allocated / reserved    — memory pressure
  - PCIe/NVLink bandwidth (2-GPU only) — inter-GPU cost

Context tiers (input tokens):
  short      ~256  | output budget 128
  medium    ~1024  | output budget 256
  long      ~2048  | output budget 256
  near-limit ~3584 | output budget 128

GPU Presets:
  RTX Ada 5000  → bf16 + flash_attention_2  (Ada Lovelace, 32GB)
  A100          → bf16 + flash_attention_2  (Ampere, 40/80GB)

Usage:
  # Auto-detect GPU and pick best settings
  python benchmark_param2.py --gpus 1

  # Explicit flash attention
  python benchmark_param2.py --gpus 1 --flash-attn

  # INT8 quantization (cuts VRAM ~50%)
  python benchmark_param2.py --gpus 1 --quantize int8

  # INT4 quantization (cuts VRAM ~75%)
  python benchmark_param2.py --gpus 1 --quantize int4

  CUDA_VISIBLE_DEVICES=0 python test_gpu_4k_common.py --gpus 1 --quantize int4


  # Flash attention + INT8 together
  python benchmark_param2.py --gpus 1 --flash-attn --quantize int8

  # Full 2-GPU sweep with all options
  python benchmark_param2.py --gpus 1 2 --flash-attn --quantize int8 --output results.json
"""

import sys, time, argparse, json, gc, os, threading, statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from transformers.utils.import_utils import is_torch_fx_available
except ImportError:
    def is_torch_fx_available():
        return False

sys.path.append("/home/kishor/LLM_Bench/data/Param2-17B")

MODEL_PATH   = "/home/kishor/LLM_Bench/data/Param2-17B"
MAX_CTX      = 4096
WARMUP_RUNS  = 1
MEASURE_RUNS = 3
TEMPERATURE  = 0.7
TOP_K        = 50
TOP_P        = 0.9

BATCH_SIZES  = [1, 2, 4, 8, 16, 32]
#BATCH_SIZES  = [64]

# ── GPU architecture detection ────────────────────────────────────────────────

# Compute capability → architecture name
ARCH_MAP = {
    (8, 0): "Ampere",       # A100
    (8, 6): "Ampere",       # A10, A30, A40
    (8, 9): "Ada Lovelace", # RTX 4090, RTX Ada 5000/6000
    (9, 0): "Hopper",       # H100
    (7, 0): "Volta",        # V100
    (7, 5): "Turing",       # T4, RTX 20xx
}

# Architectures that support Flash Attention 2 (requires sm >= 8.0)
FA2_SUPPORTED_ARCHS = {"Ampere", "Ada Lovelace", "Hopper"}

# Architectures that support bfloat16 natively (sm >= 8.0)
BF16_SUPPORTED_ARCHS = {"Ampere", "Ada Lovelace", "Hopper"}


def detect_gpu_info(device_id: int = 0) -> dict:
    """Returns GPU name, architecture, compute capability, and VRAM."""
    props = torch.cuda.get_device_properties(device_id)
    cc    = (props.major, props.minor)
    arch  = ARCH_MAP.get(cc, f"Unknown (sm_{props.major}{props.minor})")
    vram_gb = round(props.total_memory / 1024**3, 1)
    return {
        "name":          props.name,
        "arch":          arch,
        "compute_cap":   f"sm_{props.major}{props.minor}",
        "vram_gb":       vram_gb,
        "fa2_supported": arch in FA2_SUPPORTED_ARCHS,
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
        print(f"         Flash Attn 2 : {'✅ Supported' if info['fa2_supported'] else '❌ Not supported (need sm_80+)'}")
        print(f"         bfloat16     : {'✅ Supported' if info['bf16_supported'] else '⚠️  Falling back to float16'}")
    print()


def auto_select_dtype(num_gpus: int) -> torch.dtype:
    """Pick bf16 if all GPUs support it, else fp16."""
    all_bf16 = all(detect_gpu_info(i)["bf16_supported"] for i in range(num_gpus))
    dtype = torch.bfloat16 if all_bf16 else torch.float16
    label = "bfloat16" if all_bf16 else "float16"
    print(f"  dtype auto-selected : {label}")
    return dtype


# ── Quantization config builder ───────────────────────────────────────────────

def build_quant_config(quantize: str, dtype: torch.dtype):
    """
    Returns a BitsAndBytesConfig (or None) and the effective load dtype.

    int4 → 4-bit NF4, double-quant, bf16 compute
    int8 → 8-bit LLM.int8(), fp16 compute
    none → no quantization
    """
    if quantize == "int4":
        print("  Quantization        : INT4 (NF4, double-quant) — ~75% VRAM reduction")
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,   # compute in bf16 even if weights are int4
        )
        return cfg, None   # device_map="auto" handles dtype internally

    elif quantize == "int8":
        print("  Quantization        : INT8 (LLM.int8) — ~50% VRAM reduction")
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        return cfg, None

    else:
        print("  Quantization        : None (full precision)")
        return None, dtype


# ── GPU utilisation sampler (background thread) ───────────────────────────────

class GPUStatsSampler:
    """Polls GPU utilisation % every 100ms in a background thread."""
    def __init__(self, device_ids):
        self.device_ids = device_ids
        self._util      = {i: [] for i in device_ids}
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._stop.clear()
        self._util   = {i: [] for i in self.device_ids}
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
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
                    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    self._util[i].append(util)
                time.sleep(0.1)
        except Exception:
            while not self._stop.is_set():
                for i in self.device_ids:
                    self._util[i].append(0)
                time.sleep(0.1)


# ── Prompt builder ────────────────────────────────────────────────────────────

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


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(num_gpus: int, use_flash_attn: bool, quantize: str):
    print(f"\n{'='*64}")
    print(f"  Loading model on {num_gpus} GPU(s) ...")
    print(f"{'='*64}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    # ── Print GPU info and pick dtype ─────────────────────────────────────────
    print_gpu_info(num_gpus)
    dtype = auto_select_dtype(num_gpus)

    # ── Flash Attention 2 ─────────────────────────────────────────────────────
    gpu0_info = detect_gpu_info(0)
    attn_impl = None

    if use_flash_attn:
        if gpu0_info["fa2_supported"]:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                print(f"  Flash Attention 2   : ✅ Enabled")
            except ImportError:
                print(f"  Flash Attention 2   : ⚠️  flash-attn not installed — falling back to SDPA")
                print(f"                         Run: pip install flash-attn --no-build-isolation")
                attn_impl = "sdpa"
        else:
            print(f"  Flash Attention 2   : ❌ GPU is {gpu0_info['arch']} ({gpu0_info['compute_cap']}) — need sm_80+")
            print(f"                         Falling back to SDPA (still faster than eager)")
            attn_impl = "sdpa"
    else:
        attn_impl = "sdpa"
        print(f"  Flash Attention 2   : Not requested — using SDPA")

    # ── Quantization ──────────────────────────────────────────────────────────
    quant_cfg, load_dtype = build_quant_config(quantize, dtype)
    effective_dtype = load_dtype if load_dtype is not None else dtype

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=False, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Model ─────────────────────────────────────────────────────────────────
    load_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
        local_files_only=True,
        attn_implementation=attn_impl,
    )

    # Quantization and dtype are mutually exclusive at load time for BnB
    if quant_cfg is not None:
        load_kwargs["quantization_config"] = quant_cfg
        # BnB handles dtype internally; don't pass torch_dtype with int4
        if quantize == "int8":
            load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = effective_dtype

    print(f"\n  Loading with settings:")
    print(f"    attn_implementation = {attn_impl}")
    print(f"    torch_dtype         = {load_kwargs.get('torch_dtype', 'managed by BnB')}")
    print(f"    quantize            = {quantize}")
    print()

    t0    = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)
    load_time = time.perf_counter() - t0
    model.eval()

    device_ids = list(range(num_gpus))
    print(f"  Loaded in {load_time:.1f}s")
    for i in device_ids:
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU:{i}  {alloc:.2f} / {total:.2f} GB")

    return model, tokenizer, device_ids, {
        "attn_impl":  attn_impl,
        "dtype":      str(effective_dtype),
        "quantize":   quantize,
        "load_time_s": round(load_time, 2),
    }


# ── GPU memory snapshot ───────────────────────────────────────────────────────

def gpu_memory_snapshot(device_ids: list) -> dict:
    snap = {}
    for i in device_ids:
        snap[f"GPU:{i}"] = {
            "allocated_GB": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
            "reserved_GB":  round(torch.cuda.memory_reserved(i)  / 1024**3, 2),
            "total_GB":     round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
        }
    return snap


# ── Single batched inference run ──────────────────────────────────────────────

def run_batch(model, tokenizer, prompt: str, batch_size: int,
              max_new_tokens: int, device_ids: list) -> dict:
    conversations = [
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user",   "content": prompt}]
    ] * batch_size

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
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids=padded, attention_mask=attn_mask)
    torch.cuda.synchronize()
    ttft_s = time.perf_counter() - t_start

    # ── Full generation ───────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t_gen_start = time.perf_counter()
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
    t_end = time.perf_counter()

    e2e_s       = t_end - t_gen_start
    output_toks = output.shape[1] - input_len
    total_new   = output_toks * batch_size
    decode_s    = max(e2e_s - ttft_s, 1e-6)
    tok_per_s   = total_new / decode_s

    return {
        "ttft_s":           round(ttft_s,    4),
        "decode_s":         round(decode_s,  4),
        "e2e_s":            round(e2e_s,     4),
        "output_tokens":    output_toks,
        "total_new_tokens": total_new,
        "tok_per_s":        round(tok_per_s, 2),
        "input_len":        input_len,
        "gpu_memory":       gpu_memory_snapshot(device_ids),
    }


# ── Batch sweep for one context tier ─────────────────────────────────────────

def sweep_batch_sizes(model, tokenizer, tier: dict, device_ids: list) -> list:
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
                run_batch(model, tokenizer, prompt, bs, max_new_tokens, device_ids)
        except torch.cuda.OutOfMemoryError:
            print(f"     ⛔ OOM during warmup at batch_size={bs} — stopping sweep.")
            results.append({"batch_size": bs, "oom": True, "phase": "warmup"})
            torch.cuda.empty_cache()
            break

        # Measure
        run_records = []
        oom_hit     = False

        for r_idx in range(MEASURE_RUNS):
            try:
                sampler.start()
                r         = run_batch(model, tokenizer, prompt, bs, max_new_tokens, device_ids)
                gpu_stats = sampler.stop()
                r["gpu_util"] = gpu_stats
                run_records.append(r)

                print(
                    f"       run {r_idx+1}: "
                    f"TTFT={r['ttft_s']:.3f}s  "
                    f"e2e={r['e2e_s']:.3f}s  "
                    f"throughput={r['tok_per_s']:.1f} tok/s  "
                    f"output={r['output_tokens']} tok  "
                    f"mem={r['gpu_memory']['GPU:0']['allocated_GB']:.2f}GB"
                )
            except torch.cuda.OutOfMemoryError:
                sampler.stop()
                print(f"       ⛔ OOM at run {r_idx+1}, batch_size={bs} — stopping sweep.")
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
            "batch_size":        bs,
            "oom":               False,
            "input_tokens":      run_records[0]["input_len"],
            "output_tokens_avg": avg("output_tokens"),
            "ttft_avg_s":        avg("ttft_s"),
            "ttft_std_s":        std("ttft_s"),
            "e2e_avg_s":         avg("e2e_s"),
            "e2e_std_s":         std("e2e_s"),
            "decode_avg_s":      avg("decode_s"),
            "tok_per_s_avg":     avg("tok_per_s"),
            "tok_per_s_std":     std("tok_per_s"),
            "gpu_memory":        run_records[-1]["gpu_memory"],
            "gpu_util":          run_records[-1].get("gpu_util", {}),
        }

        print(
            f"     AVG → TTFT={agg['ttft_avg_s']:.3f}s (±{agg['ttft_std_s']:.3f})  "
            f"e2e={agg['e2e_avg_s']:.3f}s (±{agg['e2e_std_s']:.3f})  "
            f"throughput={agg['tok_per_s_avg']:.1f} tok/s (±{agg['tok_per_s_std']:.1f})"
        )

        results.append(agg)

    return results


# ── Top-level benchmark ───────────────────────────────────────────────────────

def benchmark(num_gpus: int, use_flash_attn: bool, quantize: str) -> dict:
    model, tokenizer, device_ids, load_meta = load_model(num_gpus, use_flash_attn, quantize)
    tiers   = build_context_tiers(tokenizer)
    results = {
        "num_gpus":    num_gpus,
        "load_config": load_meta,
        "tiers":       [],
    }

    for tier in tiers:
        tier_result = {
            "label":          tier["label"],
            "target_input":   tier["target_input"],
            "max_new_tokens": tier["max_new_tokens"],
            "batch_sweeps":   sweep_batch_sizes(model, tokenizer, tier, device_ids),
        }
        results["tiers"].append(tier_result)

    results["gpu_memory_final"] = gpu_memory_snapshot(device_ids)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(all_results: list):
    print("\n")
    print("╔" + "═"*90 + "╗")
    print("║" + "  BENCHMARK SUMMARY — Param2-17B  (max ctx 4096 tok)".center(90) + "║")
    print("╚" + "═"*90 + "╝")

    for res in all_results:
        n   = res["num_gpus"]
        cfg = res.get("load_config", {})
        print(f"\n{'─'*90}")
        print(f"  {n} GPU{'s' if n > 1 else ' '} Configuration")
        print(f"  attn={cfg.get('attn_impl','?')}  dtype={cfg.get('dtype','?')}  "
              f"quantize={cfg.get('quantize','?')}  load_time={cfg.get('load_time_s','?')}s")
        print(f"{'─'*90}")

        for tier in res["tiers"]:
            print(f"\n  📊 {tier['label']}")
            print(f"  {'Batch':>6} {'Input tok':>10} {'TTFT (s)':>12} {'E2E (s)':>12} "
                  f"{'±StdDev':>9} {'Throughput':>13} {'GPU Mem':>10} {'Util%':>7}")
            print(f"  {'─'*6} {'─'*10} {'─'*12} {'─'*12} {'─'*9} {'─'*13} {'─'*10} {'─'*7}")

            for sw in tier["batch_sweeps"]:
                bs = sw["batch_size"]
                if sw["oom"]:
                    print(f"  {bs:>6}  {'⛔ OOM':>60}")
                    continue

                mem0  = sw["gpu_memory"].get("GPU:0", {}).get("allocated_GB", 0)
                util0 = sw.get("gpu_util", {}).get("GPU:0", {}).get("avg_util_pct", 0)

                print(
                    f"  {bs:>6} "
                    f"{sw['input_tokens']:>10} "
                    f"{sw['ttft_avg_s']:>11.3f}s "
                    f"{sw['e2e_avg_s']:>11.3f}s "
                    f"±{sw['e2e_std_s']:>7.3f} "
                    f"{sw['tok_per_s_avg']:>11.1f} t/s "
                    f"{mem0:>8.2f} GB "
                    f"{util0:>6.1f}%"
                )

    print(f"\n{'─'*90}")
    print("  💾 Final GPU Memory:")
    for res in all_results:
        print(f"\n    {res['num_gpus']} GPU ({res['load_config'].get('quantize','none')}):")
        for gpu, mem in res["gpu_memory_final"].items():
            print(f"      {gpu}: {mem['allocated_GB']} GB / {mem['total_GB']} GB total")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Param2-17B batch-sweep benchmark with Flash Attention and quantization",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=[1],
        help="Number of GPUs to test. E.g. --gpus 1 2",
    )
    parser.add_argument(
        "--flash-attn", action="store_true",
        help=(
            "Enable Flash Attention 2 (requires sm_80+ GPU and flash-attn package).\n"
            "Falls back to SDPA automatically if unavailable.\n"
            "Install: pip install flash-attn --no-build-isolation"
        ),
    )
    parser.add_argument(
        "--quantize", choices=["none", "int8", "int4"], default="none",
        help=(
            "Quantization mode:\n"
            "  none  — full precision (default)\n"
            "  int8  — LLM.int8 via bitsandbytes (~50%% VRAM reduction)\n"
            "  int4  — NF4 double-quant via bitsandbytes (~75%% VRAM reduction)\n"
            "Install: pip install bitsandbytes"
        ),
    )
    parser.add_argument(
        "--output", default="benchmark_results.json",
        help="Path to save JSON results (default: benchmark_results.json)",
    )
    args = parser.parse_args()

    available = torch.cuda.device_count()
    print(f"\n  Detected {available} GPU(s) on this machine.")

    # Validate quantize dependency
    if args.quantize != "none":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("\n  ❌ bitsandbytes not installed. Run:")
            print("       pip install bitsandbytes")
            sys.exit(1)

    all_results = []
    for n in args.gpus:
        if n > available:
            print(f"  ⚠️  Skipping {n}-GPU run: only {available} GPU(s) available.")
            continue
        all_results.append(benchmark(n, args.flash_attn, args.quantize))

    if all_results:
        print_summary(all_results)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  ✅ Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
