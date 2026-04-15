"""
GPU auto-detection: architecture, compute capability, feature support.

Supports: Volta (V100), Turing (T4), Ampere (A100), Ada Lovelace (RTX 5000),
           Hopper (H100), and Grace Hopper (GH200).
"""

import torch


# ── Architecture lookup ──────────────────────────────────────────────────────

ARCH_MAP = {
    (7, 0): "Volta",          # V100
    (7, 5): "Turing",         # T4, RTX 20xx
    (8, 0): "Ampere",         # A100
    (8, 6): "Ampere",         # A10, A30, A40
    (8, 9): "Ada Lovelace",   # RTX 4090, RTX Ada 5000/6000
    (9, 0): "Hopper",         # H100, GH200
}

FA2_SUPPORTED_ARCHS = {"Ampere", "Ada Lovelace", "Hopper"}
BF16_SUPPORTED_ARCHS = {"Ampere", "Ada Lovelace", "Hopper"}
FP8_SUPPORTED_ARCHS = {"Hopper"}  # H100, GH200


# ── Detection ────────────────────────────────────────────────────────────────

def detect_gpu_info(device_id: int = 0) -> dict:
    """
    Returns a dict with GPU name, architecture, compute capability,
    VRAM, and feature support flags.
    """
    props = torch.cuda.get_device_properties(device_id)
    cc = (props.major, props.minor)
    arch = ARCH_MAP.get(cc, f"Unknown (sm_{props.major}{props.minor})")
    vram_gb = round(props.total_memory / 1024**3, 1)

    return {
        "name": props.name,
        "arch": arch,
        "compute_cap": f"sm_{props.major}{props.minor}",
        "vram_gb": vram_gb,
        "fa2_supported": arch in FA2_SUPPORTED_ARCHS,
        "bf16_supported": arch in BF16_SUPPORTED_ARCHS,
        "fp8_supported": arch in FP8_SUPPORTED_ARCHS,
        "multi_gpu": torch.cuda.device_count() > 1,
    }


def detect_all_gpus() -> list[dict]:
    """Detect info for every available GPU."""
    return [detect_gpu_info(i) for i in range(torch.cuda.device_count())]


def print_gpu_info(num_gpus: int):
    """Print detected GPU details to stdout."""
    print(f"\n{'─' * 64}")
    print(f"  GPU Detection ({num_gpus} device{'s' if num_gpus > 1 else ''})")
    print(f"{'─' * 64}")
    for i in range(num_gpus):
        info = detect_gpu_info(i)
        print(f"  GPU:{i}  {info['name']}")
        print(f"         Architecture : {info['arch']} ({info['compute_cap']})")
        print(f"         VRAM         : {info['vram_gb']} GB")
        print(f"         Flash Attn 2 : {'✅' if info['fa2_supported'] else '❌ (need sm_80+)'}")
        print(f"         bfloat16     : {'✅' if info['bf16_supported'] else '⚠️  float16 fallback'}")
        print(f"         FP8          : {'✅' if info['fp8_supported'] else '❌ (Hopper only)'}")
    print()


def auto_select_dtype(num_gpus: int) -> torch.dtype:
    """Pick bf16 if all GPUs support it, else fp16."""
    all_bf16 = all(detect_gpu_info(i)["bf16_supported"] for i in range(num_gpus))
    dtype = torch.bfloat16 if all_bf16 else torch.float16
    print(f"  dtype auto-selected : {'bfloat16' if all_bf16 else 'float16'}")
    return dtype
