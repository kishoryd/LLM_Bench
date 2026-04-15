"""
GPU memory snapshots and tracking utilities.
"""

import torch


def gpu_memory_snapshot(device_ids: list[int]) -> dict:
    """Capture current GPU memory allocation for each device."""
    snap = {}
    for i in device_ids:
        snap[f"GPU:{i}"] = {
            "allocated_GB": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
            "reserved_GB": round(torch.cuda.memory_reserved(i) / 1024**3, 2),
            "total_GB": round(
                torch.cuda.get_device_properties(i).total_memory / 1024**3, 2
            ),
        }
    return snap


def gpu_memory_used_pct(device_id: int = 0) -> float:
    """Return memory utilisation as a percentage."""
    alloc = torch.cuda.memory_allocated(device_id)
    total = torch.cuda.get_device_properties(device_id).total_memory
    return round((alloc / total) * 100, 1)


def reset_memory_stats(device_ids: list[int] | None = None):
    """Reset peak memory tracking for specified devices."""
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    for i in device_ids:
        torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.empty_cache()
