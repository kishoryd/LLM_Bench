"""
Profiler registry — maps profiler names to their classes.
"""

from src.profilers.base import BaseProfiler


def get_profiler(name: str, config: dict) -> BaseProfiler:
    """Instantiate a profiler by name."""
    if name == "pytorch":
        from src.profilers.pytorch_profiler import PyTorchProfiler
        return PyTorchProfiler(config)
    elif name == "nsight":
        from src.profilers.nsight_profiler import NsightProfiler
        return NsightProfiler(config)
    elif name == "none" or name is None:
        return None
    else:
        raise ValueError(f"Unknown profiler: '{name}'. Available: pytorch, nsight")
