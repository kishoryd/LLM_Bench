"""
Backend registry — maps backend names to their classes.

Adding a new backend:
  1. Create src/backends/your_backend.py
  2. Inherit from BaseBackend
  3. Add it to BACKEND_REGISTRY below
"""

from src.backends.base import BaseBackend


# Lazy imports to avoid pulling in optional dependencies
def _get_backend_class(name: str) -> type[BaseBackend]:
    if name == "native":
        from src.backends.native import NativeBackend
        return NativeBackend
    elif name == "vllm":
        from src.backends.vllm_backend import VLLMBackend
        return VLLMBackend
    elif name == "sglang":
        from src.backends.sglang_backend import SGLangBackend
        return SGLangBackend
    elif name == "triton":
        from src.backends.triton_backend import TritonBackend
        return TritonBackend
    elif name == "ollama":
        from src.backends.ollama_backend import OllamaBackend
        return OllamaBackend
    else:
        raise ValueError(f"Unknown backend: '{name}'. Available: {ALL_BACKENDS}")


ALL_BACKENDS = ["native", "vllm", "sglang", "triton", "ollama"]


def get_backend(name: str, config: dict) -> BaseBackend:
    """Instantiate a backend by name with its config."""
    cls = _get_backend_class(name)
    return cls(config)
