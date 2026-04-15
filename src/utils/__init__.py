"""
YAML config loader and validator.

Loads main bench config, model configs, backend configs, and profiler configs.
"""

import os
import yaml
from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parent.parent.parent / "configs"


def load_yaml(path: str | Path) -> dict:
    """Load a single YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_bench_config(config_path: str | None = None) -> dict:
    """Load the main benchmark config."""
    path = Path(config_path) if config_path else CONFIG_ROOT / "bench.yaml"
    return load_yaml(path)


def load_model_config(model_name: str) -> dict:
    """
    Load a model config by name.

    Args:
        model_name: Filename without .yaml (e.g. 'param2_17b')
    """
    path = CONFIG_ROOT / "models" / f"{model_name}.yaml"
    cfg = load_yaml(path)

    # Validate required fields
    required = ["name", "path", "max_context"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Model config {path} missing required fields: {missing}")

    return cfg


def load_backend_config(backend_name: str) -> dict:
    """
    Load a backend config by name.

    Args:
        backend_name: 'native', 'vllm', 'sglang', 'triton', 'ollama'
    """
    path = CONFIG_ROOT / "backends" / f"{backend_name}.yaml"
    return load_yaml(path)


def load_profiler_config(profiler_name: str) -> dict:
    """
    Load a profiler config by name.

    Args:
        profiler_name: 'nsight' or 'pytorch_profiler'
    """
    path = CONFIG_ROOT / "profiles" / f"{profiler_name}.yaml"
    return load_yaml(path)


def list_available_models() -> list[str]:
    """List all model config names available in configs/models/."""
    models_dir = CONFIG_ROOT / "models"
    if not models_dir.exists():
        return []
    return [f.stem for f in models_dir.glob("*.yaml")]


def list_available_backends() -> list[str]:
    """List all backend config names available in configs/backends/."""
    backends_dir = CONFIG_ROOT / "backends"
    if not backends_dir.exists():
        return []
    return [f.stem for f in backends_dir.glob("*.yaml")]
