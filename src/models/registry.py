"""
Model registry: load model configs, resolve paths, load tokenizers.

Adding a new model:
  1. Create configs/models/your_model.yaml
  2. That's it — the registry picks it up automatically.
"""

from transformers import AutoTokenizer
from src.utils.config import load_model_config, list_available_models


class ModelRegistry:
    """
    Loads and caches model configurations from YAML files.

    Usage:
        registry = ModelRegistry()
        cfg = registry.get("param2_17b")
        tokenizer = registry.load_tokenizer("param2_17b")
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}

    def get(self, model_name: str) -> dict:
        """Get model config by name (cached)."""
        if model_name not in self._cache:
            self._cache[model_name] = load_model_config(model_name)
        return self._cache[model_name]

    def load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Load the HuggingFace tokenizer for a model."""
        cfg = self.get(model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["path"],
            trust_remote_code=cfg.get("trust_remote_code", False),
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    @staticmethod
    def list_models() -> list[str]:
        """List all available model config names."""
        return list_available_models()
