"""
Abstract base class for all inference backends.

Every backend must implement:
  - setup()       → load model / connect to server
  - generate()    → run inference, return timing metrics
  - teardown()    → cleanup resources
"""

from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """
    Interface that all backends implement.

    Lifecycle:
        backend = SomeBackend(config)
        backend.setup(model_config)
        result = backend.generate(prompt, max_new_tokens, batch_size)
        backend.teardown()
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Backend-specific config loaded from YAML.
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)

    @abstractmethod
    def setup(self, model_config: dict, num_gpus: int = 1, **kwargs):
        """
        Initialise the backend — load model weights, connect to server, etc.

        Args:
            model_config: Model config dict (path, max_context, dtype, ...).
            num_gpus: Number of GPUs to use.
            **kwargs: Extra options (quantize, flash_attn, ...).
        """
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        batch_size: int = 1,
        **kwargs,
    ) -> dict:
        """
        Run inference and return metrics.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            batch_size: Number of concurrent sequences.

        Returns:
            dict with keys:
                ttft_s          - time to first token (seconds)
                decode_s        - decode phase time
                e2e_s           - end-to-end latency
                output_tokens   - number of tokens generated
                tok_per_s       - decode throughput
                gpu_memory      - memory snapshot dict
        """
        ...

    @abstractmethod
    def teardown(self):
        """Release resources — unload model, disconnect, free GPU memory."""
        ...

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"
