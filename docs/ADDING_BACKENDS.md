# Adding a New Backend

## Steps

### 1. Create the backend module

Create `src/backends/your_backend.py`:

```python
"""Your Backend — description."""

from src.backends.base import BaseBackend


class YourBackend(BaseBackend):

    def __init__(self, config: dict):
        super().__init__(config)
        # Your init code

    def setup(self, model_config: dict, num_gpus: int = 1, **kwargs):
        """Load model or connect to server."""
        # ...

    def generate(self, prompt: str, max_new_tokens: int, batch_size: int = 1, **kwargs) -> dict:
        """
        Run inference. Must return a dict with these keys:
          ttft_s, decode_s, e2e_s, output_tokens,
          total_new_tokens, tok_per_s, input_len, gpu_memory
        """
        # ...

    def teardown(self):
        """Cleanup."""
        # ...
```

### 2. Register it

In `src/backends/__init__.py`, add to `_get_backend_class()`:

```python
elif name == "your_backend":
    from src.backends.your_backend import YourBackend
    return YourBackend
```

And add it to `ALL_BACKENDS`:

```python
ALL_BACKENDS = ["native", "vllm", "sglang", "triton", "ollama", "your_backend"]
```

### 3. Create the config

Create `configs/backends/your_backend.yaml`:

```yaml
name: "your_backend"
type: "your_backend"
description: "Your Backend — what it does"

server:
  host: "localhost"
  port: 9000
```

### 4. Run it

```bash
python scripts/run_bench.py --backend your_backend --model param2_17b
```

## generate() return format

Every backend must return this dict from `generate()`:

```python
{
    "ttft_s": float,          # time to first token (seconds)
    "decode_s": float,        # decode phase duration
    "e2e_s": float,           # end-to-end latency
    "output_tokens": int,     # tokens per sequence
    "total_new_tokens": int,  # tokens across all sequences
    "tok_per_s": float,       # decode throughput
    "input_len": int,         # input token count
    "gpu_memory": dict,       # memory snapshot (can be {} for server backends)
}
```

If your backend doesn't expose a metric (e.g. TTFT for server backends), set it to 0.
