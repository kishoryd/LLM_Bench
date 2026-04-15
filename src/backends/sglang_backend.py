"""
SGLang backend — RadixAttention, continuous batching via HTTP API.
"""

import time
import requests

from src.backends.base import BaseBackend


class SGLangBackend(BaseBackend):
    """SGLang runtime via its REST API."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = None

    def setup(self, model_config: dict, num_gpus: int = 1, **kwargs):
        server_cfg = self.config.get("server", {})
        host = server_cfg.get("host", "localhost")
        port = server_cfg.get("port", 30000)
        api_path = server_cfg.get("api_path", "/generate")
        self.server_url = f"http://{host}:{port}{api_path}"

        # Health check
        try:
            resp = requests.get(f"http://{host}:{port}/health", timeout=5)
            print(f"  SGLang server connected at {self.server_url}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach SGLang server at {host}:{port} — "
                f"start it with: ./scripts/start_sglang.sh\n"
                f"Error: {e}"
            )

    def generate(self, prompt: str, max_new_tokens: int, batch_size: int = 1, **kwargs) -> dict:
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": kwargs.get("temperature", 0.7),
                "top_k": kwargs.get("top_k", 50),
                "top_p": kwargs.get("top_p", 0.9),
            },
        }

        # SGLang doesn't natively batch via REST — send sequential requests
        results = []
        t_start = time.perf_counter()
        for _ in range(batch_size):
            resp = requests.post(self.server_url, json=payload, timeout=120)
            resp.raise_for_status()
            results.append(resp.json())
        t_end = time.perf_counter()

        e2e_s = t_end - t_start
        total_tokens = sum(
            r.get("meta_info", {}).get("completion_tokens", 0)
            or len(r.get("text", "").split())
            for r in results
        )
        tok_per_s = total_tokens / max(e2e_s, 1e-6)

        return {
            "ttft_s": 0,
            "decode_s": round(e2e_s, 4),
            "e2e_s": round(e2e_s, 4),
            "output_tokens": total_tokens // max(batch_size, 1),
            "total_new_tokens": total_tokens,
            "tok_per_s": round(tok_per_s, 2),
            "input_len": 0,
            "gpu_memory": {},
        }

    def teardown(self):
        print(f"  {self.name} backend teardown complete (server still running).")
