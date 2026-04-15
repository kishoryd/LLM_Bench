"""
Ollama backend — local model runner via REST API.
"""

import time
import json
import requests

from src.backends.base import BaseBackend


class OllamaBackend(BaseBackend):
    """Ollama REST API client."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = None
        self.model_name = None

    def setup(self, model_config: dict, num_gpus: int = 1, **kwargs):
        server_cfg = self.config.get("server", {})
        host = server_cfg.get("host", "localhost")
        port = server_cfg.get("port", 11434)
        api_path = server_cfg.get("api_path", "/api/generate")
        self.server_url = f"http://{host}:{port}{api_path}"
        self.model_name = model_config.get("ollama_name", model_config["name"])

        # Health check
        try:
            resp = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"  Ollama connected. Available models: {models}")
            if self.model_name not in models:
                print(f"  ⚠️  Model '{self.model_name}' not found. Run: ollama pull {self.model_name}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {host}:{port} — "
                f"start it with: ollama serve\n"
                f"Error: {e}"
            )

    def generate(self, prompt: str, max_new_tokens: int, batch_size: int = 1, **kwargs) -> dict:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": kwargs.get("temperature", 0.7),
                "top_k": kwargs.get("top_k", 50),
                "top_p": kwargs.get("top_p", 0.9),
            },
        }

        # Ollama doesn't support batching — send sequential requests
        results = []
        t_start = time.perf_counter()
        for _ in range(batch_size):
            resp = requests.post(self.server_url, json=payload, timeout=120)
            resp.raise_for_status()
            results.append(resp.json())
        t_end = time.perf_counter()

        e2e_s = t_end - t_start
        total_tokens = sum(r.get("eval_count", 0) for r in results)
        tok_per_s = total_tokens / max(e2e_s, 1e-6)

        # Ollama provides some timing info
        first = results[0] if results else {}
        prompt_eval_duration = first.get("prompt_eval_duration", 0) / 1e9  # ns → s

        return {
            "ttft_s": round(prompt_eval_duration, 4),
            "decode_s": round(e2e_s - prompt_eval_duration, 4),
            "e2e_s": round(e2e_s, 4),
            "output_tokens": total_tokens // max(batch_size, 1),
            "total_new_tokens": total_tokens,
            "tok_per_s": round(tok_per_s, 2),
            "input_len": first.get("prompt_eval_count", 0),
            "gpu_memory": {},
        }

    def teardown(self):
        print(f"  {self.name} backend teardown complete (Ollama still running).")
