"""
vLLM backend — supports both offline (in-process) and server (HTTP API) modes.
"""

import time
import json
import requests

from src.backends.base import BaseBackend


class VLLMBackend(BaseBackend):
    """vLLM inference with paged attention and continuous batching."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.engine = None          # vLLM LLM engine (offline mode)
        self.server_url = None      # HTTP endpoint (server mode)
        self.model_name = None
        self.mode = "server"        # "offline" or "server"

    def setup(self, model_config: dict, num_gpus: int = 1, **kwargs):
        self.model_name = model_config["name"]
        self.mode = kwargs.get("mode", "server")

        if self.mode == "offline":
            self._setup_offline(model_config, num_gpus, **kwargs)
        else:
            self._setup_server()

    def _setup_offline(self, model_config: dict, num_gpus: int, **kwargs):
        """Load vLLM engine in-process."""
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")

        engine_cfg = self.config.get("engine", {})
        tp = kwargs.get("tensor_parallel", engine_cfg.get("tensor_parallel_size", 1))

        self.engine = LLM(
            model=model_config["path"],
            tensor_parallel_size=min(tp, num_gpus),
            gpu_memory_utilization=engine_cfg.get("gpu_memory_utilization", 0.90),
            max_num_seqs=engine_cfg.get("max_num_seqs", 256),
            enforce_eager=engine_cfg.get("enforce_eager", False),
            dtype=engine_cfg.get("dtype", "auto"),
            quantization=engine_cfg.get("quantization"),
            trust_remote_code=True,
        )
        print(f"  vLLM engine loaded (offline mode, TP={tp})")

    def _setup_server(self):
        """Connect to a running vLLM server."""
        server_cfg = self.config.get("server", {})
        host = server_cfg.get("host", "localhost")
        port = server_cfg.get("port", 8000)
        api_path = server_cfg.get("api_path", "/v1/completions")
        self.server_url = f"http://{host}:{port}{api_path}"

        # Health check
        try:
            resp = requests.get(f"http://{host}:{port}/health", timeout=5)
            resp.raise_for_status()
            print(f"  vLLM server connected at {self.server_url}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach vLLM server at {host}:{port} — "
                f"start it with: ./scripts/start_vllm.sh\n"
                f"Error: {e}"
            )

    def generate(self, prompt: str, max_new_tokens: int, batch_size: int = 1, **kwargs) -> dict:
        if self.mode == "offline":
            return self._generate_offline(prompt, max_new_tokens, batch_size, **kwargs)
        else:
            return self._generate_server(prompt, max_new_tokens, batch_size, **kwargs)

    def _generate_offline(self, prompt: str, max_new_tokens: int, batch_size: int, **kwargs) -> dict:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=kwargs.get("temperature", 0.7),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 0.9),
        )

        prompts = [prompt] * batch_size

        t_start = time.perf_counter()
        outputs = self.engine.generate(prompts, params)
        t_end = time.perf_counter()

        e2e_s = t_end - t_start
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tok_per_s = total_tokens / max(e2e_s, 1e-6)

        return {
            "ttft_s": 0,  # vLLM offline doesn't expose TTFT
            "decode_s": round(e2e_s, 4),
            "e2e_s": round(e2e_s, 4),
            "output_tokens": total_tokens // batch_size,
            "total_new_tokens": total_tokens,
            "tok_per_s": round(tok_per_s, 2),
            "input_len": 0,
            "gpu_memory": {},
        }

    def _generate_server(self, prompt: str, max_new_tokens: int, batch_size: int, **kwargs) -> dict:
        payload = {
            "model": self.model_name,
            "prompt": [prompt] * batch_size,
            "max_tokens": max_new_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_k": kwargs.get("top_k", 50),
            "top_p": kwargs.get("top_p", 0.9),
        }

        t_start = time.perf_counter()
        resp = requests.post(self.server_url, json=payload, timeout=120)
        t_end = time.perf_counter()
        resp.raise_for_status()
        data = resp.json()

        e2e_s = t_end - t_start
        choices = data.get("choices", [])
        total_tokens = sum(c.get("usage", {}).get("completion_tokens", 0) for c in choices)
        if total_tokens == 0:
            total_tokens = sum(len(c.get("text", "").split()) for c in choices)
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
        if self.engine is not None:
            del self.engine
            self.engine = None
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
        print(f"  {self.name} backend teardown complete.")
