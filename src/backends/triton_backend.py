"""
Triton Inference Server backend — gRPC or HTTP client.
"""

import time
import json
import requests
import numpy as np

from src.backends.base import BaseBackend


class TritonBackend(BaseBackend):
    """NVIDIA Triton Inference Server client."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.http_url = None
        self.grpc_url = None
        self.client = None
        self.model_name = None

    def setup(self, model_config: dict, num_gpus: int = 1, **kwargs):
        server_cfg = self.config.get("server", {})
        host = server_cfg.get("host", "localhost")
        http_port = server_cfg.get("http_port", 8000)
        grpc_port = server_cfg.get("grpc_port", 8001)
        self.model_name = model_config["name"].lower().replace("-", "_")

        self.http_url = f"http://{host}:{http_port}"
        self.grpc_url = f"{host}:{grpc_port}"

        # Try gRPC client first, fall back to HTTP
        try:
            import tritonclient.grpc as grpcclient
            self.client = grpcclient.InferenceServerClient(url=self.grpc_url)
            if self.client.is_server_ready():
                print(f"  Triton connected via gRPC at {self.grpc_url}")
                return
        except Exception:
            pass

        # HTTP fallback
        try:
            resp = requests.get(f"{self.http_url}/v2/health/ready", timeout=5)
            resp.raise_for_status()
            self.client = None  # Use HTTP requests directly
            print(f"  Triton connected via HTTP at {self.http_url}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Triton server — "
                f"start it with: ./scripts/start_triton.sh\n"
                f"Error: {e}"
            )

    def generate(self, prompt: str, max_new_tokens: int, batch_size: int = 1, **kwargs) -> dict:
        if self.client is not None:
            return self._generate_grpc(prompt, max_new_tokens, batch_size, **kwargs)
        return self._generate_http(prompt, max_new_tokens, batch_size, **kwargs)

    def _generate_grpc(self, prompt: str, max_new_tokens: int, batch_size: int, **kwargs) -> dict:
        import tritonclient.grpc as grpcclient

        inputs = [
            grpcclient.InferInput("text_input", [batch_size], "BYTES"),
            grpcclient.InferInput("max_tokens", [batch_size], "INT32"),
        ]
        inputs[0].set_data_from_numpy(
            np.array([prompt.encode()] * batch_size, dtype=object)
        )
        inputs[1].set_data_from_numpy(
            np.array([max_new_tokens] * batch_size, dtype=np.int32)
        )

        outputs = [grpcclient.InferRequestedOutput("text_output")]

        t_start = time.perf_counter()
        result = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
        )
        t_end = time.perf_counter()

        e2e_s = t_end - t_start
        output_texts = result.as_numpy("text_output")
        total_tokens = sum(len(t.decode().split()) for t in output_texts)
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

    def _generate_http(self, prompt: str, max_new_tokens: int, batch_size: int, **kwargs) -> dict:
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [batch_size],
                    "datatype": "BYTES",
                    "data": [prompt] * batch_size,
                },
                {
                    "name": "max_tokens",
                    "shape": [batch_size],
                    "datatype": "INT32",
                    "data": [max_new_tokens] * batch_size,
                },
            ],
        }

        t_start = time.perf_counter()
        resp = requests.post(
            f"{self.http_url}/v2/models/{self.model_name}/infer",
            json=payload,
            timeout=120,
        )
        t_end = time.perf_counter()
        resp.raise_for_status()
        data = resp.json()

        e2e_s = t_end - t_start
        output_data = data.get("outputs", [{}])[0].get("data", [])
        total_tokens = sum(len(str(t).split()) for t in output_data)
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
        self.client = None
        print(f"  {self.name} backend teardown complete (server still running).")
