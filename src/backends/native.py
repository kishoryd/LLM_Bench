"""
Native backend — direct HuggingFace transformers inference, no server.
"""

import os
import gc
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.backends.base import BaseBackend
from src.gpu.detection import detect_gpu_info, auto_select_dtype, print_gpu_info
from src.gpu.memory import gpu_memory_snapshot


class NativeBackend(BaseBackend):
    """HuggingFace transformers — direct model loading and generation."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device_ids: list[int] = []
        self.load_meta: dict = {}

    def setup(self, model_config: dict, num_gpus: int = 1, **kwargs):
        quantize = kwargs.get("quantize", self.config.get("optimizations", {}).get("quantization", "none"))
        use_flash_attn = kwargs.get("flash_attn", self.config.get("optimizations", {}).get("flash_attention", False))
        model_path = model_config["path"]

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
        print_gpu_info(num_gpus)

        dtype = auto_select_dtype(num_gpus)
        attn_impl = self._resolve_attention(use_flash_attn)
        quant_cfg, load_dtype = self._build_quant_config(quantize, dtype)
        effective_dtype = load_dtype if load_dtype is not None else dtype

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=model_config.get("trust_remote_code", False),
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Model
        load_kwargs = dict(
            trust_remote_code=model_config.get("trust_remote_code", True),
            device_map="auto",
            local_files_only=True,
            attn_implementation=attn_impl,
        )
        if quant_cfg is not None:
            load_kwargs["quantization_config"] = quant_cfg
            if quantize == "int8":
                load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = effective_dtype

        print(f"\n  Loading {model_config['name']} ...")
        t0 = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        load_time = time.perf_counter() - t0
        self.model.eval()

        self.device_ids = list(range(num_gpus))
        self.load_meta = {
            "attn_impl": attn_impl,
            "dtype": str(effective_dtype),
            "quantize": quantize,
            "load_time_s": round(load_time, 2),
        }
        print(f"  Loaded in {load_time:.1f}s")

    def generate(self, prompt: str, max_new_tokens: int, batch_size: int = 1, **kwargs) -> dict:
        temperature = kwargs.get("temperature", 0.7)
        top_k = kwargs.get("top_k", 50)
        top_p = kwargs.get("top_p", 0.9)

        # Build batch
        conversations = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        ] * batch_size

        encoded = [
            self.tokenizer.apply_chat_template(c, return_tensors="pt", add_generation_prompt=True)[0]
            for c in conversations
        ]
        max_len = max(e.shape[0] for e in encoded)
        padded = torch.stack([
            torch.cat([
                torch.full((max_len - e.shape[0],), self.tokenizer.pad_token_id, dtype=torch.long),
                e,
            ])
            for e in encoded
        ])
        attn_mask = (padded != self.tokenizer.pad_token_id).long()

        first_device = next(self.model.parameters()).device
        padded = padded.to(first_device)
        attn_mask = attn_mask.to(first_device)
        input_len = padded.shape[1]

        # Prefill / TTFT
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = self.model(input_ids=padded, attention_mask=attn_mask)
        torch.cuda.synchronize()
        ttft_s = time.perf_counter() - t_start

        # Full generation
        torch.cuda.synchronize()
        t_gen = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                padded,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        e2e_s = t_end - t_gen
        output_toks = output.shape[1] - input_len
        total_new = output_toks * batch_size
        decode_s = max(e2e_s - ttft_s, 1e-6)
        tok_per_s = total_new / decode_s

        return {
            "ttft_s": round(ttft_s, 4),
            "decode_s": round(decode_s, 4),
            "e2e_s": round(e2e_s, 4),
            "output_tokens": output_toks,
            "total_new_tokens": total_new,
            "tok_per_s": round(tok_per_s, 2),
            "input_len": input_len,
            "gpu_memory": gpu_memory_snapshot(self.device_ids),
        }

    def teardown(self):
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  {self.name} backend teardown complete.")

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_attention(use_flash_attn: bool) -> str:
        if not use_flash_attn:
            return "sdpa"
        gpu_info = detect_gpu_info(0)
        if not gpu_info["fa2_supported"]:
            print(f"  Flash Attn 2 not supported on {gpu_info['arch']} — using SDPA")
            return "sdpa"
        try:
            import flash_attn  # noqa: F401
            print(f"  Flash Attention 2 ✅")
            return "flash_attention_2"
        except ImportError:
            print(f"  flash-attn not installed — using SDPA")
            return "sdpa"

    @staticmethod
    def _build_quant_config(quantize: str, dtype: torch.dtype):
        if quantize == "int4":
            cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            return cfg, None
        elif quantize == "int8":
            cfg = BitsAndBytesConfig(load_in_8bit=True)
            return cfg, None
        else:
            return None, dtype
