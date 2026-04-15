"""
Runner — orchestrates backends, profilers, and benchmark sweeps.

This is the core engine that:
  1. Loads configs
  2. Initialises the selected backend
  3. Optionally wraps runs with a profiler
  4. Sweeps batch sizes across context tiers
  5. Collects and returns all results
"""

import time
import statistics
import torch

from src.backends import get_backend, ALL_BACKENDS
from src.profilers import get_profiler
from src.models import ModelRegistry, build_context_tiers
from src.gpu import GPUStatsSampler, gpu_memory_snapshot
from src.utils.config import load_bench_config, load_backend_config, load_profiler_config


class BenchmarkRunner:
    """
    Main benchmark orchestrator.

    Usage:
        runner = BenchmarkRunner(
            model_name="param2_17b",
            backend_names=["native", "vllm"],
            profiler_name="pytorch",
            num_gpus=1,
            quantize="none",
            flash_attn=False,
        )
        results = runner.run()
    """

    def __init__(
        self,
        model_name: str,
        backend_names: list[str],
        profiler_name: str | None = None,
        num_gpus: int = 1,
        quantize: str = "none",
        flash_attn: bool = False,
        batch_sizes: list[int] | None = None,
        bench_config_path: str | None = None,
    ):
        self.model_name = model_name
        self.backend_names = backend_names
        self.profiler_name = profiler_name
        self.num_gpus = num_gpus
        self.quantize = quantize
        self.flash_attn = flash_attn

        # Load configs
        self.bench_config = load_bench_config(bench_config_path)
        self.batch_sizes = batch_sizes or self.bench_config.get("benchmark", {}).get(
            "batch_sizes", [1, 2, 4, 8, 16, 32]
        )
        self.warmup_runs = self.bench_config.get("benchmark", {}).get("warmup_runs", 1)
        self.measure_runs = self.bench_config.get("benchmark", {}).get("measure_runs", 3)
        self.gen_config = self.bench_config.get("generation", {})

        # Model registry
        self.registry = ModelRegistry()
        self.model_config = self.registry.get(model_name)

        # Profiler
        self.profiler = None
        if profiler_name and profiler_name != "none":
            prof_config = load_profiler_config(
                "pytorch_profiler" if profiler_name == "pytorch" else profiler_name
            )
            self.profiler = get_profiler(profiler_name, prof_config)

    def run(self) -> list[dict]:
        """Run benchmarks across all specified backends. Returns list of result dicts."""
        all_results = []

        for backend_name in self.backend_names:
            print(f"\n{'=' * 70}")
            print(f"  Backend: {backend_name}")
            print(f"{'=' * 70}")

            try:
                result = self._run_backend(backend_name)
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ Backend '{backend_name}' failed: {e}")
                all_results.append({
                    "backend": backend_name,
                    "model": self.model_name,
                    "error": str(e),
                })

        return all_results

    def _run_backend(self, backend_name: str) -> dict:
        """Run the full benchmark for a single backend."""
        backend_config = load_backend_config(backend_name)
        backend = get_backend(backend_name, backend_config)

        # Setup
        backend.setup(
            self.model_config,
            num_gpus=self.num_gpus,
            quantize=self.quantize,
            flash_attn=self.flash_attn,
        )

        # Build context tiers
        tokenizer = self.registry.load_tokenizer(self.model_name)
        tiers = build_context_tiers(tokenizer)

        device_ids = list(range(self.num_gpus))
        result = {
            "backend": backend_name,
            "model": self.model_name,
            "num_gpus": self.num_gpus,
            "load_config": getattr(backend, "load_meta", {}),
            "tiers": [],
        }

        for tier in tiers:
            tier_result = {
                "label": tier["label"],
                "target_input": tier["target_input"],
                "max_new_tokens": tier["max_new_tokens"],
                "batch_sweeps": self._sweep_batch_sizes(
                    backend, tier, device_ids
                ),
            }
            result["tiers"].append(tier_result)

        # Final memory snapshot (only for native backend)
        try:
            result["gpu_memory_final"] = gpu_memory_snapshot(device_ids)
        except Exception:
            result["gpu_memory_final"] = {}

        backend.teardown()
        return result

    def _sweep_batch_sizes(self, backend, tier: dict, device_ids: list) -> list:
        """Sweep batch sizes for one context tier."""
        label = tier["label"]
        prompt = tier["prompt"]
        max_new_tokens = tier["max_new_tokens"]
        results = []
        sampler = GPUStatsSampler(device_ids)

        print(f"\n  ── {label} {'─' * 40}")

        for bs in self.batch_sizes:
            print(f"\n     batch_size={bs}")

            # Warmup
            try:
                for _ in range(self.warmup_runs):
                    backend.generate(
                        prompt, max_new_tokens, bs,
                        **self.gen_config,
                    )
            except (torch.cuda.OutOfMemoryError, Exception) as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    print(f"     ⛔ OOM during warmup at batch_size={bs}")
                    results.append({"batch_size": bs, "oom": True, "phase": "warmup"})
                    torch.cuda.empty_cache()
                    break
                raise

            # Measure
            run_records = []
            oom_hit = False

            for r_idx in range(self.measure_runs):
                try:
                    sampler.start()

                    # Optionally wrap with profiler
                    if self.profiler and bs == 1 and r_idx == 0:
                        run_name = f"{label.split()[0]}_{bs}"
                        with self.profiler.profile(run_name=run_name):
                            r = backend.generate(
                                prompt, max_new_tokens, bs,
                                **self.gen_config,
                            )
                    else:
                        r = backend.generate(
                            prompt, max_new_tokens, bs,
                            **self.gen_config,
                        )

                    gpu_stats = sampler.stop()
                    r["gpu_util"] = gpu_stats
                    run_records.append(r)

                    print(
                        f"       run {r_idx + 1}: "
                        f"TTFT={r['ttft_s']:.3f}s  "
                        f"e2e={r['e2e_s']:.3f}s  "
                        f"throughput={r['tok_per_s']:.1f} tok/s  "
                        f"output={r['output_tokens']} tok"
                    )

                except (torch.cuda.OutOfMemoryError, Exception) as e:
                    sampler.stop()
                    if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                        print(f"       ⛔ OOM at run {r_idx + 1}")
                        results.append({"batch_size": bs, "oom": True, "phase": f"run_{r_idx + 1}"})
                        oom_hit = True
                        torch.cuda.empty_cache()
                        break
                    raise

            if oom_hit:
                break

            if not run_records:
                continue

            n = len(run_records)

            def avg(key):
                return round(sum(r[key] for r in run_records) / n, 4)

            def std(key):
                vals = [r[key] for r in run_records]
                return round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)

            agg = {
                "batch_size": bs,
                "oom": False,
                "input_tokens": run_records[0].get("input_len", 0),
                "output_tokens_avg": avg("output_tokens"),
                "ttft_avg_s": avg("ttft_s"),
                "ttft_std_s": std("ttft_s"),
                "e2e_avg_s": avg("e2e_s"),
                "e2e_std_s": std("e2e_s"),
                "decode_avg_s": avg("decode_s"),
                "tok_per_s_avg": avg("tok_per_s"),
                "tok_per_s_std": std("tok_per_s"),
                "gpu_memory": run_records[-1].get("gpu_memory", {}),
                "gpu_util": run_records[-1].get("gpu_util", {}),
            }

            print(
                f"     AVG → TTFT={agg['ttft_avg_s']:.3f}s  "
                f"e2e={agg['e2e_avg_s']:.3f}s  "
                f"throughput={agg['tok_per_s_avg']:.1f} tok/s"
            )

            results.append(agg)

        return results
