"""
PyTorch Profiler — operator-level breakdown, memory, FLOPS, Chrome traces.
"""

import os
from contextlib import contextmanager

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from src.profilers.base import BaseProfiler


class PyTorchProfiler(BaseProfiler):
    """torch.profiler wrapper with Chrome trace and TensorBoard export."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._traces: list[str] = []
        self._last_profiler = None

        # Parse config
        self.output_dir = config.get("output_dir", "results/torch_profiler/")
        sched = config.get("schedule", {})
        self.wait = sched.get("wait", 1)
        self.warmup = sched.get("warmup", 1)
        self.active = sched.get("active", 3)
        self.repeat = sched.get("repeat", 1)

        # Activities
        activity_names = config.get("activities", ["cpu", "cuda"])
        self.activities = []
        if "cpu" in activity_names:
            self.activities.append(ProfilerActivity.CPU)
        if "cuda" in activity_names:
            self.activities.append(ProfilerActivity.CUDA)

        self.record_shapes = config.get("record_shapes", True)
        self.profile_memory = config.get("profile_memory", True)
        self.with_stack = config.get("with_stack", True)
        self.with_flops = config.get("with_flops", True)
        self.export_chrome = config.get("export_chrome_trace", True)
        self.export_tb = config.get("export_tensorboard", False)
        self.tb_dir = config.get("tensorboard_dir", "results/tb_logs/")

    @contextmanager
    def profile(self, run_name: str = "default"):
        """
        Wrap an inference call with PyTorch profiling.

        Usage:
            with profiler.profile("batch_1_short"):
                result = backend.generate(...)
        """
        os.makedirs(self.output_dir, exist_ok=True)

        prof = profile(
            activities=self.activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
        )

        with prof as p:
            with record_function(run_name):
                yield p
            # Sync GPU so all CUDA kernels are captured before the profiler stops
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        self._last_profiler = p
        self._on_trace_ready(run_name)(p)

        # Print summary
        print(f"\n    ── PyTorch Profiler Summary ({run_name}) ──")
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    def _on_trace_ready(self, run_name: str):
        """Export Chrome trace once; optionally symlink it for TensorBoard."""
        import shutil

        def handler(p):
            os.makedirs(self.output_dir, exist_ok=True)
            trace_path = os.path.join(self.output_dir, f"{run_name}_trace.json")
            p.export_chrome_trace(trace_path)
            self._traces.append(trace_path)
            print(f"    Chrome trace: {trace_path}")
            if self.export_tb:
                os.makedirs(self.tb_dir, exist_ok=True)
                tb_path = os.path.join(self.tb_dir, f"{run_name}.json")
                shutil.copy2(trace_path, tb_path)
        return handler

    def export(self, output_dir: str):
        """Copy the last trace to output_dir if not already there."""
        if self._traces:
            import shutil
            src = self._traces[-1]
            dst = os.path.join(output_dir, "final_trace.json")
            if os.path.abspath(src) != os.path.abspath(dst):
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  Final trace exported: {dst}")

    def get_summary_table(self, sort_by: str = "cuda_time_total", row_limit: int = 20) -> str:
        """Return the profiler summary as a string."""
        if self._last_profiler:
            return self._last_profiler.key_averages().table(
                sort_by=sort_by, row_limit=row_limit
            )
        return "No profiling data available."
