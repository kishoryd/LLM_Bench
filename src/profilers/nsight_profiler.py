"""
NVIDIA Nsight Systems / Nsight Compute profiler wrapper.

Nsight works by wrapping the entire Python process, so this profiler
generates shell commands and optionally launches them via subprocess.
"""

import os
import shutil
import subprocess
from contextlib import contextmanager

from src.profilers.base import BaseProfiler


class NsightProfiler(BaseProfiler):
    """
    Wraps Nsight Systems (nsys) and Nsight Compute (ncu) profiling.

    Two modes:
      1. Command generation — build the nsys/ncu command for manual use
      2. Subprocess launch — run the benchmark script under nsys/ncu
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Nsight Systems config
        nsys_cfg = config.get("nsight_systems", {})
        self.nsys_enabled = nsys_cfg.get("enabled", True)
        self.nsys_output_dir = nsys_cfg.get("output_dir", "results/nsight/")
        self.nsys_trace = nsys_cfg.get("trace", ["cuda", "nvtx", "osrt"])
        self.nsys_duration = nsys_cfg.get("duration", 60)
        self.nsys_extra = nsys_cfg.get("extra_args", "--stats=true")

        # Nsight Compute config
        ncu_cfg = config.get("nsight_compute", {})
        self.ncu_enabled = ncu_cfg.get("enabled", False)
        self.ncu_output_dir = ncu_cfg.get("output_dir", "results/ncu/")
        self.ncu_metrics = ncu_cfg.get("metrics", [])
        self.ncu_kernel_filter = ncu_cfg.get("kernel_filter", "")

        # Check availability
        self.nsys_path = shutil.which("nsys")
        self.ncu_path = shutil.which("ncu")

    @contextmanager
    def profile(self, run_name: str = "default"):
        """
        Context manager — for Nsight, this is a no-op wrapper.
        Nsight profiles the entire process, not individual blocks.

        Use build_nsys_command() or launch() instead for full profiling.
        """
        print(f"    ⚠️  Nsight profiles the entire process, not individual code blocks.")
        print(f"    Use `profiler.build_nsys_command(script, args)` to generate the command.")
        yield None

    def build_nsys_command(self, script: str, script_args: str = "", run_name: str = "profile") -> str:
        """
        Build the nsys profile command.

        Args:
            script: Path to the Python script to profile.
            script_args: Arguments to pass to the script.
            run_name: Name for the output report.

        Returns:
            Full nsys command as a string.
        """
        if not self.nsys_path:
            raise FileNotFoundError(
                "nsys not found. Install NVIDIA Nsight Systems:\n"
                "  https://developer.nvidia.com/nsight-systems"
            )

        os.makedirs(self.nsys_output_dir, exist_ok=True)
        output_path = os.path.join(self.nsys_output_dir, run_name)
        trace_str = ",".join(self.nsys_trace)

        cmd = (
            f"nsys profile "
            f"--trace={trace_str} "
            f"--duration={self.nsys_duration} "
            f"--output={output_path} "
            f"{self.nsys_extra} "
            f"python {script} {script_args}"
        )
        return cmd.strip()

    def build_ncu_command(self, script: str, script_args: str = "", run_name: str = "kernel_profile") -> str:
        """
        Build the ncu (Nsight Compute) command for kernel-level profiling.

        Returns:
            Full ncu command as a string.
        """
        if not self.ncu_path:
            raise FileNotFoundError(
                "ncu not found. Install NVIDIA Nsight Compute:\n"
                "  https://developer.nvidia.com/nsight-compute"
            )

        os.makedirs(self.ncu_output_dir, exist_ok=True)
        output_path = os.path.join(self.ncu_output_dir, f"{run_name}.ncu-rep")

        cmd = f"ncu --output {output_path}"
        if self.ncu_metrics:
            cmd += f" --metrics {','.join(self.ncu_metrics)}"
        if self.ncu_kernel_filter:
            cmd += f" --kernel-name {self.ncu_kernel_filter}"
        cmd += f" python {script} {script_args}"
        return cmd.strip()

    def launch(self, script: str, script_args: str = "", run_name: str = "profile") -> int:
        """
        Launch the benchmark script under nsys profiling.

        Returns:
            Process return code.
        """
        cmd = self.build_nsys_command(script, script_args, run_name)
        print(f"  🚀 Running: {cmd}")
        result = subprocess.run(cmd, shell=True)
        return result.returncode

    def export(self, output_dir: str):
        """Print locations of generated reports."""
        print(f"\n  📊 Nsight Reports:")
        if self.nsys_enabled:
            print(f"    Systems : {self.nsys_output_dir}")
        if self.ncu_enabled:
            print(f"    Compute : {self.ncu_output_dir}")
        print(f"    Open with: nsys-ui <file>.nsys-rep")
        print(f"              ncu-ui  <file>.ncu-rep")

    def print_commands(self, script: str = "scripts/run_bench.py", script_args: str = ""):
        """Print ready-to-copy profiling commands."""
        print(f"\n  ── Nsight Profiling Commands ──")
        if self.nsys_enabled and self.nsys_path:
            print(f"\n  Nsight Systems (timeline + API trace):")
            print(f"    {self.build_nsys_command(script, script_args)}")
        if self.ncu_enabled and self.ncu_path:
            print(f"\n  Nsight Compute (kernel-level):")
            print(f"    {self.build_ncu_command(script, script_args)}")
        print()
