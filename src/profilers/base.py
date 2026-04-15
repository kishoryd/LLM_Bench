"""
Abstract base class for profilers.

Profilers wrap around benchmark runs to capture performance traces.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager


class BaseProfiler(ABC):
    """
    Interface for profilers.

    Usage:
        profiler = SomeProfiler(config)
        with profiler.profile(run_name="batch_1_short"):
            # ... run inference ...
        profiler.export(output_dir)
    """

    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", self.__class__.__name__)

    @abstractmethod
    @contextmanager
    def profile(self, run_name: str = "default"):
        """
        Context manager that wraps an inference run with profiling.

        Args:
            run_name: Label for this profiling run.
        """
        ...

    @abstractmethod
    def export(self, output_dir: str):
        """Export captured traces / reports to the output directory."""
        ...

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"
