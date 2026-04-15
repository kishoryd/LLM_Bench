"""
Background GPU utilisation poller — samples GPU % every 100ms.
"""

import time
import threading
import statistics


class GPUStatsSampler:
    """Polls GPU utilisation in a background thread using pynvml."""

    def __init__(self, device_ids: list[int]):
        self.device_ids = device_ids
        self._util: dict[int, list[float]] = {i: [] for i in device_ids}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        """Begin sampling."""
        self._stop.clear()
        self._util = {i: [] for i in self.device_ids}
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        """Stop sampling and return aggregated stats."""
        self._stop.set()
        if self._thread:
            self._thread.join()
        return {
            f"GPU:{i}": {
                "avg_util_pct": round(statistics.mean(v), 1) if v else 0,
                "max_util_pct": round(max(v), 1) if v else 0,
                "samples": len(v),
            }
            for i, v in self._util.items()
        }

    def _poll(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handles = {
                i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.device_ids
            }
            while not self._stop.is_set():
                for i, h in handles.items():
                    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    self._util[i].append(util)
                time.sleep(0.1)
        except Exception:
            # Fallback: record zeros if pynvml unavailable
            while not self._stop.is_set():
                for i in self.device_ids:
                    self._util[i].append(0)
                time.sleep(0.1)
