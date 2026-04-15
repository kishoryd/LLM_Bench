"""Tests for profiler interface and registry."""

import unittest


class TestProfilerRegistry(unittest.TestCase):
    def test_none_returns_none(self):
        from src.profilers import get_profiler
        self.assertIsNone(get_profiler("none", {}))
        self.assertIsNone(get_profiler(None, {}))

    def test_unknown_profiler_raises(self):
        from src.profilers import get_profiler
        with self.assertRaises(ValueError):
            get_profiler("nonexistent", {})

    def test_pytorch_profiler_instantiates(self):
        from src.profilers import get_profiler
        p = get_profiler("pytorch", {"name": "pytorch_profiler"})
        self.assertIsNotNone(p)
        self.assertEqual(p.name, "pytorch_profiler")


class TestBaseProfiler(unittest.TestCase):
    def test_abstract_methods(self):
        from src.profilers.base import BaseProfiler
        with self.assertRaises(TypeError):
            BaseProfiler({"name": "test"})


if __name__ == "__main__":
    unittest.main()
