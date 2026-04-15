"""Tests for backend interface and registry."""

import unittest


class TestBackendRegistry(unittest.TestCase):
    def test_all_backends_listed(self):
        from src.backends import ALL_BACKENDS
        expected = ["native", "vllm", "sglang", "triton", "ollama"]
        self.assertEqual(ALL_BACKENDS, expected)

    def test_unknown_backend_raises(self):
        from src.backends import get_backend
        with self.assertRaises(ValueError):
            get_backend("nonexistent", {})

    def test_native_backend_instantiates(self):
        from src.backends import get_backend
        backend = get_backend("native", {"name": "native"})
        self.assertEqual(backend.name, "native")


class TestBaseBackend(unittest.TestCase):
    def test_abstract_methods(self):
        from src.backends.base import BaseBackend
        # Cannot instantiate abstract class
        with self.assertRaises(TypeError):
            BaseBackend({"name": "test"})


if __name__ == "__main__":
    unittest.main()
