"""Tests for GPU detection, memory, and sampler modules."""

import unittest


class TestArchMap(unittest.TestCase):
    def test_known_architectures(self):
        from src.gpu.detection import ARCH_MAP
        self.assertEqual(ARCH_MAP[(8, 0)], "Ampere")
        self.assertEqual(ARCH_MAP[(8, 9)], "Ada Lovelace")
        self.assertEqual(ARCH_MAP[(9, 0)], "Hopper")
        self.assertEqual(ARCH_MAP[(7, 0)], "Volta")

    def test_feature_sets(self):
        from src.gpu.detection import FA2_SUPPORTED_ARCHS, BF16_SUPPORTED_ARCHS, FP8_SUPPORTED_ARCHS
        self.assertIn("Ampere", FA2_SUPPORTED_ARCHS)
        self.assertIn("Hopper", FP8_SUPPORTED_ARCHS)
        self.assertNotIn("Volta", FA2_SUPPORTED_ARCHS)
        self.assertNotIn("Ampere", FP8_SUPPORTED_ARCHS)


class TestGPUSampler(unittest.TestCase):
    def test_sampler_lifecycle(self):
        from src.gpu.sampler import GPUStatsSampler
        import time
        sampler = GPUStatsSampler([0])
        sampler.start()
        time.sleep(0.3)
        result = sampler.stop()
        self.assertIn("GPU:0", result)
        self.assertIn("avg_util_pct", result["GPU:0"])


if __name__ == "__main__":
    unittest.main()
