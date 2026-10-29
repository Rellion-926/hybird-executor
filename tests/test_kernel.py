import unittest

from hybrid_executor.kernel import inspect_plugin_kernel
from hybrid_executor.manifests import get_plugin


class KernelTest(unittest.TestCase):
    def test_vector_add_prebuilt(self):
        info = inspect_plugin_kernel(get_plugin("vector_add_mlir"), "v69")
        self.assertEqual(info.machine, 0xA4)
        self.assertEqual(info.object_type, 1)
        self.assertIn("vector_add_mlir", info.defined_symbols)
        self.assertEqual(info.undefined_symbols, ())

    def test_fused_prebuilt_allowlist(self):
        info = inspect_plugin_kernel(get_plugin("fused_qknorm_rope"), "v69")
        self.assertIn("fused_qknorm_rope", info.defined_symbols)
        self.assertEqual(set(info.undefined_symbols), {"free", "malloc"})


if __name__ == "__main__":
    unittest.main()
