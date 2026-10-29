import unittest
from pathlib import Path

from hybrid_executor.config import ToolchainConfig
from hybrid_executor.deployment import DEVICE_ROOT, clean_build, plan_bundle
from hybrid_executor.manifests import get_plugin


class DeploymentTest(unittest.TestCase):
    def test_bundle_plan_has_model_packages_and_v69_runtime(self):
        plugin = get_plugin("vector_add_mlir")
        config = ToolchainConfig(
            qairt_sdk_root=Path("/opt/qairt"),
            android_ndk_root=Path("/opt/ndk"),
        )
        relative = {item.relative.as_posix() for item in plan_bundle(plugin, "v69", config)}
        self.assertIn("runtime/qnn-net-run", relative)
        self.assertIn("runtime/libQnnHtpV69Skel.so", relative)
        self.assertIn("model/libvector_add_mlir.so", relative)
        self.assertIn("op_packages/arm/libQnnMlirBridgePackage.so", relative)
        self.assertIn("op_packages/dsp/libQnnMlirBridgePackage.so", relative)

    def test_device_root_is_narrow(self):
        self.assertTrue(DEVICE_ROOT.startswith("/data/local/tmp/"))
        self.assertNotEqual(DEVICE_ROOT, "/data/local/tmp")

    def test_clean_dry_run_does_not_delete(self):
        plugin = get_plugin("custom_add")
        result = clean_build(plugin, "v69", dry_run=True)
        self.assertIn("build/custom_add/v69", result)


if __name__ == "__main__":
    unittest.main()
