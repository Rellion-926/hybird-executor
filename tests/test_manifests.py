import tempfile
import unittest
from pathlib import Path

from hybrid_executor.errors import ManifestError
from hybrid_executor.manifests import discover_plugins, load_manifest


class ManifestTest(unittest.TestCase):
    def test_public_examples(self):
        plugins = discover_plugins()
        self.assertIn("custom_add", plugins)
        self.assertIn("vector_add_mlir", plugins)
        self.assertIn("fused_qknorm_rope", plugins)
        self.assertEqual(plugins["custom_add"].kind, "cpp")
        self.assertEqual(plugins["vector_add_mlir"].kind, "mlir")

    def test_rejects_path_traversal(self):
        text = '''schema_version = 1
[plugin]
name = "Bad"
kind = "cpp"
package_name = "BadPackage"
interface_provider = "BadProvider"
op_type = "BadOp"
[target]
supported = ["v69"]
[graph]
generator = "../escape.py"
custom_domain = "Bad"
[op_package]
xml = "config/x.xml"
interface_source = "src/x.cpp"
implementation_source = "src/y.cpp"
[validation]
reference = "reference/r.py"
atol = 0.0
rtol = 0.0
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plugin.toml"
            path.write_text(text)
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_rejects_negative_tolerance(self):
        original = Path("examples/custom_add/plugin.toml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("graph/generate_onnx.py", "config/CustomOpPackage.xml", "src/PackageInterface.cpp", "src/ops/CustomOp.cpp", "reference/reference.py"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            path = root / "plugin.toml"
            path.write_text(original.replace("atol = 1.0e-6", "atol = -1.0"))
            with self.assertRaises(ManifestError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
