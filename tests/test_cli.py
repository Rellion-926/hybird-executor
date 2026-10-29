import contextlib
import io
import unittest

from hybrid_executor.cli import main


class CliTest(unittest.TestCase):
    def capture(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(argv)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_list(self):
        status, output, _ = self.capture(["list"])
        self.assertEqual(status, 0)
        self.assertIn("vector_add_mlir", output)
        self.assertIn("fused_qknorm_rope", output)

    def test_prebuilt_dry_run_without_mlir_root(self):
        status, output, error = self.capture([
            "build", "vector_add_mlir", "--target", "v69",
            "--kernel-mode", "prebuilt", "--dry-run",
        ])
        self.assertEqual(status, 0, error)
        self.assertIn("generate_onnx.py", output)
        self.assertNotIn("HEXAGON_MLIR_ROOT", output)

    def test_source_requires_mlir_root(self):
        status, _, error = self.capture([
            "build", "vector_add_mlir", "--target", "v69",
            "--kernel-mode", "source", "--dry-run",
        ])
        self.assertEqual(status, 2)
        self.assertIn("HEXAGON_MLIR_ROOT", error)


if __name__ == "__main__":
    unittest.main()
