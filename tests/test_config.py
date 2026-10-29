import os
import tempfile
import unittest
from pathlib import Path

from hybrid_executor.config import load_config


class ConfigTest(unittest.TestCase):
    def test_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toolchains.toml"
            path.write_text('[toolchains]\nqairt_sdk_root = "/file/qairt"\n')
            config = load_config(path, {"qairt_sdk_root": "/cli/qairt"}, {"QAIRT_SDK_ROOT": "/env/qairt"})
            self.assertEqual(config.qairt_sdk_root, Path("/cli/qairt"))

    def test_file_over_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toolchains.toml"
            path.write_text('[toolchains]\nhexagon_sdk_root = "/file/sdk"\n')
            config = load_config(path, environ={"HEXAGON_SDK_ROOT": "/env/sdk"})
            self.assertEqual(config.hexagon_sdk_root, Path("/file/sdk"))

    def test_no_machine_default_and_redaction(self):
        config = load_config(environ={"ADB_SERIAL": "secret-device", "QAIRT_SDK_ROOT": "/private/user/qairt"})
        self.assertIsNone(config.hexagon_sdk_root)
        redacted = config.redacted()
        self.assertEqual(redacted["adb_serial"], "<redacted-device>")
        self.assertNotIn("/private/user", redacted["qairt_sdk_root"])


if __name__ == "__main__":
    unittest.main()
