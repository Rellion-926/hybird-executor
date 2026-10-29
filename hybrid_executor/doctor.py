from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .config import ToolchainConfig


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _path_check(name: str, path: Optional[Path], required: bool) -> Check:
    if path is None:
        return Check(name, "FAIL" if required else "SKIP", "not configured")
    if not path.exists():
        return Check(name, "FAIL", f"configured path does not exist: {path}")
    return Check(name, "PASS", str(path))


def run_checks(config: ToolchainConfig, kernel_mode: str, device: bool = False) -> List[Check]:
    checks = [
        Check("Python", "PASS" if sys.version_info >= (3, 9) else "FAIL", sys.version.split()[0]),
        _path_check("QAIRT/QNN SDK", config.qairt_sdk_root, True),
        _path_check("Hexagon SDK", config.hexagon_sdk_root, True),
        _path_check("Hexagon Tools", config.hexagon_tools_root, True),
        _path_check("Android NDK", config.android_ndk_root, True),
        _path_check("Hexagon-MLIR", config.hexagon_mlir_root, kernel_mode == "source"),
        _path_check("Hexagon-MLIR Python", config.hexagon_mlir_python, kernel_mode == "source"),
    ]
    adb = shutil.which("adb")
    checks.append(Check("ADB", "PASS" if adb else ("FAIL" if device else "SKIP"), adb or "not found"))
    if device and adb:
        from .process import run
        try:
            argv = [adb]
            if config.adb_serial:
                argv += ["-s", config.adb_serial]
            argv += ["get-state"]
            state = run(argv, timeout=15).stdout.strip()
            checks.append(Check("ADB device", "PASS" if state == "device" else "FAIL", state))
        except Exception as exc:  # user-facing diagnostic aggregation
            checks.append(Check("ADB device", "FAIL", str(exc).splitlines()[-1]))
    return checks


def format_checks(checks: Iterable[Check]) -> str:
    return "\n".join(f"[{item.status}] {item.name}: {item.detail}" for item in checks)
