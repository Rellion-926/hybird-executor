from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

from .errors import ConfigurationError
from .toml_compat import load as load_toml


ROOT = Path(__file__).resolve().parents[1]

_KEYS = {
    "qairt_sdk_root": "QAIRT_SDK_ROOT",
    "hexagon_sdk_root": "HEXAGON_SDK_ROOT",
    "hexagon_tools_root": "HEXAGON_TOOLS_ROOT",
    "android_ndk_root": "ANDROID_NDK_ROOT",
    "hexagon_mlir_root": "HEXAGON_MLIR_ROOT",
    "hexagon_mlir_python": "HEXAGON_MLIR_PYTHON",
    "adb_serial": "ADB_SERIAL",
}


@dataclass(frozen=True)
class ToolchainConfig:
    qairt_sdk_root: Optional[Path] = None
    hexagon_sdk_root: Optional[Path] = None
    hexagon_tools_root: Optional[Path] = None
    android_ndk_root: Optional[Path] = None
    hexagon_mlir_root: Optional[Path] = None
    hexagon_mlir_python: Optional[Path] = None
    adb_serial: Optional[str] = None

    def require(self, name: str) -> Path:
        value = getattr(self, name)
        if not isinstance(value, Path):
            raise ConfigurationError(f"missing required configuration: {name}")
        return value

    def redacted(self) -> Dict[str, Optional[str]]:
        result: Dict[str, Optional[str]] = {}
        for name in _KEYS:
            value = getattr(self, name)
            if isinstance(value, Path):
                result[name] = f"<configured>/{value.name}"
            elif name == "adb_serial" and value:
                result[name] = "<redacted-device>"
            else:
                result[name] = value
        return result


def _path(value: object) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def load_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, object]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> ToolchainConfig:
    environment = dict(os.environ if environ is None else environ)
    file_values: Dict[str, object] = {}
    if config_path is not None:
        if not config_path.is_file():
            raise ConfigurationError(f"configuration file does not exist: {config_path}")
        document = load_toml(config_path)
        file_values.update(document.get("toolchains", {}))
        file_values.update(document.get("device", {}))
    cli_values = dict(overrides or {})

    values: Dict[str, object] = {}
    for field, env_name in _KEYS.items():
        value = cli_values.get(field)
        if value is None:
            value = file_values.get(field)
        if value is None:
            value = environment.get(env_name)
        values[field] = value

    return ToolchainConfig(
        qairt_sdk_root=_path(values["qairt_sdk_root"]),
        hexagon_sdk_root=_path(values["hexagon_sdk_root"]),
        hexagon_tools_root=_path(values["hexagon_tools_root"]),
        android_ndk_root=_path(values["android_ndk_root"]),
        hexagon_mlir_root=_path(values["hexagon_mlir_root"]),
        hexagon_mlir_python=_path(values["hexagon_mlir_python"]),
        adb_serial=str(values["adb_serial"]).strip() if values["adb_serial"] else None,
    )
