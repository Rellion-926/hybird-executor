"""Small TOML loader with a dependency-free fallback for Python 3.9/3.10.

The fallback intentionally supports only the public manifest/config subset:
tables, strings, integers, floats, booleans, and flat arrays.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict

try:  # Python 3.11+
    import tomllib as _toml
except ImportError:  # pragma: no cover - depends on interpreter
    _toml = None


_TABLE = re.compile(r"^\[([A-Za-z0-9_.-]+)\]$")
_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _value(text: str) -> Any:
    text = text.strip()
    if text in {"true", "false"}:
        return text == "true"
    if text.startswith('"') or text.startswith("'") or text.startswith("["):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"unsupported TOML value: {text}") from exc
    try:
        return float(text) if any(c in text.lower() for c in (".", "e")) else int(text)
    except ValueError as exc:
        raise ValueError(f"unsupported TOML value: {text}") from exc


def _fallback_load(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    current = root
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _TABLE.match(line)
        if match:
            current = root
            for part in match.group(1).split("."):
                if not _KEY.match(part):
                    raise ValueError(f"invalid table name at line {number}")
                current = current.setdefault(part, {})
            continue
        if "=" not in line:
            raise ValueError(f"invalid TOML line {number}: {line}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not _KEY.match(key):
            raise ValueError(f"invalid key at line {number}: {key}")
        if key in current:
            raise ValueError(f"duplicate key at line {number}: {key}")
        current[key] = _value(value.split(" #", 1)[0])
    return root


def loads(text: str) -> Dict[str, Any]:
    if _toml is not None:
        return _toml.loads(text)
    return _fallback_load(text)


def load(path: Path) -> Dict[str, Any]:
    return loads(path.read_text(encoding="utf-8"))
