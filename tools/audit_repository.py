#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {
    "qnn-net-run", "libQnnHtp.so", "libQnnHtpPrepare.so",
    "libQnnHtpV69Skel.so", "libQnnHtpV69Stub.so",
}
FORBIDDEN_SUFFIXES = {".so", ".raw", ".log", ".onnx"}
PRIVATE_PATTERNS = [
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"/export/home/[A-Za-z0-9._-]+/"),
    re.compile(r"my_target_[0-9]+"),
]


def tracked() -> list[Path]:
    if (ROOT / ".git").exists():
        result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
        return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]
    # Release archives intentionally have no .git directory.  Mirror the
    # public .gitignore boundary so the audit remains useful after extraction
    # even when the user has generated local build/deployment artifacts.
    excluded_roots = {
        "build", "deploy", "dist", "out", "runs", "executor", "contracts",
        "references", ".venv", "venv", "__pycache__",
    }
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in excluded_roots for part in relative.parts):
            continue
        if path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    errors = []
    for path in tracked():
        relative = path.relative_to(ROOT)
        if path.name.startswith("._") or path.name == ".DS_Store":
            errors.append(f"macOS metadata artifact: {relative}")
        if path.name in FORBIDDEN_NAMES or path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden generated/runtime artifact: {relative}")
        if path.stat().st_size > 2 * 1024 * 1024:
            errors.append(f"tracked file exceeds 2 MiB: {relative}")
        if path.suffix == ".o" and "prebuilt" not in path.parts:
            errors.append(f"unmanifested object file: {relative}")
        if path.is_symlink():
            try:
                path.resolve().relative_to(ROOT)
            except ValueError:
                errors.append(f"symlink escapes repository: {relative}")
        if path.suffix.lower() in {".md", ".py", ".toml", ".json", ".cpp", ".hpp", ".xml"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in PRIVATE_PATTERNS:
                if pattern.search(text):
                    errors.append(f"private path/host pattern in {relative}: {pattern.pattern}")
    if errors:
        print("Repository audit: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Repository audit: PASS ({len(tracked())} public files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
