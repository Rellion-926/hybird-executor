from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

from .errors import BuildError


@dataclass(frozen=True)
class CommandResult:
    argv: Sequence[str]
    returncode: int
    stdout: str
    stderr: str


def render(argv: Iterable[object]) -> str:
    import shlex
    return " ".join(shlex.quote(str(item)) for item in argv)


def run(
    argv: Sequence[object],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    dry_run: bool = False,
    timeout: int = 600,
) -> CommandResult:
    command = [str(item) for item in argv]
    if dry_run:
        return CommandResult(command, 0, render(command) + "\n", "")
    merged = dict(os.environ)
    if env:
        merged.update(env)
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=merged,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    result = CommandResult(command, completed.returncode, completed.stdout, completed.stderr)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-30:])
        raise BuildError(f"command failed ({result.returncode}): {render(command)}\n{tail}")
    return result


def write_record(path: Path, result: CommandResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "argv": list(result.argv),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }, indent=2) + "\n", encoding="utf-8")
