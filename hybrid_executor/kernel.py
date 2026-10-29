from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set

from .errors import BuildError
from .manifests import PluginManifest


@dataclass(frozen=True)
class ObjectInfo:
    path: Path
    sha256: str
    elf_class: int
    endian: str
    machine: int
    object_type: int
    defined_symbols: Sequence[str]
    undefined_symbols: Sequence[str]


def _cstring(data: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\0", offset)
    if end < 0:
        end = len(data)
    return data[offset:end].decode("utf-8", errors="replace")


def inspect_elf(path: Path) -> ObjectInfo:
    data = path.read_bytes()
    if len(data) < 52 or data[:4] != b"\x7fELF":
        raise BuildError(f"not an ELF object: {path}")
    elf_class = data[4]
    encoding = data[5]
    if elf_class != 1 or encoding != 1:
        raise BuildError(f"expected ELF32 little-endian object: {path}")
    endian = "<"
    object_type, machine = struct.unpack_from(endian + "HH", data, 16)
    shoff = struct.unpack_from(endian + "I", data, 32)[0]
    shentsize, shnum = struct.unpack_from(endian + "HH", data, 46)
    if shentsize < 40 or shoff + shentsize * shnum > len(data):
        raise BuildError(f"invalid ELF section table: {path}")
    sections = []
    for index in range(shnum):
        offset = shoff + index * shentsize
        values = struct.unpack_from(endian + "IIIIIIIIII", data, offset)
        sections.append({
            "type": values[1], "offset": values[4], "size": values[5],
            "link": values[6], "entsize": values[9],
        })
    defined: Set[str] = set()
    undefined: Set[str] = set()
    for section in sections:
        if section["type"] not in {2, 11}:  # SYMTAB or DYNSYM
            continue
        link = section["link"]
        if link >= len(sections):
            continue
        string_section = sections[link]
        strings = data[string_section["offset"]:string_section["offset"] + string_section["size"]]
        entry_size = section["entsize"] or 16
        for pos in range(section["offset"], section["offset"] + section["size"], entry_size):
            if pos + 16 > len(data):
                break
            name_offset, _, _, info, _, shndx = struct.unpack_from(endian + "IIIBBH", data, pos)
            name = _cstring(strings, name_offset)
            if not name or (info & 0xF) in {3, 4}:  # section/file symbols
                continue
            (undefined if shndx == 0 else defined).add(name)
    return ObjectInfo(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        elf_class=32,
        endian="little",
        machine=machine,
        object_type=object_type,
        defined_symbols=tuple(sorted(defined)),
        undefined_symbols=tuple(sorted(undefined)),
    )


def inspect_plugin_kernel(plugin: PluginManifest, target: str) -> ObjectInfo:
    if plugin.kernel is None:
        raise BuildError(f"plugin {plugin.name} has no MLIR kernel")
    if target not in plugin.kernel.prebuilt:
        raise BuildError(f"plugin {plugin.name} has no prebuilt kernel for {target}")
    path = plugin.kernel.prebuilt[target]
    if not path.is_file():
        raise BuildError(f"missing prebuilt kernel: {path}")
    info = inspect_elf(path)
    if info.machine != 0xA4 or info.object_type != 1:
        raise BuildError(f"expected Qualcomm DSP6 relocatable ELF, got machine={info.machine} type={info.object_type}")
    if plugin.kernel.symbol not in info.defined_symbols:
        raise BuildError(f"missing declared kernel symbol {plugin.kernel.symbol!r}")
    unexpected = sorted(set(info.undefined_symbols) - set(plugin.kernel.undefined_symbols))
    if unexpected:
        raise BuildError(f"unexpected undefined kernel symbols: {', '.join(unexpected)}")
    abi = json.loads(plugin.kernel.abi.read_text(encoding="utf-8"))
    if abi.get("symbol") != plugin.kernel.symbol or abi.get("target") != target:
        raise BuildError("kernel ABI symbol/target does not match plugin manifest")
    sums = path.parent.parent.parent / "SHA256SUMS"
    if not sums.is_file():
        raise BuildError(f"missing kernel SHA256SUMS: {sums}")
    expected = None
    for line in sums.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and Path(parts[1].lstrip("*")) == Path(target) / path.name:
            expected = parts[0]
    if expected != info.sha256:
        raise BuildError(f"kernel hash mismatch: expected {expected}, got {info.sha256}")
    return info


def format_object(info: ObjectInfo) -> str:
    return json.dumps({
        "path": str(info.path),
        "sha256": info.sha256,
        "elf_class": info.elf_class,
        "endian": info.endian,
        "machine": info.machine,
        "object_type": info.object_type,
        "defined_symbols": list(info.defined_symbols),
        "undefined_symbols": list(info.undefined_symbols),
    }, indent=2, sort_keys=True)
