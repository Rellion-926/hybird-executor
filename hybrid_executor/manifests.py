from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .config import ROOT
from .errors import ManifestError
from .toml_compat import load as load_toml


_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_TARGET = re.compile(r"^v[0-9]+$")


def _required(table: Dict[str, object], key: str, context: str) -> object:
    if key not in table:
        raise ManifestError(f"missing {context}.{key}")
    return table[key]


def _identifier(value: object, context: str) -> str:
    text = str(value)
    if not _SAFE_NAME.fullmatch(text):
        raise ManifestError(f"invalid identifier for {context}: {text!r}")
    return text


def _relative(root: Path, value: object, context: str) -> Path:
    declared = Path(str(value))
    if declared.is_absolute() or ".." in declared.parts:
        raise ManifestError(f"{context} must be a path inside the plugin: {declared}")
    resolved = (root / declared).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ManifestError(f"{context} escapes plugin root: {declared}") from exc
    return resolved


@dataclass(frozen=True)
class KernelSpec:
    symbol: str
    source: Path
    abi: Path
    build_metadata: Path
    prebuilt: Dict[str, Path]
    undefined_symbols: Tuple[str, ...]


@dataclass(frozen=True)
class PluginManifest:
    root: Path
    manifest_path: Path
    name: str
    kind: str
    package_name: str
    interface_provider: str
    op_type: str
    targets: Tuple[str, ...]
    graph_generator: Path
    custom_domain: str
    xml: Path
    interface_source: Path
    implementation_source: Path
    reference: Path
    atol: float
    rtol: float
    kernel: Optional[KernelSpec]

    def supports(self, target: str) -> bool:
        return target in self.targets


def load_manifest(path: Path) -> PluginManifest:
    path = path.resolve()
    root = path.parent
    try:
        doc = load_toml(path)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot parse {path}: {exc}") from exc
    if doc.get("schema_version") != 1:
        raise ManifestError(f"unsupported schema_version in {path}")

    plugin = dict(_required(doc, "plugin", "document"))
    target = dict(_required(doc, "target", "document"))
    graph = dict(_required(doc, "graph", "document"))
    package = dict(_required(doc, "op_package", "document"))
    validation = dict(_required(doc, "validation", "document"))

    name = _identifier(_required(plugin, "name", "plugin"), "plugin.name")
    kind = str(_required(plugin, "kind", "plugin"))
    if kind not in {"cpp", "mlir"}:
        raise ManifestError(f"plugin.kind must be cpp or mlir, got {kind!r}")
    targets_raw = target.get("supported", [target.get("default_arch")])
    if not isinstance(targets_raw, list) or not targets_raw:
        raise ManifestError("target.supported must be a non-empty array")
    targets = tuple(str(item) for item in targets_raw)
    if any(not _TARGET.fullmatch(item) for item in targets):
        raise ManifestError(f"invalid targets: {targets}")
    atol = float(_required(validation, "atol", "validation"))
    rtol = float(_required(validation, "rtol", "validation"))
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ManifestError("validation tolerances must be finite and non-negative")

    kernel_spec: Optional[KernelSpec] = None
    if kind == "mlir":
        kernel = dict(_required(doc, "kernel", "document"))
        prebuilt: Dict[str, Path] = {}
        for arch in targets:
            key = f"prebuilt_{arch}"
            prebuilt[arch] = _relative(root, _required(kernel, key, "kernel"), f"kernel.{key}")
        undefined = kernel.get("undefined_symbols", [])
        if not isinstance(undefined, list):
            raise ManifestError("kernel.undefined_symbols must be an array")
        kernel_spec = KernelSpec(
            symbol=_identifier(_required(kernel, "symbol", "kernel"), "kernel.symbol"),
            source=_relative(root, _required(kernel, "source", "kernel"), "kernel.source"),
            abi=_relative(root, _required(kernel, "abi", "kernel"), "kernel.abi"),
            build_metadata=_relative(root, _required(kernel, "build_metadata", "kernel"), "kernel.build_metadata"),
            prebuilt=prebuilt,
            undefined_symbols=tuple(str(item) for item in undefined),
        )

    return PluginManifest(
        root=root,
        manifest_path=path,
        name=name,
        kind=kind,
        package_name=_identifier(_required(plugin, "package_name", "plugin"), "plugin.package_name"),
        interface_provider=_identifier(_required(plugin, "interface_provider", "plugin"), "plugin.interface_provider"),
        op_type=_identifier(_required(plugin, "op_type", "plugin"), "plugin.op_type"),
        targets=targets,
        graph_generator=_relative(root, _required(graph, "generator", "graph"), "graph.generator"),
        custom_domain=str(_required(graph, "custom_domain", "graph")),
        xml=_relative(root, _required(package, "xml", "op_package"), "op_package.xml"),
        interface_source=_relative(root, _required(package, "interface_source", "op_package"), "op_package.interface_source"),
        implementation_source=_relative(root, _required(package, "implementation_source", "op_package"), "op_package.implementation_source"),
        reference=_relative(root, _required(validation, "reference", "validation"), "validation.reference"),
        atol=atol,
        rtol=rtol,
        kernel=kernel_spec,
    )


def discover_plugins(search_roots: Optional[Iterable[Path]] = None) -> Dict[str, PluginManifest]:
    roots = tuple(search_roots or (ROOT / "examples", ROOT / "templates"))
    plugins: Dict[str, PluginManifest] = {}
    for base in roots:
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*/plugin.toml")):
            manifest = load_manifest(path)
            if manifest.name in plugins:
                raise ManifestError(f"duplicate plugin name: {manifest.name}")
            plugins[manifest.name] = manifest
    return plugins


def get_plugin(name: str) -> PluginManifest:
    plugins = discover_plugins((ROOT / "examples",))
    try:
        return plugins[name]
    except KeyError as exc:
        choices = ", ".join(sorted(plugins)) or "none"
        raise ManifestError(f"unknown plugin {name!r}; available: {choices}") from exc
