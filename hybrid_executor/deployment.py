from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

from .build import build_dir
from .config import ROOT, ToolchainConfig
from .errors import BuildError, ConfigurationError
from .manifests import PluginManifest
from .process import render, run


GENERATED_DEPLOY_ROOT = ROOT / "deploy" / "generated"
DEVICE_ROOT = "/data/local/tmp/hyexec"


@dataclass(frozen=True)
class BundleFile:
    source: Path
    relative: Path
    executable: bool = False


def bundle_dir(plugin: PluginManifest, target: str) -> Path:
    return GENERATED_DEPLOY_ROOT / plugin.name / target / "bundle"


def _require_file(path: Path, purpose: str) -> Path:
    if not path.is_file():
        raise BuildError(f"missing {purpose}: {path}; run `hyexec build` first")
    return path


def _runtime_files(config: ToolchainConfig, target: str) -> Sequence[BundleFile]:
    qairt = config.require("qairt_sdk_root")
    ndk = config.require("android_ndk_root")
    arch = target.removeprefix("v")
    android_lib = qairt / "lib" / "aarch64-android"
    return (
        BundleFile(qairt / "bin/aarch64-android/qnn-net-run", Path("runtime/qnn-net-run"), True),
        BundleFile(android_lib / "libQnnHtp.so", Path("runtime/libQnnHtp.so")),
        BundleFile(android_lib / "libQnnHtpPrepare.so", Path("runtime/libQnnHtpPrepare.so")),
        BundleFile(android_lib / f"libQnnHtpV{arch}Stub.so", Path(f"runtime/libQnnHtpV{arch}Stub.so")),
        BundleFile(
            qairt / f"lib/hexagon-v{arch}/unsigned/libQnnHtpV{arch}Skel.so",
            Path(f"runtime/libQnnHtpV{arch}Skel.so"),
        ),
        BundleFile(
            ndk / "toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so",
            Path("runtime/libc++_shared.so"),
        ),
    )


def plan_bundle(plugin: PluginManifest, target: str, config: ToolchainConfig) -> Sequence[BundleFile]:
    if not plugin.supports(target):
        raise BuildError(f"plugin {plugin.name} does not support {target}")
    work = build_dir(plugin, target)
    package_library = f"libQnn{plugin.package_name}.so"
    files: List[BundleFile] = list(_runtime_files(config, target))
    files.extend((
        BundleFile(
            work / "model_lib/aarch64-android" / f"lib{plugin.name}.so",
            Path("model") / f"lib{plugin.name}.so",
        ),
        BundleFile(
            work / "op_package/aarch64-android" / package_library,
            Path("op_packages/arm") / package_library,
        ),
        BundleFile(
            work / "op_package/hexagon-v69" / package_library,
            Path("op_packages/dsp") / package_library,
        ),
    ))
    graph = work / "graph"
    for path in sorted(graph.iterdir()) if graph.is_dir() else ():
        if path.is_file() and path.suffix in {".raw", ".txt", ".json"}:
            files.append(BundleFile(path, Path("inputs") / path.name))
    return tuple(files)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_bundle(
    plugin: PluginManifest, target: str, config: ToolchainConfig, dry_run: bool = False
) -> str:
    destination = bundle_dir(plugin, target)
    files = plan_bundle(plugin, target, config)
    lines = [f"bundle: {destination}"]
    if dry_run:
        lines.extend(f"copy {item.source} -> {item.relative}" for item in files)
        return "\n".join(lines)

    manifest_files = []
    for item in files:
        source = _require_file(item.source, str(item.relative))
        output = destination / item.relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        if item.executable:
            output.chmod(output.stat().st_mode | 0o111)
        manifest_files.append({
            "path": item.relative.as_posix(),
            "sha256": _sha256(output),
            "bytes": output.stat().st_size,
        })
    manifest = {
        "schema_version": 1,
        "plugin": plugin.name,
        "target": target,
        "package_name": plugin.package_name,
        "interface_provider": plugin.interface_provider,
        "files": manifest_files,
    }
    manifest_path = destination / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines.append(f"manifest: {manifest_path}")
    lines.append(f"files: {len(manifest_files)}")
    return "\n".join(lines)


def _adb(config: ToolchainConfig) -> List[str]:
    executable = shutil.which("adb")
    if executable is None:
        raise ConfigurationError("adb is not available on PATH")
    argv = [executable]
    if config.adb_serial:
        argv += ["-s", config.adb_serial]
    return argv


def _device_dir(plugin: PluginManifest, target: str) -> str:
    return f"{DEVICE_ROOT}/{plugin.name}/{target}"


def deploy_bundle(
    plugin: PluginManifest, target: str, config: ToolchainConfig, dry_run: bool = False
) -> str:
    source = bundle_dir(plugin, target)
    manifest = source / "bundle_manifest.json"
    if not dry_run:
        _require_file(manifest, "bundle manifest")
    remote = _device_dir(plugin, target)
    adb = _adb(config)
    commands = (
        adb + ["shell", "mkdir", "-p", remote],
        adb + ["push", str(source) + os.sep + ".", remote + "/"],
        adb + ["shell", "chmod", "0755", remote + "/runtime/qnn-net-run"],
    )
    lines = []
    for command in commands:
        result = run(command, dry_run=dry_run, timeout=300)
        lines.append(render(command))
        if result.stdout and not dry_run:
            lines.append(result.stdout.rstrip())
    state = GENERATED_DEPLOY_ROOT / plugin.name / target / "deployment.json"
    if not dry_run:
        state.write_text(json.dumps({
            "schema_version": 1,
            "plugin": plugin.name,
            "target": target,
            "device_dir": remote,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines.append(f"device_dir: {remote}")
    return "\n".join(lines)


def run_on_device(
    plugin: PluginManifest,
    target: str,
    config: ToolchainConfig,
    dry_run: bool = False,
    profiling: bool = False,
) -> str:
    adb = _adb(config)
    remote = _device_dir(plugin, target)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = f"{remote}/outputs/{stamp}"
    package_library = f"libQnn{plugin.package_name}.so"
    command = [
        f"cd {shlex.quote(remote + '/inputs')}",
        f"export LD_LIBRARY_PATH={shlex.quote(remote + '/runtime')}",
        "export ADSP_LIBRARY_PATH=" + shlex.quote(
            remote + "/op_packages/dsp;" + remote +
            "/runtime;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
        ),
        f"mkdir -p {shlex.quote(output)}",
    ]
    qnn_argv = [
        remote + "/runtime/qnn-net-run",
        "--model", remote + f"/model/lib{plugin.name}.so",
        "--input_list", "input_list.txt",
        "--backend", remote + "/runtime/libQnnHtp.so",
        "--op_packages",
        (remote + f"/op_packages/arm/{package_library}:{plugin.interface_provider}:CPU," +
         f"{package_library}:{plugin.interface_provider}:HTP"),
        "--output_dir", output,
        "--use_native_input_files",
        "--use_native_output_files",
        "--perf_profile", "burst",
        "--log_level", "info",
    ]
    if profiling:
        qnn_argv += ["--profiling_level", "detailed", "--profiling_option", "optrace"]
    command.append(render(qnn_argv))
    shell_command = " && ".join(command)
    result = run(adb + ["shell", shell_command], dry_run=dry_run, timeout=600)
    state = GENERATED_DEPLOY_ROOT / plugin.name / target / "last_run.json"
    if not dry_run:
        state.write_text(json.dumps({
            "schema_version": 1,
            "plugin": plugin.name,
            "target": target,
            "device_dir": remote,
            "output_dir": output,
            "profiling": profiling,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [render(adb + ["shell", shell_command]), f"output_dir: {output}"]
    if result.stdout and not dry_run:
        lines.append(result.stdout.rstrip())
    if result.stderr and not dry_run:
        lines.append(result.stderr.rstrip())
    return "\n".join(lines)


def _load_last_run(plugin: PluginManifest, target: str) -> dict:
    state = GENERATED_DEPLOY_ROOT / plugin.name / target / "last_run.json"
    if not state.is_file():
        raise BuildError(f"missing run state: {state}; run `hyexec run` first")
    document = json.loads(state.read_text(encoding="utf-8"))
    if document.get("plugin") != plugin.name or document.get("target") != target:
        raise BuildError(f"run state does not match {plugin.name}/{target}")
    return document


def _unique_output(root: Path, filename: str) -> Path:
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if len(matches) != 1:
        raise BuildError(
            f"expected exactly one device output named {filename}, found {len(matches)} under {root}"
        )
    return matches[0]


def compare_last_run(
    plugin: PluginManifest, target: str, config: ToolchainConfig, dry_run: bool = False
) -> str:
    document = _load_last_run(plugin, target)
    remote_output = str(document["output_dir"])
    stamp = Path(remote_output).name
    local_output = build_dir(plugin, target) / "device_output" / stamp
    command = _adb(config) + ["pull", remote_output, str(local_output)]
    if dry_run:
        return render(command)
    run(command, timeout=300)

    try:
        import numpy as np
    except ImportError as exc:
        raise BuildError("comparison requires NumPy") from exc

    inputs = bundle_dir(plugin, target) / "inputs"
    if (inputs / "reference.raw").is_file():
        comparisons = (("out", inputs / "reference.raw", "out.raw", "<f4"),)
    elif (inputs / "q_expected.raw").is_file() and (inputs / "k_expected.raw").is_file():
        comparisons = (
            ("q_out", inputs / "q_expected.raw", "q_out.raw", "<f2"),
            ("k_out", inputs / "k_expected.raw", "k_out.raw", "<f2"),
        )
    else:
        raise BuildError(f"bundle has no recognized reference outputs: {inputs}")

    records = []
    overall = True
    for name, expected_path, actual_name, dtype in comparisons:
        actual_path = _unique_output(local_output, actual_name)
        expected = np.fromfile(expected_path, dtype=dtype).astype(np.float64)
        actual = np.fromfile(actual_path, dtype=dtype).astype(np.float64)
        if expected.shape != actual.shape:
            raise BuildError(
                f"{name} element count mismatch: expected {expected.size}, got {actual.size}"
            )
        absolute = np.abs(actual - expected)
        denominator = np.maximum(np.abs(expected), 1.0e-12)
        passed = bool(np.allclose(actual, expected, atol=plugin.atol, rtol=plugin.rtol))
        overall = overall and passed
        records.append({
            "name": name,
            "passed": passed,
            "elements": int(expected.size),
            "max_abs": float(absolute.max(initial=0.0)),
            "max_rel": float((absolute / denominator).max(initial=0.0)),
            "atol": plugin.atol,
            "rtol": plugin.rtol,
        })
    report = {
        "schema_version": 1,
        "plugin": plugin.name,
        "target": target,
        "passed": overall,
        "outputs": records,
    }
    report_path = local_output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not overall:
        raise BuildError(f"device output comparison failed; see {report_path}")
    return json.dumps(report, indent=2, sort_keys=True)


def clean_build(plugin: PluginManifest, target: str, dry_run: bool = False) -> str:
    path = build_dir(plugin, target)
    try:
        path.resolve().relative_to((ROOT / "build").resolve())
    except ValueError as exc:
        raise BuildError(f"refusing to clean path outside build root: {path}") from exc
    if dry_run:
        return f"would remove generated build directory: {path}"
    existed = path.exists()
    if existed:
        shutil.rmtree(path)
    return f"removed generated build directory: {path}" if existed else f"already clean: {path}"
