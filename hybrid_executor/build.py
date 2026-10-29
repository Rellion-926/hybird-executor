from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from .config import ROOT, ToolchainConfig
from .errors import BuildError
from .kernel import inspect_plugin_kernel
from .manifests import PluginManifest
from .process import render, run


@dataclass(frozen=True)
class BuildStep:
    name: str
    argv: Sequence[str]
    cwd: Path
    env: Optional[Mapping[str, str]] = None


def build_dir(plugin: PluginManifest, target: str) -> Path:
    return ROOT / "build" / plugin.name / target


def hexagon_mlir_environment(
    config: ToolchainConfig, target: str
) -> tuple[Path, Mapping[str, str]]:
    root = config.require("hexagon_mlir_root")
    python = config.require("hexagon_mlir_python")
    if not python.is_file():
        raise BuildError(f"Hexagon-MLIR Python does not exist: {python}")
    version_result = run(
        [python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        cwd=root,
    )
    python_version = version_result.stdout.strip()
    if not python_version:
        raise BuildError(f"cannot determine Python version from {python}")

    triton_root = root / "triton"
    triton_build = triton_root / f"build/cmake.linux-x86_64-cpython-{python_version}"
    shared_dir = triton_build / "third_party/triton_shared/tools/triton-shared-opt"
    shared_opt = shared_dir / "triton-shared-opt"
    backend_bin = triton_build / "third_party/qcom_hexagon_backend/bin"
    if not shared_opt.is_file():
        raise BuildError(f"missing Hexagon-MLIR triton-shared-opt: {shared_opt}")
    if not backend_bin.is_dir():
        raise BuildError(f"missing Hexagon-MLIR backend tools: {backend_bin}")

    arch = target.removeprefix("v")
    env = {
        "HEXAGON_MLIR_ROOT": str(root),
        "TRITON_ROOT": str(triton_root),
        "TRITON_SHARED_OPT_PATH": str(shared_opt),
        "TRITON_HOME": str(root),
        "TRITON_PLUGIN_DIRS": f"{root / 'triton_shared'};{root / 'qcom_hexagon_backend'}",
        "HEXAGON_ARCH_VERSION": arch,
        "PATH": os.pathsep.join((str(backend_bin), str(shared_dir), os.environ.get("PATH", ""))),
        "PYTHONPATH": os.pathsep.join((str(triton_root / "python"), os.environ.get("PYTHONPATH", ""))),
    }
    if config.hexagon_tools_root is not None:
        env["HEXAGON_TOOLS"] = str(config.hexagon_tools_root)
        env["HEXAGON_TOOLS_ROOT"] = str(config.hexagon_tools_root)
    if config.hexagon_sdk_root is not None:
        env["HEXAGON_SDK_ROOT"] = str(config.hexagon_sdk_root)
    return python, env


def plan_build(plugin: PluginManifest, target: str, kernel_mode: str, config: ToolchainConfig) -> List[BuildStep]:
    if not plugin.supports(target):
        raise BuildError(f"plugin {plugin.name} does not support {target}")
    work = build_dir(plugin, target)
    graph_dir = work / "graph"
    steps = [BuildStep(
        "generate graph",
        [sys.executable, str(plugin.graph_generator), "--output", str(graph_dir)],
        plugin.root,
    )]
    kernel_object: Optional[Path] = None
    if plugin.kind == "mlir":
        if kernel_mode == "prebuilt":
            inspect_plugin_kernel(plugin, target)
            assert plugin.kernel is not None
            kernel_object = plugin.kernel.prebuilt[target]
        elif kernel_mode == "source":
            if config.hexagon_mlir_root is None:
                raise BuildError("HEXAGON_MLIR_ROOT is required by --kernel-mode source")
            if config.hexagon_mlir_python is None:
                raise BuildError("HEXAGON_MLIR_PYTHON is required by --kernel-mode source")
            python, mlir_env = hexagon_mlir_environment(config, target)
            assert plugin.kernel is not None
            kernel_object = work / "kernel" / "kernel.o"
            steps.append(BuildStep(
                "compile MLIR kernel",
                [str(python), str(plugin.kernel.source), "--output", str(kernel_object)],
                plugin.kernel.source.parent,
                mlir_env,
            ))
        else:
            raise BuildError(f"unknown kernel mode: {kernel_mode}")
    qairt = config.qairt_sdk_root
    if qairt is not None:
        converter = qairt / "bin" / "x86_64-linux-clang" / "qnn-onnx-converter"
        model_cpp = work / "model" / f"{plugin.name}.cpp"
        converter_argv = [str(converter), "--input_network", str(graph_dir / f"{plugin.name}.onnx"),
                          "--output_path", str(model_cpp), "--op_package_config", str(plugin.xml)]
        if plugin.name == "fused_qknorm_rope":
            converter_argv += ["--preserve_io", "datatype", "q", "k", "gamma_q", "gamma_k", "sin", "cos", "q_out", "k_out"]
        steps.append(BuildStep(
            "convert ONNX graph",
            converter_argv,
            work,
        ))
        if plugin.name == "fused_qknorm_rope":
            steps.append(BuildStep(
                "freeze FP16 Custom Op boundary",
                [sys.executable, str(plugin.root / "tools" / "freeze_fp16_custom_boundary.py"), str(model_cpp)],
                work,
            ))
        model_generator = qairt / "bin" / "x86_64-linux-clang" / "qnn-model-lib-generator"
        model_env = None
        if config.android_ndk_root is not None:
            # qnn-model-lib-generator shells out to `ndk-build` by name.  Merely
            # setting ANDROID_NDK_ROOT is not sufficient for QAIRT 2.41, so make
            # the dependency explicit in the deterministic build step.
            model_env = {
                "ANDROID_NDK_ROOT": str(config.android_ndk_root),
                "PATH": str(config.android_ndk_root) + os.pathsep + os.environ.get("PATH", ""),
            }
        steps.append(BuildStep(
            "build Android model library",
            [str(model_generator), "-c", str(model_cpp), "-t", "aarch64-android",
             "-l", plugin.name, "-o", str(work / "model_lib")],
            work,
            model_env,
        ))

        if config.hexagon_sdk_root and config.hexagon_tools_root:
            qnn_include = qairt / "include" / "QNN"
            make = "make"
            htp_argv = [
                make, "-f", str(ROOT / "build_rules" / "qnn_htp_package.mk"),
                f"QNN_INCLUDE={qnn_include}", f"HEXAGON_SDK_ROOT={config.hexagon_sdk_root}",
                f"HEXAGON_TOOLS_ROOT={config.hexagon_tools_root}", f"PACKAGE_NAME={plugin.package_name}",
                f"INTERFACE_SOURCE={plugin.interface_source}", f"OP_SOURCE={plugin.implementation_source}",
                f"PROJECT_INCLUDE={ROOT / 'include'}", f"BUILD_DIR={work / 'op_package' / 'hexagon-v69'}",
                f"OUTPUT={work / 'op_package' / 'hexagon-v69' / ('libQnn' + plugin.package_name + '.so')}",
            ]
            if kernel_object:
                htp_argv.append(f"KERNEL_OBJECT={kernel_object}")
            steps.append(BuildStep("build V69 DSP OpPackage", htp_argv, ROOT))
        if config.android_ndk_root and config.hexagon_sdk_root and config.hexagon_tools_root:
            arm_argv = [
                "make", "-f", str(ROOT / "build_rules" / "qnn_arm_package.mk"),
                f"QNN_INCLUDE={qairt / 'include' / 'QNN'}", f"QNN_TARGET_LIB={qairt / 'lib' / 'aarch64-android'}",
                f"HEXAGON_SDK_ROOT={config.hexagon_sdk_root}", f"HEXAGON_TOOLS_ROOT={config.hexagon_tools_root}",
                f"ANDROID_NDK_ROOT={config.android_ndk_root}", f"PACKAGE_NAME={plugin.package_name}",
                f"INTERFACE_SOURCE={plugin.interface_source}", f"OP_SOURCE={plugin.implementation_source}",
                f"PROJECT_INCLUDE={ROOT / 'include'}", f"BUILD_DIR={work / 'op_package' / 'aarch64-android'}",
                f"OUTPUT={work / 'op_package' / 'aarch64-android' / ('libQnn' + plugin.package_name + '.so')}",
            ]
            steps.append(BuildStep("build Android ARM OpPackage", arm_argv, ROOT))
    return steps


def execute_plan(steps: Sequence[BuildStep], dry_run: bool) -> str:
    lines = []
    for step in steps:
        result = run(step.argv, cwd=step.cwd, env=step.env, dry_run=dry_run)
        lines.append(f"[{step.name}] {render(step.argv)}")
        if result.stdout and not dry_run:
            lines.append(result.stdout.rstrip())
    return "\n".join(lines)
