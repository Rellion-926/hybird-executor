from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from . import __version__
from .build import execute_plan, plan_build
from .config import ROOT, load_config
from .deployment import clean_build, compare_last_run, create_bundle, deploy_bundle, run_on_device
from .doctor import format_checks, run_checks
from .errors import HybridExecutorError
from .kernel import format_object, inspect_plugin_kernel
from .manifests import discover_plugins, get_plugin


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)


def _config(args: argparse.Namespace):
    return load_config(getattr(args, "config", None))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyexec", description="QNN Hexagon hybrid executor")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    list_cmd = commands.add_parser("list", help="list manifest-defined plugins")
    list_cmd.add_argument("--json", action="store_true")

    doctor = commands.add_parser("doctor", help="check toolchains and optional device")
    _common_config(doctor)
    doctor.add_argument("--kernel-mode", choices=("prebuilt", "source"), default="prebuilt")
    doctor.add_argument("--device", action="store_true")

    graph = commands.add_parser("graph", help="generate plugin ONNX graph and inputs")
    _common_config(graph)
    graph.add_argument("plugin")
    graph.add_argument("--target", default="v69")
    graph.add_argument("--dry-run", action="store_true")

    kernel = commands.add_parser("kernel", help="inspect/build/freeze an MLIR kernel")
    kernel_commands = kernel.add_subparsers(dest="kernel_command", required=True)
    inspect = kernel_commands.add_parser("inspect")
    inspect.add_argument("plugin")
    inspect.add_argument("--target", default="v69")
    for name in ("build", "freeze"):
        sub = kernel_commands.add_parser(name)
        _common_config(sub)
        sub.add_argument("plugin")
        sub.add_argument("--target", default="v69")
        sub.add_argument("--dry-run", action="store_true")

    build = commands.add_parser("build", help="build graph/model and OpPackage")
    _common_config(build)
    build.add_argument("plugin")
    build.add_argument("--target", default="v69")
    build.add_argument("--kernel-mode", choices=("prebuilt", "source"), default="prebuilt")
    build.add_argument("--dry-run", action="store_true")

    for name in ("bundle", "deploy", "run", "compare", "profile", "clean"):
        sub = commands.add_parser(name)
        _common_config(sub)
        sub.add_argument("plugin")
        sub.add_argument("--target", default="v69")
        sub.add_argument("--dry-run", action="store_true")
    return parser


def _print_plugins(as_json: bool) -> None:
    plugins = discover_plugins((ROOT / "examples",))
    rows = []
    for item in plugins.values():
        rows.append({
            "name": item.name,
            "kind": item.kind,
            "targets": list(item.targets),
            "prebuilt": bool(item.kernel and all(p.is_file() for p in item.kernel.prebuilt.values())),
        })
    if as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print("NAME                 KIND  TARGETS  PREBUILT")
        for row in rows:
            print(f"{row['name']:<20} {row['kind']:<5} {','.join(row['targets']):<8} {'yes' if row['prebuilt'] else 'no'}")


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            _print_plugins(args.json)
        elif args.command == "doctor":
            checks = run_checks(_config(args), args.kernel_mode, args.device)
            print(format_checks(checks))
            return 1 if any(item.status == "FAIL" for item in checks) else 0
        elif args.command == "kernel" and args.kernel_command == "inspect":
            print(format_object(inspect_plugin_kernel(get_plugin(args.plugin), args.target)))
        elif args.command in {"build", "graph"}:
            plugin = get_plugin(args.plugin)
            mode = getattr(args, "kernel_mode", "prebuilt")
            plan = plan_build(plugin, args.target, mode, _config(args))
            if args.command == "graph":
                plan = plan[:1]
            print(execute_plan(plan, args.dry_run))
        elif args.command == "bundle":
            print(create_bundle(get_plugin(args.plugin), args.target, _config(args), args.dry_run))
        elif args.command == "deploy":
            print(deploy_bundle(get_plugin(args.plugin), args.target, _config(args), args.dry_run))
        elif args.command in {"run", "profile"}:
            print(run_on_device(
                get_plugin(args.plugin), args.target, _config(args), args.dry_run,
                profiling=args.command == "profile",
            ))
        elif args.command == "compare":
            print(compare_last_run(get_plugin(args.plugin), args.target, _config(args), args.dry_run))
        elif args.command == "clean":
            print(clean_build(get_plugin(args.plugin), args.target, args.dry_run))
        elif args.command == "kernel" and args.kernel_command == "build":
            plugin = get_plugin(args.plugin)
            plan = plan_build(plugin, args.target, "source", _config(args))
            compile_steps = [step for step in plan if step.name == "compile MLIR kernel"]
            if len(compile_steps) != 1:
                raise HybridExecutorError(f"plugin {plugin.name} does not define one MLIR compile step")
            print(execute_plan(compile_steps, args.dry_run))
        elif args.command == "kernel":
            raise HybridExecutorError(f"kernel {args.kernel_command} will be enabled after source-mode validation")
        else:
            raise HybridExecutorError(f"{args.command} is scaffolded but not implemented yet")
    except HybridExecutorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
