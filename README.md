# QNN Hexagon Hybrid Executor

A lightweight, manifest-driven reference for compiling QNN HTP graphs with either conventional C++/HVX Custom Ops or Hexagon-MLIR-backed Custom Ops.

The repository does not require an mllm checkout. In the default MLIR mode it also does not require a Hexagon-MLIR checkout: verified V69 relocatable kernel objects are checked in with ABI and hash metadata. Developers can opt into source mode through an external `HEXAGON_MLIR_ROOT` and `HEXAGON_MLIR_PYTHON`.

## Included examples

- `custom_add`: independent C++/HVX Custom Op template.
- `vector_add_mlir`: minimal QNN-to-MLIR memref bridge.
- `fused_qknorm_rope`: fused Q/K RMSNorm and RoPE region followed by a QNN-native Split.

## Quick start

```bash
python3 -m unittest discover -s tests -v
./hyexec list
./hyexec doctor --kernel-mode prebuilt
./hyexec kernel inspect vector_add_mlir --target v69
./hyexec build vector_add_mlir --target v69 --kernel-mode prebuilt --dry-run
```

Source mode additionally requires Hexagon-MLIR:

```bash
export HEXAGON_MLIR_ROOT=/path/to/hexagon-mlir
export HEXAGON_MLIR_PYTHON=/path/to/hexagon-mlir-env/bin/python
./hyexec build vector_add_mlir --target v69 --kernel-mode source
```

After a successful build, the same entry point creates an ignored deployment
bundle and drives an attached Android device:

```bash
./hyexec bundle vector_add_mlir --target v69
./hyexec deploy vector_add_mlir --target v69
./hyexec run vector_add_mlir --target v69
./hyexec compare vector_add_mlir --target v69
./hyexec profile vector_add_mlir --target v69
```

`bundle` copies external QAIRT/NDK runtime files only into
`deploy/generated/`; those binaries remain ignored and are never part of the
public source tree.

The actual QNN/HTP build requires user-installed QAIRT/QNN, Hexagon SDK/Tools, and Android NDK. See `configs/toolchains.example.toml` and `docs/environment.md`.

## Current scope

The verified target is Snapdragon SM8475 HTP V69. The fused example is a correctness and integration baseline. Its current adapter invokes the MLIR program once per token/head and is not presented as a performance win.

See `PUBLICATION.md` before publishing or redistributing the repository.
