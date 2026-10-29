# Hexagon-MLIR plugin template

An MLIR plugin adds `kernel/abi.json`, `kernel/build.json`, `kernel/SHA256SUMS`,
an optional checked-in V69 object, and a QNN tensor-to-ranked-memref adapter.
Run `hyexec kernel inspect` before linking a prebuilt object.

Source mode uses `HEXAGON_MLIR_PYTHON` only for the kernel step while the QNN
graph conversion continues in the QAIRT Python environment. The executor
derives the matching `cmake.linux-x86_64-cpython-X.Y` paths, configures the
Triton plugins and `HEXAGON_ARCH_VERSION`, and writes the result to
`build/<plugin>/<target>/kernel/kernel.o`. It never overwrites the checked-in
prebuilt object.
