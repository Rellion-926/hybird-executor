# Third-party dependencies

| Component | Role | Distribution policy |
|---|---|---|
| Qualcomm QAIRT/QNN | Graph conversion, model generation, HTP runtime and OpPackage interfaces | User-installed; not redistributed |
| Qualcomm Hexagon SDK/Tools | V69 compilation and linking | User-installed; not redistributed |
| Android NDK | AArch64 model and package compilation | User-installed; not redistributed |
| Hexagon-MLIR and Triton components | Optional MLIR kernel source compilation | External checkout selected by `HEXAGON_MLIR_ROOT` |
| mllm | Architectural inspiration and historical B2 baseline | No source/runtime dependency in the public core |
| NumPy and ONNX | Deterministic graph/input generation | Installed through optional Python dependencies |

Exact versions and licenses must be recorded before public publication.
