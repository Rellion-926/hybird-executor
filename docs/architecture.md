# Architecture

```text
plugin.toml
   ├── graph generator → ONNX custom-domain graph
   ├── OpPackage XML → QAIRT converter
   ├── ARM package provider
   └── DSP implementation
          ├── C++/HVX compute, or
          └── QNN tensor → memref → MLIR kernel object
```

Both plugin kinds use one QNN graph/model/OpPackage pipeline. MLIR plugins select either a hash-checked prebuilt object or an object regenerated from an external Hexagon-MLIR environment.
