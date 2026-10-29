# Known limitations

- Only V69 is verified.
- QAIRT 2.41 promotes the fused Custom Op boundary to FP32 without the example-specific post-pass.
- The fused adapter currently invokes `S*40` MLIR programs serially.
- The fused reduction object imports `malloc` and `free`.
- The checked-in objects require publication provenance review.
- Actual QNN builds require a licensed Linux toolchain environment; macOS supports control-plane tests and ADB deployment.
