# C++/HVX Custom Op template

`templates/custom_op` contains an independent FP32 `CustomAdd` package. Keep package registration and tensor contracts stable while replacing the correctness-first scalar loop with an HVX routine. The concrete `examples/custom_add` plugin demonstrates graph generation and manifests.
