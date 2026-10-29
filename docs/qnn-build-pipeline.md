# QNN build pipeline

```text
graph generator
→ ONNX custom-domain graph
→ qnn-onnx-converter + OpPackage XML
→ generated QNN C++
→ optional deterministic post-pass
→ qnn-model-lib-generator
→ Android model library
→ ARM and V69 OpPackage libraries
→ device bundle
```

The organization resembles the practical mllm QNN flow but is implemented independently and does not require an mllm checkout.
