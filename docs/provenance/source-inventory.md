# Source inventory

| Public component | Classification | Internal provenance | Publication action |
|---|---|---|---|
| Python CLI/config/manifest/object inspection | Newly implemented | None | Retain after review |
| `CustomAdd` template | Newly implemented against QNN OpPackage interfaces | QNN interface behavior validated by internal experiments | Retain after QNN SDK license/interface review |
| VectorAdd package and bridge | Refactored internal implementation | Step 7 verified bridge | Retain after interface review |
| VectorAdd V69 object | Generated binary artifact | Step 7, SHA-256 recorded | Retain only if Hexagon-MLIR source license and binary redistribution permit |
| Fused QKNormRoPE kernel source/bridge | Refactored internal implementation | Steps 8/9 | Retain after source provenance review |
| Fused QKNormRoPE V69 object | Generated binary artifact | Steps 8/9, SHA-256 recorded | Retain only if source license and binary redistribution permit |
| FP16 generated-source post-pass | Internal implementation | Step 9 | Retain with QAIRT-version limitation documented |
| V69 thread-atexit shim | Internal compatibility implementation | Steps 6/8 | Retain after standard ABI review |
| mllm RMSNorm/RoPE package | External-derived experiment code | Step 5 | Excluded from public source |
| Qualcomm runtime libraries and tools | Proprietary external dependency | Internal deployment bundles | Excluded |
| Raw vectors, outputs, logs and profiles | Generated experiment artifacts | Internal workspace | Excluded from Git; compact results only |
