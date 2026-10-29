#!/usr/bin/env python3
"""Make the fused Custom Op boundary FP16 in QAIRT-generated C++.

QAIRT 2.41 applies its global 32-bit float policy to custom-op internals even
when the converter XML declares FP16.  Public I/O is already preserved as
FP16.  This deterministic post-pass removes the six redundant conversions,
feeds the public FP16 tensors directly to the custom op, and makes the native
Split publish the two FP16 outputs.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path


def transform(text: str) -> str:
    custom_begin = text.index("static ModelError_t addNode_FusedQKNormRoPE_0")
    split_begin = text.index("static ModelError_t addNode_Split_0", custom_begin)
    custom = text[custom_begin:split_begin]
    for name in ("q", "k", "gamma_q", "gamma_k"):
        old = f'"{name}_converted"'
        if old not in custom:
            raise RuntimeError(f"expected converted custom input is missing: {old}")
        custom = custom.replace(old, f'"{name}"')
    if "QNN_DATATYPE_FLOAT_32" not in custom:
        raise RuntimeError("expected FP32 custom output is missing")
    custom = custom.replace("QNN_DATATYPE_FLOAT_32", "QNN_DATATYPE_FLOAT_16")
    text = text[:custom_begin] + custom + text[split_begin:]

    split_begin = text.index("static ModelError_t addNode_Split_0")
    next_function = text.index(
        "static ModelError_t addNode_QNN_DATATYPE_FLOAT_32_converted_q_out",
        split_begin,
    )
    split = text[split_begin:next_function]
    replacements = {
        "dimensions_q_out_converted": "dimensions_q_out",
        "dimensions_k_out_converted": "dimensions_k_out",
        '"q_out_converted"': '"q_out"',
        '"k_out_converted"': '"k_out"',
    }
    for old, new in replacements.items():
        if old not in split:
            raise RuntimeError(f"expected converted Split output is missing: {old}")
        split = split.replace(old, new)
    if "QNN_TENSOR_TYPE_NATIVE" not in split or "QNN_DATATYPE_FLOAT_32" not in split:
        raise RuntimeError("expected native FP32 Split outputs are missing")
    split = split.replace("QNN_TENSOR_TYPE_NATIVE", "QNN_TENSOR_TYPE_APP_READ")
    split = split.replace("QNN_DATATYPE_FLOAT_32", "QNN_DATATYPE_FLOAT_16")
    text = text[:split_begin] + split + text[next_function:]

    remove_patterns = (
        r"^\s*VALIDATE\(addNode_QNN_DATATYPE_FLOAT_16_converted_(?:q|k|gamma_q|gamma_k)_QNN_DATATYPE_FLOAT_32\([^\n]+\n",
        r"^\s*VALIDATE\(addNode_QNN_DATATYPE_FLOAT_32_converted_(?:q_out|k_out)_QNN_DATATYPE_FLOAT_16\([^\n]+\n",
    )
    expected_counts = (4, 2)
    for pattern, expected_count in zip(remove_patterns, expected_counts):
        text, count = re.subn(pattern, "", text, flags=re.MULTILINE)
        if count != expected_count:
            raise RuntimeError(
                f"expected to remove {expected_count} Convert calls, removed {count}"
            )

    compose = text[text.index("ModelError_t QnnModel_composeGraphs"):]
    if "addNode_QNN_DATATYPE_FLOAT_16_converted_" in compose:
        raise RuntimeError("input Convert call remains in graph composition")
    if "addNode_QNN_DATATYPE_FLOAT_32_converted_" in compose:
        raise RuntimeError("output Convert call remains in graph composition")

    custom = text[text.index("static ModelError_t addNode_FusedQKNormRoPE_0"):]
    custom = custom[:custom.index("static ModelError_t addNode_Split_0")]
    if custom.count("QNN_DATATYPE_FLOAT_16") != 1:
        raise RuntimeError("custom output is not exactly one FP16 tensor")
    if custom.count("_converted\"") != 0:
        raise RuntimeError("custom input still references a converted tensor")

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    text = transform(args.path.read_text(encoding="utf-8"))

    # Keep the converter output unchanged if validation or writing fails.
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.path.parent,
            prefix=args.path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, args.path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
