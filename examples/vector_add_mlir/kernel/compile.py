#!/usr/bin/env python3
"""Compile the vector-add kernel using an external Hexagon-MLIR environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.htp import triton_htp_compile


@triton.jit
def vector_add_mlir(x_ptr, y_ptr, output_ptr, block_size: tl.constexpr):
    offsets = tl.arange(0, block_size)
    tl.store(output_ptr + offsets, tl.load(x_ptr + offsets) + tl.load(y_ptr + offsets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    obj = triton_htp_compile(
        vector_add_mlir,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32"},
        constants={"block_size": 1024},
        file_ext="o",
        extra_options={
            "enableVectorization": True,
            "enableVTCMTiling": False,
            "enableConvertToHexagonmem": False,
            "enableHexagonmemCopyToDMA": False,
            "enableMultiThreading": False,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(obj)


if __name__ == "__main__":
    main()
