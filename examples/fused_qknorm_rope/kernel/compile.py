#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from triton.backends.qcom_hexagon_backend.htp import triton_htp_compile
from kernel import fused_qknorm_rope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    obj = triton_htp_compile(
        fused_qknorm_rope,
        signature={
            "q_ptr": "*fp16", "k_ptr": "*fp16", "gamma_q_ptr": "*fp16",
            "gamma_k_ptr": "*fp16", "sin_ptr": "*fp32", "cos_ptr": "*fp32",
            "combined_out_ptr": "*fp16",
        },
        constants={"Q_HEADS": 32, "K_HEADS": 8, "HEAD_DIM": 128, "ROTARY_HALF": 64, "EPSILON": 1.0e-6},
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
