#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=1, choices=(1, 16, 64, 128))
    args = parser.parse_args()
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise SystemExit("graph generation requires: pip install -e '.[graph]'") from exc

    sequence = args.sequence
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = [
        helper.make_tensor_value_info("q", TensorProto.FLOAT16, [1, sequence, 32, 128]),
        helper.make_tensor_value_info("k", TensorProto.FLOAT16, [1, sequence, 8, 128]),
        helper.make_tensor_value_info("gamma_q", TensorProto.FLOAT16, [128]),
        helper.make_tensor_value_info("gamma_k", TensorProto.FLOAT16, [128]),
        helper.make_tensor_value_info("sin", TensorProto.FLOAT, [sequence, 64]),
        helper.make_tensor_value_info("cos", TensorProto.FLOAT, [sequence, 64]),
    ]
    outputs = [
        helper.make_tensor_value_info("q_out", TensorProto.FLOAT16, [1, sequence, 32, 128]),
        helper.make_tensor_value_info("k_out", TensorProto.FLOAT16, [1, sequence, 8, 128]),
    ]
    split_sizes = numpy_helper.from_array(np.array([32, 8], dtype=np.int64), "split_sizes")
    nodes = [
        helper.make_node("FusedQKNormRoPE", ["q", "k", "gamma_q", "gamma_k", "sin", "cos"], ["combined"], domain="GraphAgent"),
        helper.make_node("Split", ["combined", "split_sizes"], ["q_out", "k_out"], axis=2),
    ]
    combined = helper.make_tensor_value_info("combined", TensorProto.FLOAT16, [1, sequence, 40, 128])
    graph = helper.make_graph(nodes, "fused_qknorm_rope", inputs, outputs, [split_sizes], value_info=[combined])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("GraphAgent", 1)])
    onnx.save(model, args.output / "fused_qknorm_rope.onnx")

    rng = np.random.default_rng(20261027 + sequence)
    q = rng.normal(size=(1, sequence, 32, 128)).astype("<f2")
    k = rng.normal(size=(1, sequence, 8, 128)).astype("<f2")
    gamma_q = rng.normal(1.0, 0.1, size=(128,)).astype("<f2")
    gamma_k = rng.normal(1.0, 0.1, size=(128,)).astype("<f2")
    positions = np.arange(sequence, dtype=np.float32)[:, None]
    frequencies = 1.0 / (1_000_000.0 ** (np.arange(64, dtype=np.float32) / 64.0))
    angles = positions * frequencies[None, :]
    sin, cos = np.sin(angles).astype("<f4"), np.cos(angles).astype("<f4")

    ref_path = Path(__file__).resolve().parents[1] / "reference" / "reference.py"
    spec = importlib.util.spec_from_file_location("fused_reference", ref_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    q_out, k_out = module.fused_qknorm_rope(q, k, gamma_q, gamma_k, sin, cos)
    tensors = {"q": q, "k": k, "gamma_q": gamma_q, "gamma_k": gamma_k, "sin": sin, "cos": cos,
               "q_expected": q_out, "k_expected": k_out}
    for name, value in tensors.items():
        value.tofile(args.output / f"{name}.raw")
    (args.output / "input_list.txt").write_text(
        "q:=q.raw k:=k.raw gamma_q:=gamma_q.raw gamma_k:=gamma_k.raw sin:=sin.raw cos:=cos.raw\n",
        encoding="utf-8",
    )
    (args.output / "graph_manifest.json").write_text(json.dumps({
        "contract": "fused_qk_norm_rope_v1", "sequence": sequence,
        "seed": 20261027 + sequence, "physical_output": [1, sequence, 40, 128],
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
