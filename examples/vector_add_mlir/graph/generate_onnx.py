#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:
        raise SystemExit("graph generation requires: pip install -e '.[graph]'") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    shape = [1, 1, 1, 1024]
    graph = helper.make_graph(
        [helper.make_node("VectorAddMlir", ["lhs", "rhs"], ["out"], domain="GraphAgent")],
        "vector_add_mlir",
        [helper.make_tensor_value_info("lhs", TensorProto.FLOAT, shape),
         helper.make_tensor_value_info("rhs", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("GraphAgent", 1)])
    onnx.save(model, args.output / "vector_add_mlir.onnx")
    rng = np.random.default_rng(20261028)
    lhs = rng.normal(size=shape).astype("<f4")
    rhs = rng.normal(size=shape).astype("<f4")
    lhs.tofile(args.output / "lhs.raw")
    rhs.tofile(args.output / "rhs.raw")
    (lhs + rhs).astype("<f4").tofile(args.output / "reference.raw")
    (args.output / "input_list.txt").write_text("lhs:=lhs.raw rhs:=rhs.raw\n", encoding="utf-8")
    (args.output / "graph_manifest.json").write_text(json.dumps({"seed": 20261028, "shape": shape, "dtype": "fp32"}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
