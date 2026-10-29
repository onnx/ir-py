# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Save before/after vmap graphs and verify practical dynamic-batch examples.

Run ``python examples/vmap.py --output-dir vmap_models`` from the repository root.
The examples use NumPy, ONNX's reference evaluator, and ONNX IR only.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnx  # ruff: ignore[banned-api]
from onnx.reference import ReferenceEvaluator  # ruff: ignore[banned-api]

import onnx_ir as ir
from onnx_ir.passes.common import VmapPass


def _initializer(name, array):
    tensor = ir.tensor(array)
    return ir.val(name, dtype=tensor.dtype, shape=tensor.shape, const_value=tensor)


def _op(nodes, op, inputs, name, **attributes):
    node = ir.node(op, inputs, attributes)
    node.outputs[0].name = name
    node.outputs[0].dtype = ir.DataType.FLOAT
    nodes.append(node)
    return node.outputs[0]


def _model(name, inputs, outputs, nodes, initializers):
    model = ir.Model(
        ir.Graph(
            inputs,
            outputs,
            nodes=nodes,
            initializers=initializers,
            opset_imports={"": 18},
            name=name,
        ),
        ir_version=10,
    )
    return ir.from_proto(onnx.shape_inference.infer_shapes(ir.to_proto(model)))


def image_classifier():
    """A small convolutional classifier with shared, runtime channel calibration."""
    rng = np.random.default_rng(42)
    image = ir.val("image", dtype=ir.DataType.FLOAT, shape=[1, 1, 8, 8])
    calibration = ir.val("calibration", dtype=ir.DataType.FLOAT, shape=[4])
    kernel = _initializer("kernel", rng.normal(size=(4, 1, 3, 3)).astype(np.float32) / 5)
    projection = _initializer("projection", rng.normal(size=(4, 3)).astype(np.float32))
    bias = _initializer("bias", np.array([-0.2, 0.1, 0.3], dtype=np.float32))
    nodes = []
    features = _op(nodes, "Conv", [image, kernel], "convolution", pads=[1, 1, 1, 1])
    features = _op(nodes, "Relu", [features], "activation")
    features = _op(nodes, "GlobalAveragePool", [features], "pooled")
    features = _op(nodes, "Flatten", [features], "features", axis=1)
    features = _op(nodes, "Add", [features, calibration], "calibrated")
    logits = _op(nodes, "Gemm", [features, projection, bias], "logits")
    probabilities = _op(nodes, "Softmax", [logits], "probabilities", axis=-1)
    return _model(
        "image_classifier",
        [image, calibration],
        [probabilities],
        nodes,
        [kernel, projection, bias],
    )


def recommendation():
    """Compare a user's pooled history embedding with a candidate embedding."""
    history = ir.val("history_ids", dtype=ir.DataType.INT64, shape=[3])
    candidate = ir.val("candidate", dtype=ir.DataType.FLOAT, shape=[4])
    table = ir.val("embedding_table", dtype=ir.DataType.FLOAT, shape=[16, 4])
    axes = _initializer("history_axis", np.array([0], dtype=np.int64))
    nodes = []
    embeddings = _op(nodes, "Gather", [table, history], "history_embeddings", axis=0)
    pooled = _op(nodes, "ReduceMean", [embeddings, axes], "pooled_history", keepdims=0)
    normalized = _op(nodes, "LpNormalization", [pooled], "user_embedding", axis=0, p=2)
    normalized_candidate = _op(
        nodes, "LpNormalization", [candidate], "candidate_embedding", axis=0, p=2
    )
    score = _op(nodes, "MatMul", [normalized, normalized_candidate], "cosine_score")
    return _model(
        "recommendation",
        [history, candidate, table],
        [normalized, score],
        nodes,
        [axes],
    )


def _image_feeds(batch_size, rng):
    return {
        "image": rng.normal(size=(batch_size, 1, 1, 8, 8)).astype(np.float32),
        "calibration": np.array([0.1, -0.2, 0.05, 0.3], dtype=np.float32),
    }


def _recommendation_feeds(batch_size, rng):
    return {
        "history_ids": rng.integers(0, 16, size=(3, batch_size), dtype=np.int64),
        "candidate": rng.normal(size=(batch_size, 4)).astype(np.float32),
        "embedding_table": rng.normal(size=(16, 4)).astype(np.float32),
    }


def _run_example(name, model, in_axes, out_axes, make_feeds, output_dir):
    batched = VmapPass(in_axes, out_axes=out_axes)(model).model
    paths = {}
    for kind, current in (("single", model), ("batched", batched)):
        proto = ir.to_proto(current)
        onnx.checker.check_model(proto, full_check=True)
        path = os.path.join(output_dir, f"{name}.{kind}.onnx")
        ir.save(current, path)
        paths[kind] = path
        text = (
            "\n".join(line.rstrip() for line in onnx.printer.to_text(proto).splitlines())
            + "\n"
        )
        with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as file:
            file.write(text)
        print(f"\n=== {name}: {kind} graph ===\n{text}")
    single = ReferenceEvaluator(onnx.load(paths["single"]))
    vectorized = ReferenceEvaluator(onnx.load(paths["batched"]))
    rng = np.random.default_rng(2026)
    for batch_size in (1, 3, 7):
        feeds = make_feeds(batch_size, rng)
        actual = vectorized.run(None, feeds)
        independent = [
            single.run(
                None,
                {
                    value.name: feeds[value.name]
                    if axis is None
                    else np.take(feeds[value.name], i, axis=axis)
                    for value, axis in zip(model.graph.inputs, in_axes)
                },
            )
            for i in range(batch_size)
        ]
        for j, (result, axis) in enumerate(zip(actual, out_axes)):
            expected = np.stack([outputs[j] for outputs in independent], axis=axis)
            np.testing.assert_allclose(result, expected, rtol=2e-5, atol=1e-6)
        print(
            f"{name}: batch_size={batch_size}, independent execution matches; "
            f"output shapes: {[value.shape for value in actual]}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="vmap_models")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    _run_example(
        "image_classifier",
        image_classifier(),
        (0, None),
        (0,),
        _image_feeds,
        args.output_dir,
    )
    _run_example(
        "recommendation",
        recommendation(),
        (-1, 0, None),
        (-1, 0),
        _recommendation_feeds,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
