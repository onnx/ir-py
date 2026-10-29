# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fit a tiny linear regressor and inspect its sensitivities using ONNX autodiff.

Run from the repository root with ``python tools/examples/autodiff.py`` after
installing onnx-ir from this checkout. Only ONNX and NumPy are needed.

The original model computes predictions and mean-squared error for a symbolic
batch size. This example prints the before/after IR graphs, checks parameter
gradients against NumPy, takes one loss-reducing optimization step, and computes
both directional prediction sensitivities and a reverse-mode attribution.

All derivative models are ordinary, serializable ONNX models. For example,
``ir.save(gradient_model, "linear_regression_grad.onnx")`` exports the executable
gradient graph, not a Python callback or a training-domain operator.
"""

from __future__ import annotations

import numpy as np
import onnx  # ruff: ignore[banned-api]
from onnx.reference import ReferenceEvaluator  # ruff: ignore[banned-api]

import onnx_ir as ir
from onnx_ir.passes.common import GradPass, JvpPass, VjpPass


def linear_regression(weight: np.ndarray, bias: np.ndarray) -> ir.Model:
    """Create prediction = features @ weight + bias and its scalar MSE loss."""
    features = ir.val("features", dtype=ir.DataType.DOUBLE, shape=["batch", 2])
    target = ir.val("target", dtype=ir.DataType.DOUBLE, shape=["batch", 1])
    tape = ir.tape.Tape()
    w = tape.initializer(ir.tensor(weight, name="weight"))
    b = tape.initializer(ir.tensor(bias, name="bias"))
    prediction = tape.op("Add", [tape.op("MatMul", [features, w]), b])
    prediction.name = "prediction"
    residual = tape.op("Sub", [prediction, target])
    loss = tape.op("ReduceMean", [tape.op("Mul", [residual, residual])], {"keepdims": 0})
    loss.name = "mse"
    return ir.Model(
        ir.Graph(
            [features, target],
            [prediction, loss],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="linear_regression",
        ),
        ir_version=9,
    )


def evaluate(model: ir.Model, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Check and execute a generated graph without an autodiff runtime."""
    proto = onnx.shape_inference.infer_shapes(ir.to_proto(model), strict_mode=True)
    onnx.checker.check_model(proto, full_check=True)
    return ReferenceEvaluator(proto).run(None, feeds)


def main() -> None:
    weight = np.array([[0.2], [-0.1]])
    bias = np.array([0.3])
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    target = features @ np.array([[1.5], [-0.5]]) + 0.7
    feeds = {"features": features, "target": target}
    original = linear_regression(weight, bias)
    source_bytes = ir.to_proto(original).SerializeToString()
    print("BEFORE: predictions and scalar MSE")
    print(original)

    gradient_model = GradPass(inputs=["weight", "bias"], output="mse")(original).model
    print("\nAFTER GradPass: predictions, MSE, weight gradient, bias gradient")
    print(gradient_model)
    prediction, loss, grad_weight, grad_bias = evaluate(gradient_model, feeds)
    residual = prediction - target
    expected_weight = 2 * features.T @ residual / residual.size
    expected_bias = 2 * residual.mean(axis=0)
    np.testing.assert_allclose(grad_weight, expected_weight)
    np.testing.assert_allclose(grad_bias, expected_bias)
    updated = linear_regression(weight - 0.1 * grad_weight, bias - 0.1 * grad_bias)
    _, updated_loss = evaluate(updated, feeds)
    assert updated_loss < loss
    print(f"MSE before/after one gradient step: {loss:.6f} -> {updated_loss:.6f}")
    print("d(MSE)/d(weight):", grad_weight.ravel())
    print("d(MSE)/d(bias):", grad_bias)

    jvp_model = JvpPass(inputs=["features"], outputs=["prediction"])(original).model
    direction = np.broadcast_to(np.array([1.0, 0.0]), features.shape).copy()
    jvp_feeds = {**feeds, jvp_model.graph.inputs[-1].name: direction}
    print("\nAFTER JvpPass: sensitivity to increasing the first feature")
    print(jvp_model)
    _, _, prediction_tangent = evaluate(jvp_model, jvp_feeds)
    np.testing.assert_allclose(prediction_tangent, direction @ weight)
    print("Directional prediction derivative:", prediction_tangent.ravel())

    vjp_model = VjpPass(inputs=["features"], outputs=["prediction"])(original).model
    cotangent = np.array([[1.0], [0.0], [0.0], [0.0]])
    vjp_feeds = {**feeds, vjp_model.graph.inputs[-1].name: cotangent}
    print("\nAFTER VjpPass: attribute the first prediction to its input features")
    print(vjp_model)
    _, _, feature_cotangent = evaluate(vjp_model, vjp_feeds)
    np.testing.assert_allclose(feature_cotangent, cotangent @ weight.T)
    print("Feature cotangent:\n", feature_cotangent)
    np.testing.assert_allclose(
        np.sum(prediction_tangent * cotangent),
        np.sum(direction * feature_cotangent),
    )
    assert ir.to_proto(original).SerializeToString() == source_bytes


if __name__ == "__main__":
    main()
