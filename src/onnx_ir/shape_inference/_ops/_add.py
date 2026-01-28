# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for element-wise operators (Add, Sub, Mul, etc.)."""

from __future__ import annotations

__all__ = [
    "infer_add",
]

from typing import TYPE_CHECKING

from onnx_ir.shape_inference._broadcast import broadcast_shapes
from onnx_ir.shape_inference._registry import registry

if TYPE_CHECKING:
    import onnx_ir as ir
    from onnx_ir.shape_inference._context import ShapeInferenceContext


@registry.register("", "Add", versions=7)
def infer_add(ctx: ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Add operator.

    Add performs element-wise addition with NumPy-style broadcasting.
    Output shape is the broadcast of the two input shapes.
    Output dtype is the same as input dtype (inputs must have same dtype).

    Spec: https://onnx.ai/onnx/operators/onnx__Add.html
    """
    if len(node.inputs) < 2:
        return

    input_a = node.inputs[0]
    input_b = node.inputs[1]

    if input_a is None or input_b is None:
        return

    shape_a = input_a.shape
    shape_b = input_b.shape

    output_shape = broadcast_shapes(shape_a, shape_b)

    # Output dtype is same as input dtype (ONNX requires inputs have same dtype)
    output_dtype = input_a.dtype or input_b.dtype

    if len(node.outputs) > 0:
        output = node.outputs[0]
        ctx.set_shape_and_dtype(output, output_shape, output_dtype)
