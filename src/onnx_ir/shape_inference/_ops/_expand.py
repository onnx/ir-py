# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Expand operator."""

from __future__ import annotations

__all__ = [
    "infer_expand",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _broadcast, _context, _registry


@_registry.registry.register("", "Expand", since_version=8)
def infer_expand(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Expand operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Expand.html
    """
    (data, shape_input) = _context.check_inputs(node, "input", "shape")

    input_dtype = data.dtype
    input_shape = data.shape

    # Try to read the target shape from const_value
    target_shape: ir.Shape | None = None
    shape_const = shape_input.const_value
    if shape_const is not None:
        target_dims = [int(x) for x in shape_const.numpy().flatten()]
        target_shape = ir.Shape(target_dims)

    output_shape: ir.Shape | None = None
    if input_shape is not None and target_shape is not None:
        output_shape = _broadcast.broadcast_shapes(input_shape, target_shape)
    elif target_shape is not None:
        output_shape = target_shape

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)
