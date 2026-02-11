# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Where operator."""

from __future__ import annotations

__all__ = [
    "infer_where",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _broadcast, _context, _registry


@_registry.registry.register("", "Where", since_version=9)
def infer_where(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Where operator.

    Output shape is the broadcast of condition, X, and Y.
    Output dtype is the same as X.

    Spec: https://onnx.ai/onnx/operators/onnx__Where.html
    """
    (condition, x, y) = _context.check_inputs(node, "condition", "X", "Y")

    output_dtype = x.dtype or y.dtype

    # Broadcast all three shapes
    shape_xy = _broadcast.broadcast_shapes(x.shape, y.shape)
    output_shape = _broadcast.broadcast_shapes(condition.shape, shape_xy)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
