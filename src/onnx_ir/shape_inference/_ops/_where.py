# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Where operator."""

from __future__ import annotations

__all__ = [
    "infer_where",
]

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir.shape_inference import _broadcast, _registry

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context


@_registry.registry.register("", "Where", since_version=9)
def infer_where(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Where operator.

    Output shape is the broadcast of condition, X, and Y.
    Output dtype is the same as X.

    Spec: https://onnx.ai/onnx/operators/onnx__Where.html
    """
    if len(node.inputs) < 3:
        ctx.record_error(node, f"Expected 3 inputs, got {len(node.inputs)}")
        return

    condition = node.inputs[0]
    x = node.inputs[1]
    y = node.inputs[2]
    if condition is None or x is None or y is None:
        return

    output_dtype = x.dtype or y.dtype

    # Broadcast all three shapes
    shape_xy = _broadcast.broadcast_shapes(x.shape, y.shape)
    output_shape = _broadcast.broadcast_shapes(condition.shape, shape_xy)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
