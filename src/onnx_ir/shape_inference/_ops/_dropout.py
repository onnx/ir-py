# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Dropout operator."""

from __future__ import annotations

__all__ = [
    "infer_dropout",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Dropout", since_version=7)
def infer_dropout(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Dropout operator.

    Output[0] has the same shape and dtype as input[0].
    Output[1] (optional mask) has the same shape as input[0] with dtype BOOL.

    Spec: https://onnx.ai/onnx/operators/onnx__Dropout.html
    """
    (data,) = _context.check_inputs(node, "data")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, data.dtype)

    # Optional mask output (opset 12+)
    if len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], data.shape, ir.DataType.BOOL)
