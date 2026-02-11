# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Softmax, LogSoftmax and Hardmax operators."""

from __future__ import annotations

__all__ = [
    "infer_softmax",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


@_reg("", "Softmax", since_version=1)
@_reg("", "LogSoftmax", since_version=1)
@_reg("", "Hardmax", since_version=1)
def infer_softmax(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Softmax / LogSoftmax / Hardmax.

    Output shape and dtype are the same as the input.  The ``axis`` attribute
    default differs across opset versions (1 for opset <13, -1 for ≥13) but
    does not affect the output shape.

    Spec: https://onnx.ai/onnx/operators/onnx__Softmax.html
    """
    (data,) = _context.check_inputs(node, "input")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, data.dtype)
