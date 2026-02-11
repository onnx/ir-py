# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Range operator."""

from __future__ import annotations

__all__ = [
    "infer_range",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Range", since_version=11)
def infer_range(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Range operator."""
    (start, _limit, _delta) = _context.check_inputs(node, "start", "limit", "delta")

    output_shape = ir.Shape([ctx.new_symbolic_dim()])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, start.dtype)
