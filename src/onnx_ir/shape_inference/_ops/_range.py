# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Range operator."""

from __future__ import annotations

__all__ = [
    "infer_range",
]

import math

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Range", since_version=11)
def infer_range(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Range operator."""
    (start, _limit, _delta) = _context.check_inputs(node, "start", "limit", "delta")

    output_len: int | ir.SymbolicDim = ctx.new_symbolic_dim()

    start_const = ir.convenience.get_const_tensor(node.inputs[0])  # type: ignore[arg-type]
    limit_const = ir.convenience.get_const_tensor(node.inputs[1])  # type: ignore[arg-type]
    delta_const = ir.convenience.get_const_tensor(node.inputs[2])  # type: ignore[arg-type]
    if start_const is not None and limit_const is not None and delta_const is not None:
        s = float(start_const.numpy().item())
        lim = float(limit_const.numpy().item())
        d = float(delta_const.numpy().item())
        output_len = max(0, math.ceil((lim - s) / d))

    output_shape = ir.Shape([output_len])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, start.dtype)
