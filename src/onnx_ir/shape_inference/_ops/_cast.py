# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Cast operator."""

from __future__ import annotations

__all__ = [
    "infer_cast",
    "infer_cast_like",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Cast", since_version=6)
def infer_cast(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Cast operator.

    Shape is identical to the input; dtype comes from the ``to`` attribute.

    Spec: https://onnx.ai/onnx/operators/onnx__Cast.html
    """
    (data,) = _context.check_inputs(node, "input")
    to_attr = _context.require_attr(node, "to")

    output_dtype = ir.DataType(to_attr.as_int())

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, output_dtype)

        # Propagate symbolic_value (casting does not change element values
        # for integer types used in shape computation)
        sym_val = ctx.get_symbolic_value(data)
        if sym_val is not None:
            ctx.set_symbolic_value(node.outputs[0], sym_val)


@_registry.registry.register("", "CastLike", since_version=15)
def infer_cast_like(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for CastLike operator.

    Shape is identical to input[0]; dtype comes from input[1].

    Spec: https://onnx.ai/onnx/operators/onnx__CastLike.html
    """
    (data, target) = _context.check_inputs(node, "input", "target_type")

    output_dtype = target.dtype
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, output_dtype)
