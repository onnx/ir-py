# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Cast operator."""

from __future__ import annotations

__all__ = [
    "infer_cast",
]

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir.shape_inference import _registry

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context


@_registry.registry.register("", "Cast", since_version=6)
def infer_cast(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Cast operator.

    Shape is identical to the input; dtype comes from the ``to`` attribute.

    Spec: https://onnx.ai/onnx/operators/onnx__Cast.html
    """
    if len(node.inputs) < 1:
        ctx.record_error(node, f"Expected 1 input, got {len(node.inputs)}")
        return

    data = node.inputs[0]
    if data is None:
        return

    to_attr = node.attributes.get("to")
    if to_attr is None:
        ctx.record_error(node, "Missing required attribute 'to'")
        return

    output_dtype = ir.DataType(to_attr.as_int())

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, output_dtype)
