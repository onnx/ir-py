# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for ConstantOfShape operator."""

from __future__ import annotations

__all__ = [
    "infer_constant_of_shape",
]

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir.shape_inference import _registry

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context


@_registry.registry.register("", "ConstantOfShape", since_version=9)
def infer_constant_of_shape(
    ctx: _context.ShapeInferenceContext, node: ir.Node
) -> None:
    """Infer shape and dtype for ConstantOfShape operator.

    Spec: https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html
    """
    if len(node.inputs) < 1:
        ctx.record_error(node, f"Expected 1 input, got {len(node.inputs)}")
        return

    shape_input = node.inputs[0]
    if shape_input is None:
        return

    # Determine output dtype from the value attribute (default: float32 zero)
    value_attr = node.attributes.get("value")
    if value_attr is not None:
        tensor = value_attr.as_tensor()
        output_dtype = tensor.dtype
    else:
        output_dtype = ir.DataType.FLOAT

    # Try to read shape from const_value
    output_shape: ir.Shape | None = None
    shape_const = shape_input.const_value
    if shape_const is not None:
        output_dims = [int(x) for x in shape_const.numpy().flatten()]
        output_shape = ir.Shape(output_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
