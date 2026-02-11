# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Constant operator."""

from __future__ import annotations

__all__ = [
    "infer_constant",
]

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir.shape_inference import _registry

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context


@_registry.registry.register("", "Constant", since_version=1)
def infer_constant(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Constant operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Constant.html
    """
    if len(node.outputs) < 1:
        return

    output = node.outputs[0]

    value_attr = node.attributes.get("value")
    if value_attr is not None:
        tensor = value_attr.as_tensor()
        ctx.set_shape_and_dtype(output, tensor.shape, tensor.dtype)  # type: ignore[arg-type]
        return

    # Handle scalar constant attributes
    value_float = node.attributes.get("value_float")
    if value_float is not None:
        ctx.set_shape_and_dtype(output, ir.Shape([]), ir.DataType.FLOAT)
        return

    value_int = node.attributes.get("value_int")
    if value_int is not None:
        ctx.set_shape_and_dtype(output, ir.Shape([]), ir.DataType.INT64)
        return

    value_floats = node.attributes.get("value_floats")
    if value_floats is not None:
        ctx.set_shape_and_dtype(
            output, ir.Shape([len(value_floats.as_floats())]), ir.DataType.FLOAT
        )
        return

    value_ints = node.attributes.get("value_ints")
    if value_ints is not None:
        ctx.set_shape_and_dtype(
            output, ir.Shape([len(value_ints.as_ints())]), ir.DataType.INT64
        )
        return

    value_string = node.attributes.get("value_string")
    if value_string is not None:
        ctx.set_shape_and_dtype(output, ir.Shape([]), ir.DataType.STRING)
        return

    value_strings = node.attributes.get("value_strings")
    if value_strings is not None:
        ctx.set_shape_and_dtype(
            output, ir.Shape([len(value_strings.as_strings())]), ir.DataType.STRING
        )
        return
