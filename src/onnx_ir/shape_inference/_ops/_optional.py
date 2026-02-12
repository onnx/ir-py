# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Optional operators."""

from __future__ import annotations

__all__ = [
    "infer_optional_get_element",
    "infer_optional_has_element",
    "infer_optional_op",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Optional", since_version=15)
def infer_optional_op(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Optional operator.

    Wraps the input in an OptionalType.
    """
    if len(node.inputs) > 0 and node.inputs[0] is not None:
        input_val = node.inputs[0]
        if len(node.outputs) > 0:
            input_type = input_val.type
            if input_type is not None:
                ctx.set_type(node.outputs[0], ir.OptionalType(input_type))
            else:
                ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)


@_registry.registry.register("", "OptionalGetElement", since_version=18)
def infer_optional_get_element(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for OptionalGetElement operator.

    Unwraps the optional type to extract the element type.
    """
    if len(node.inputs) > 0 and node.inputs[0] is not None:
        input_val = node.inputs[0]
        if len(node.outputs) > 0:
            input_type = input_val.type
            if isinstance(input_type, ir.OptionalType):
                elem_type = input_type.elem_type
                if elem_type is not None:
                    ctx.set_type(node.outputs[0], elem_type)
                    return
            # Fall back to passthrough for non-optional types
            if input_type is not None and not isinstance(input_type, ir.TensorType):
                ctx.set_type(node.outputs[0], input_type)
            else:
                ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)


@_registry.registry.register("", "OptionalHasElement", since_version=18)
def infer_optional_has_element(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for OptionalHasElement operator.

    Output: scalar BOOL.
    """
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape([]), ir.DataType.BOOL)
