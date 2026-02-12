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

    Passes through input shape and dtype if available.
    """
    if len(node.inputs) > 0 and node.inputs[0] is not None:
        input_val = node.inputs[0]
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)


@_registry.registry.register("", "OptionalGetElement", since_version=18)
def infer_optional_get_element(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for OptionalGetElement operator.

    Passes through input shape and dtype if available.
    """
    if len(node.inputs) > 0 and node.inputs[0] is not None:
        input_val = node.inputs[0]
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)


@_registry.registry.register("", "OptionalHasElement", since_version=18)
def infer_optional_has_element(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for OptionalHasElement operator.

    Output: scalar BOOL.
    """
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape([]), ir.DataType.BOOL)
