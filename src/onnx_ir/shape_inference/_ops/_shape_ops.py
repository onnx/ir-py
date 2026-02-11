# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Shape, Size, and Flatten operators."""

from __future__ import annotations

__all__ = [
    "infer_flatten",
    "infer_shape",
    "infer_size",
]

import math
from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir.shape_inference import _registry

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context


@_registry.registry.register("", "Shape", since_version=1)
def infer_shape(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Shape operator.

    Output is a 1-D INT64 tensor whose length equals the input rank.

    Spec: https://onnx.ai/onnx/operators/onnx__Shape.html
    """
    if len(node.inputs) < 1:
        ctx.record_error(node, f"Expected 1 input, got {len(node.inputs)}")
        return

    data = node.inputs[0]
    if data is None:
        return

    output_shape: ir.Shape | None = None
    if data.shape is not None:
        # Since opset 15, start/end attributes can slice the shape
        start_attr = node.attributes.get("start")
        end_attr = node.attributes.get("end")
        rank = data.shape.rank()
        start = start_attr.as_int() if start_attr is not None else 0
        end = end_attr.as_int() if end_attr is not None else rank

        if start < 0:
            start += rank
        if end < 0:
            end += rank
        start = max(0, min(start, rank))
        end = max(0, min(end, rank))

        output_shape = ir.Shape([max(0, end - start)])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.INT64)


@_registry.registry.register("", "Size", since_version=1)
def infer_size(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Size operator.

    Output is a scalar INT64 tensor.

    Spec: https://onnx.ai/onnx/operators/onnx__Size.html
    """
    if len(node.inputs) < 1:
        ctx.record_error(node, f"Expected 1 input, got {len(node.inputs)}")
        return

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape([]), ir.DataType.INT64)


@_registry.registry.register("", "Flatten", since_version=1)
def infer_flatten(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Flatten operator.

    Reshapes input to 2-D: (product of dims[:axis], product of dims[axis:]).

    Spec: https://onnx.ai/onnx/operators/onnx__Flatten.html
    """
    if len(node.inputs) < 1:
        ctx.record_error(node, f"Expected 1 input, got {len(node.inputs)}")
        return

    data = node.inputs[0]
    if data is None:
        return

    input_shape = data.shape
    input_dtype = data.dtype

    output_shape: ir.Shape | None = None
    if input_shape is not None:
        axis_attr = node.attributes.get("axis")
        axis = axis_attr.as_int() if axis_attr is not None else 1

        rank = input_shape.rank()
        if axis < 0:
            axis += rank

        if input_shape.is_static():
            left = math.prod(d if isinstance(d, int) else 1 for d in input_shape.dims[:axis])
            right = math.prod(d if isinstance(d, int) else 1 for d in input_shape.dims[axis:])
            output_shape = ir.Shape([left, right])
        else:
            output_shape = ir.Shape([ctx.new_symbolic_dim(), ctx.new_symbolic_dim()])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)
