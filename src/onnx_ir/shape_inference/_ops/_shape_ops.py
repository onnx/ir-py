# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Shape, Size, Flatten, and Det operators."""

from __future__ import annotations

__all__ = [
    "infer_det",
    "infer_flatten",
    "infer_shape",
    "infer_size",
]

import functools
import operator

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Shape", since_version=1)
def infer_shape(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Shape operator.

    Output is a 1-D INT64 tensor whose length equals the input rank.

    Spec: https://onnx.ai/onnx/operators/onnx__Shape.html
    """
    (data,) = _context.check_inputs(node, "data")

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
    _context.check_inputs(node, "data")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape([]), ir.DataType.INT64)


@_registry.registry.register("", "Flatten", since_version=1)
def infer_flatten(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Flatten operator.

    Reshapes input to 2-D: (product of dims[:axis], product of dims[axis:]).

    Spec: https://onnx.ai/onnx/operators/onnx__Flatten.html
    """
    (data,) = _context.check_inputs(node, "data")

    input_shape = data.shape
    input_dtype = data.dtype

    output_shape: ir.Shape | None = None
    if input_shape is not None:
        axis_attr = node.attributes.get("axis")
        axis = axis_attr.as_int() if axis_attr is not None else 1

        rank = input_shape.rank()
        if axis < 0:
            axis += rank

        left: int | ir.SymbolicDim = functools.reduce(operator.mul, input_shape.dims[:axis], 1)
        right: int | ir.SymbolicDim = functools.reduce(
            operator.mul, input_shape.dims[axis:], 1
        )
        output_shape = ir.Shape([left, right])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)


@_registry.registry.register("", "Det", since_version=11)
def infer_det(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Det operator.

    If input X has shape [..., M, M], output has shape [...].
    """
    (x,) = _context.check_inputs(node, "X")

    output_shape: ir.Shape | None = None
    if x.shape is not None and x.shape.rank() >= 2:
        output_shape = ir.Shape(list(x.shape.dims[:-2]))

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)
