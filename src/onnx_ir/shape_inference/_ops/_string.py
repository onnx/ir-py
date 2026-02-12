# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for string operators."""

from __future__ import annotations

__all__ = [
    "infer_string_normalizer",
    "infer_string_split",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "StringSplit", since_version=20)
def infer_string_split(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for StringSplit operator.

    Y: [*X.shape, max_splits] where max_splits is symbolic.
    Z: same shape as X (number of substrings per element).
    """
    (x,) = _context.check_inputs(node, "X")

    # Y: split strings — rank is X.rank + 1 with symbolic last dim
    if len(node.outputs) > 0:
        if x.shape is not None:
            y_shape = ir.Shape([*x.shape, ctx.new_symbolic_dim()])
        else:
            y_shape = None
        ctx.set_shape_and_dtype(node.outputs[0], y_shape, ir.DataType.STRING)
    # Z: number of splits per element — same shape as X
    if len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], x.shape, ir.DataType.INT64)


@_registry.registry.register("", "StringNormalizer", since_version=10)
def infer_string_normalizer(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for StringNormalizer operator.

    Output shape may differ from input when stopwords filtering is applied,
    so the affected dimension is symbolic.
    """
    (x,) = _context.check_inputs(node, "X")

    output_shape: ir.Shape | None = None
    if x.shape is not None:
        stopwords_attr = node.attributes.get("stopwords")
        has_stopwords = stopwords_attr is not None and len(stopwords_attr.as_strings()) > 0
        if has_stopwords:
            # Stopwords filtering may change output size — make it symbolic
            new_dims: list[int | ir.SymbolicDim] = [
                ctx.new_symbolic_dim() if isinstance(d, int) or d.value is not None else d
                for d in x.shape.dims
            ]
            output_shape = ir.Shape(new_dims)
        else:
            output_shape = x.shape

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.STRING)
