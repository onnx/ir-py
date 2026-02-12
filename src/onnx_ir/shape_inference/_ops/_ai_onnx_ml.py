# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for ai.onnx.ml operators."""

from __future__ import annotations

__all__ = [
    "infer_array_feature_extractor",
    "infer_binarizer",
    "infer_label_encoder",
    "infer_tree_ensemble",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


@_reg("ai.onnx.ml", "ArrayFeatureExtractor", since_version=1)
def infer_array_feature_extractor(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape for ArrayFeatureExtractor.

    Output shape is ``X.shape[:-1] + [Y.shape[-1]]`` where Y is the indices.
    """
    (x, y) = _context.check_inputs(node, "X", "Y")
    if len(node.outputs) == 0:
        return

    x_shape = x.shape
    y_shape = y.shape
    if x_shape is not None and y_shape is not None and y_shape.rank() >= 1:
        out_dims = list(x_shape.dims[:-1]) + [y_shape.dims[-1]]
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(out_dims), x.dtype)
    else:
        ctx.set_shape_and_dtype(node.outputs[0], None, x.dtype)


@_reg("ai.onnx.ml", "Binarizer", since_version=1)
def infer_binarizer(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape for Binarizer. Output shape and dtype match input."""
    (x,) = _context.check_inputs(node, "X")
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)


@_reg("ai.onnx.ml", "LabelEncoder", since_version=1)
def infer_label_encoder(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape for LabelEncoder. Output shape = input shape, dtype from values."""
    (x,) = _context.check_inputs(node, "X")
    if len(node.outputs) == 0:
        return

    # Determine output dtype from the values attribute
    output_dtype: ir.DataType | None = None
    if node.attributes.get("values_int64s") is not None:
        output_dtype = ir.DataType.INT64
    elif node.attributes.get("values_strings") is not None:
        output_dtype = ir.DataType.STRING
    elif node.attributes.get("values_floats") is not None:
        output_dtype = ir.DataType.FLOAT
    elif node.attributes.get("values_tensor") is not None:
        values_tensor = node.attributes["values_tensor"].as_tensor()
        if values_tensor is not None:
            output_dtype = ir.DataType(values_tensor.dtype)

    ctx.set_shape_and_dtype(node.outputs[0], x.shape, output_dtype)


@_reg("ai.onnx.ml", "TreeEnsemble", since_version=5)
def infer_tree_ensemble(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape for TreeEnsemble. Output is ``[N, n_targets]``."""
    (x,) = _context.check_inputs(node, "X")
    if len(node.outputs) == 0:
        return

    n_targets_attr = node.attributes.get("n_targets")
    n_targets = n_targets_attr.as_int() if n_targets_attr is not None else None

    batch_dim = x.shape.dims[0] if x.shape is not None and x.shape.rank() >= 1 else None

    if batch_dim is not None and n_targets is not None:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape([batch_dim, n_targets]), x.dtype)
    elif n_targets is not None:
        ctx.set_shape_and_dtype(node.outputs[0], None, x.dtype)
    else:
        ctx.set_shape_and_dtype(node.outputs[0], None, x.dtype)
