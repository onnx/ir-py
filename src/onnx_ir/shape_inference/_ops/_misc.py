# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for miscellaneous operators."""

from __future__ import annotations

__all__ = [
    "infer_eye_like",
    "infer_image_decoder",
    "infer_mel_weight_matrix",
    "infer_sequence_map",
    "infer_tfidf_vectorizer",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "ImageDecoder", since_version=20)
def infer_image_decoder(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for ImageDecoder operator.

    Output: [H, W, C] — 3-D with symbolic dims, dtype=UINT8.
    """
    _context.check_inputs(node, "encoded_stream")

    pixel_format = "RGB"
    pixel_format_attr = node.attributes.get("pixel_format")
    if pixel_format_attr is not None:
        pixel_format = pixel_format_attr.as_string()

    channels = {"RGB": 3, "BGR": 3, "Grayscale": 1}.get(pixel_format, 3)

    if len(node.outputs) > 0:
        output_shape = ir.Shape(
            [
                ctx.new_symbolic_dim(),
                ctx.new_symbolic_dim(),
                channels,
            ]
        )
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.UINT8)


@_registry.registry.register("", "MelWeightMatrix", since_version=17)
def infer_mel_weight_matrix(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for MelWeightMatrix operator.

    Output: 2-D [num_frequency_bins, num_mel_bins].
    """
    output_datatype_attr = node.attributes.get("output_datatype")
    output_dtype = (
        ir.DataType(output_datatype_attr.as_int())
        if output_datatype_attr is not None
        else ir.DataType.FLOAT
    )

    if len(node.outputs) > 0:
        num_freq: int | ir.SymbolicDim = ctx.new_symbolic_dim()
        num_mel: int | ir.SymbolicDim = ctx.new_symbolic_dim()

        if len(node.inputs) > 0 and node.inputs[0] is not None:
            bins_const = ir.convenience.get_const_tensor(node.inputs[0])
            if bins_const is not None:
                num_freq = int(bins_const.numpy().item())
        if len(node.inputs) > 1 and node.inputs[1] is not None:
            mel_const = ir.convenience.get_const_tensor(node.inputs[1])
            if mel_const is not None:
                num_mel = int(mel_const.numpy().item())

        output_shape = ir.Shape([num_freq, num_mel])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_registry.registry.register("", "TfIdfVectorizer", since_version=9)
def infer_tfidf_vectorizer(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for TfIdfVectorizer operator.

    Input 1-D → output 1-D [feature_dim]. Input 2-D → output 2-D [batch, feature_dim].
    """
    if len(node.outputs) > 0:
        output_shape = None
        if len(node.inputs) > 0 and node.inputs[0] is not None:
            x = node.inputs[0]
            if x.shape is not None:
                feat_dim = ctx.new_symbolic_dim()
                if x.shape.rank() == 1:
                    output_shape = ir.Shape([feat_dim])
                elif x.shape.rank() == 2:
                    output_shape = ir.Shape([x.shape[0], feat_dim])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.FLOAT)


@_registry.registry.register("", "SequenceMap", since_version=17)
def infer_sequence_map(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer type for SequenceMap operator.

    Output sequences have element types matching the body subgraph outputs.
    The body is run once per element. The engine infers body output types
    before calling this function.
    """
    body_attr = node.attributes.get("body")
    if body_attr is None:
        return
    body_graph = body_attr.as_graph()
    if body_graph is None:
        return

    for i, output in enumerate(node.outputs):
        if i >= len(body_graph.outputs):
            break
        body_out = body_graph.outputs[i]
        body_out_type = body_out.type
        if body_out_type is not None:
            ctx.set_type(output, ir.SequenceType(body_out_type))
        elif body_out.dtype is not None:
            ctx.set_type(output, ir.SequenceType(ir.TensorType(body_out.dtype)))


@_registry.registry.register("", "EyeLike", since_version=9)
def infer_eye_like(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for EyeLike operator.

    Output has the same shape as the input. dtype comes from the optional
    ``dtype`` attribute; when absent the input dtype is used.
    """
    (input_val,) = _context.check_inputs(node, "input")

    dtype_attr = node.attributes.get("dtype")
    if dtype_attr is not None:
        output_dtype: ir.DataType | None = ir.DataType(dtype_attr.as_int())
    else:
        output_dtype = input_val.dtype

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, output_dtype)
