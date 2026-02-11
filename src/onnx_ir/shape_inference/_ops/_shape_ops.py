# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Shape, Size, Flatten, and other shape-related operators."""

from __future__ import annotations

__all__ = [
    "infer_compress",
    "infer_det",
    "infer_einsum",
    "infer_flatten",
    "infer_image_decoder",
    "infer_mel_weight_matrix",
    "infer_non_max_suppression",
    "infer_non_zero",
    "infer_optional_get_element",
    "infer_optional_has_element",
    "infer_optional_op",
    "infer_sequence_map",
    "infer_shape",
    "infer_size",
    "infer_string_normalizer",
    "infer_string_split",
    "infer_tensor_scatter",
    "infer_tfidf_vectorizer",
    "infer_unique",
]

import math

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

        if input_shape.is_static():
            left = math.prod(d if isinstance(d, int) else 1 for d in input_shape.dims[:axis])
            right = math.prod(d if isinstance(d, int) else 1 for d in input_shape.dims[axis:])
            output_shape = ir.Shape([left, right])
        else:
            output_shape = ir.Shape([ctx.new_symbolic_dim(), ctx.new_symbolic_dim()])

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


@_registry.registry.register("", "NonZero", since_version=13)
def infer_non_zero(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for NonZero operator.

    Output: [rank(X), num_nonzero], dtype=INT64.
    """
    (x,) = _context.check_inputs(node, "X")

    if len(node.outputs) > 0:
        if x.shape is not None:
            output_shape = ir.Shape([x.shape.rank(), ctx.new_symbolic_dim("_nonzero")])
        else:
            output_shape = ir.Shape(
                [ctx.new_symbolic_dim("_nonzero_r"), ctx.new_symbolic_dim("_nonzero")]
            )
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.INT64)


@_registry.registry.register("", "Compress", since_version=11)
def infer_compress(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Compress operator.

    Output: 1-D with dynamic length.
    """
    (x,) = _context.check_inputs(node, "input")

    if len(node.outputs) > 0:
        output_shape = ir.Shape([ctx.new_symbolic_dim("_compress")])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)


@_registry.registry.register("", "Unique", since_version=11)
def infer_unique(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Unique operator.

    All outputs have dynamic shapes.
    """
    (x,) = _context.check_inputs(node, "X")

    unique_len = ctx.new_symbolic_dim("_unique")

    # Y: unique values — 1-D with dynamic length
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape([unique_len]), x.dtype)
    # indices — 1-D, same length as Y
    if len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], ir.Shape([unique_len]), ir.DataType.INT64)
    # inverse_indices — 1-D, same length as input (or flattened input)
    if len(node.outputs) > 2:
        inv_len = ctx.new_symbolic_dim("_unique_inv")
        ctx.set_shape_and_dtype(node.outputs[2], ir.Shape([inv_len]), ir.DataType.INT64)
    # counts — 1-D, same length as Y
    if len(node.outputs) > 3:
        ctx.set_shape_and_dtype(node.outputs[3], ir.Shape([unique_len]), ir.DataType.INT64)


@_registry.registry.register("", "NonMaxSuppression", since_version=10)
def infer_non_max_suppression(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for NonMaxSuppression operator.

    Output: [selected_indices_count, 3], dtype=INT64.
    """
    _context.check_inputs(node, "boxes", "scores")

    output_shape = ir.Shape([ctx.new_symbolic_dim("_nms"), 3])
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.INT64)


@_registry.registry.register("", "StringSplit", since_version=22)
def infer_string_split(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for StringSplit operator.

    Y: [*X.shape, max_splits] where max_splits is symbolic.
    Z: same shape as X (number of substrings per element).
    """
    (x,) = _context.check_inputs(node, "X")

    # Y: split strings — rank is X.rank + 1 with symbolic last dim
    if len(node.outputs) > 0:
        if x.shape is not None:
            y_shape = ir.Shape([*x.shape, ctx.new_symbolic_dim("_strsplit")])
        else:
            y_shape = None
        ctx.set_shape_and_dtype(node.outputs[0], y_shape, ir.DataType.STRING)
    # Z: number of splits per element — same shape as X
    if len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], x.shape, ir.DataType.INT64)


@_registry.registry.register("", "StringNormalizer", since_version=10)
def infer_string_normalizer(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for StringNormalizer operator."""
    (x,) = _context.check_inputs(node, "X")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, ir.DataType.STRING)


@_registry.registry.register("", "Einsum", since_version=12)
def infer_einsum(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Einsum operator.

    Graceful degradation: output shape = None (equation parsing is complex).
    """
    if len(node.inputs) < 1 or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")

    output_dtype = node.inputs[0].dtype
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)


@_registry.registry.register("", "Scan", since_version=11)
def infer_scan(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Scan operator.

    State outputs keep body output shapes. Scan outputs get a symbolic scan dim prepended.
    """
    body_attr = node.attributes.get("body")
    if body_attr is not None:
        body_graph = body_attr.as_graph()
        if body_graph is not None:
            num_scan_inputs_attr = node.attributes.get("num_scan_inputs")
            num_scan_inputs = (
                num_scan_inputs_attr.as_int() if num_scan_inputs_attr is not None else 0
            )
            # Number of state variables = total inputs - scan inputs
            num_state = len(node.inputs) - num_scan_inputs
            for i, output in enumerate(node.outputs):
                if i < len(body_graph.outputs):
                    body_out = body_graph.outputs[i]
                    if i < num_state:
                        # State output: same shape as body state output
                        ctx.set_shape_and_dtype(output, body_out.shape, body_out.dtype)
                    else:
                        # Scan output: body scan output shape + scan dim prepended
                        if body_out.shape is not None:
                            scan_len = ctx.new_symbolic_dim("_scan")
                            out_shape = ir.Shape([scan_len, *body_out.shape])
                        else:
                            out_shape = None
                        ctx.set_shape_and_dtype(output, out_shape, body_out.dtype)
            return

    for output in node.outputs:
        ctx.set_shape_and_dtype(output, None, None)


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
        output_shape = ir.Shape([
            ctx.new_symbolic_dim("_img_h"),
            ctx.new_symbolic_dim("_img_w"),
            channels,
        ])
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
        output_shape = ir.Shape([
            ctx.new_symbolic_dim("_mel_freq"),
            ctx.new_symbolic_dim("_mel_bins"),
        ])
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
                feat_dim = ctx.new_symbolic_dim("_tfidf")
                if x.shape.rank() == 1:
                    output_shape = ir.Shape([feat_dim])
                elif x.shape.rank() == 2:
                    output_shape = ir.Shape([x.shape[0], feat_dim])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.FLOAT)


@_registry.registry.register("", "SequenceMap", since_version=17)
def infer_sequence_map(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for SequenceMap operator.

    Graceful degradation: leave outputs unchanged.
    """


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


@_registry.registry.register("", "TensorScatter", since_version=24)
def infer_tensor_scatter(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for TensorScatter operator.

    Output = data (first input) shape/dtype.
    """
    (data,) = _context.check_inputs(node, "data")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, data.dtype)
