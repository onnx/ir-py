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
            output_shape = ir.Shape([x.shape.rank(), ctx.new_symbolic_dim()])
        else:
            output_shape = ir.Shape([ctx.new_symbolic_dim(), ctx.new_symbolic_dim()])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.INT64)


@_registry.registry.register("", "Compress", since_version=11)
def infer_compress(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Compress operator.

    Output: 1-D with dynamic length.
    """
    (x,) = _context.check_inputs(node, "input")

    if len(node.outputs) > 0:
        output_shape = ir.Shape([ctx.new_symbolic_dim()])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)


@_registry.registry.register("", "Unique", since_version=11)
def infer_unique(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Unique operator.

    All outputs have dynamic shapes.
    """
    (x,) = _context.check_inputs(node, "X")

    unique_len = ctx.new_symbolic_dim()

    # Y: unique values — 1-D with dynamic length
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape([unique_len]), x.dtype)
    # indices — 1-D, same length as Y
    if len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], ir.Shape([unique_len]), ir.DataType.INT64)
    # inverse_indices — 1-D, same length as input (or flattened input)
    if len(node.outputs) > 2:
        inv_len = ctx.new_symbolic_dim()
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

    output_shape = ir.Shape([ctx.new_symbolic_dim(), 3])
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
            y_shape = ir.Shape([*x.shape, ctx.new_symbolic_dim()])
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

    Parses the equation string to determine output shape from input shapes.
    Supports explicit (``->`` present) and implicit output forms, and ellipsis (``...``).
    """
    if len(node.inputs) < 1 or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")

    output_dtype = node.inputs[0].dtype

    equation_attr = node.attributes.get("equation")
    if equation_attr is None:
        raise _context.OpUsageError(node, "Missing required attribute 'equation'")
    equation = equation_attr.as_string().replace(" ", "")

    # Check all inputs have shapes
    for i, inp in enumerate(node.inputs):
        if inp is None or inp.shape is None:
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
            return

    output_shape = _einsum_shape(ctx, node, equation)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


def _einsum_shape(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    equation: str,
) -> ir.Shape:
    """Compute the output shape for an Einsum equation given input shapes."""

    def _is_letter(c: str) -> bool:
        return c.isalpha()

    mid_index = equation.find("->")
    left_equation = equation[:mid_index] if mid_index != -1 else equation

    # label_dims: maps each unique label char → the dimension it represents
    label_dims: dict[str, int | ir.SymbolicDim] = {}
    repeated_labels: set[str] = set()
    # Ordered list of unique labels (insertion order)
    label_order: list[str] = []

    ellipsis_dims: list[int | ir.SymbolicDim] = []
    num_ellipsis = 0
    num_ellipsis_indices = 0

    terms = left_equation.split(",")
    num_operands = len(terms)

    if num_operands != len(node.inputs):
        raise _context.OpUsageError(
            node,
            f"Number of inputs ({len(node.inputs)}) does not match "
            f"operands in equation ({num_operands})",
        )

    for operand_idx, term in enumerate(terms):
        inp = node.inputs[operand_idx]
        assert inp is not None and inp.shape is not None  # guaranteed by caller
        shape = inp.shape
        rank = shape.rank()

        ellipsis_pos = term.find("...")
        # Count letter indices in the term
        term_letters = sum(1 for c in term if _is_letter(c))

        if ellipsis_pos != -1:
            if rank < term_letters:
                raise _context.OpUsageError(
                    node,
                    f"Ellipsis in operand {operand_idx}: rank {rank} < "
                    f"letter indices {term_letters}",
                )
            local_ellipsis_dims = rank - term_letters

            if num_ellipsis == 0:
                num_ellipsis_indices = local_ellipsis_dims
            elif num_ellipsis_indices != local_ellipsis_dims:
                raise _context.OpUsageError(
                    node, "Ellipsis represents incompatible dimensions"
                )
            num_ellipsis += 1
        else:
            if rank != term_letters:
                raise _context.OpUsageError(
                    node,
                    f"Rank of input {operand_idx} ({rank}) does not match "
                    f"equation indices ({term_letters})",
                )

        # Walk through the term, mapping labels to dims
        shape_idx = 0  # index into the input shape
        i = 0
        while i < len(term):
            if i == ellipsis_pos:
                # Record ellipsis dims
                local_ellipsis_count = rank - term_letters
                if len(ellipsis_dims) == 0:
                    # First time seeing ellipsis — record dims
                    for j in range(local_ellipsis_count):
                        ellipsis_dims.append(shape[shape_idx + j])
                else:
                    # Broadcast: pick the larger of the two
                    for j in range(local_ellipsis_count):
                        existing = ellipsis_dims[j]
                        current = shape[shape_idx + j]
                        if isinstance(existing, int) and isinstance(current, int):
                            if existing == 1:
                                ellipsis_dims[j] = current
                            elif current != 1 and current != existing:
                                ellipsis_dims[j] = ctx.new_symbolic_dim()
                        elif existing == current:
                            pass
                        else:
                            # One is symbolic — keep the non-1 one or create new
                            if isinstance(existing, int) and existing == 1:
                                ellipsis_dims[j] = current
                            elif isinstance(current, int) and current == 1:
                                pass  # keep existing
                            else:
                                ellipsis_dims[j] = ctx.new_symbolic_dim()
                shape_idx += local_ellipsis_count
                i += 3  # skip "..."
                continue

            c = term[i]
            if _is_letter(c):
                dim = shape[shape_idx]
                if c not in label_dims:
                    label_dims[c] = dim
                    label_order.append(c)
                else:
                    repeated_labels.add(c)
                shape_idx += 1
            i += 1

    # Build output shape
    output_dims: list[int | ir.SymbolicDim] = []

    if mid_index != -1:
        # Explicit output
        right_equation = equation[mid_index + 2 :]
        right_ellipsis_pos = right_equation.find("...")

        i = 0
        while i < len(right_equation):
            if i == right_ellipsis_pos:
                output_dims.extend(ellipsis_dims[:num_ellipsis_indices])
                i += 3
                continue
            c = right_equation[i]
            if _is_letter(c):
                if c in label_dims:
                    output_dims.append(label_dims[c])
                else:
                    output_dims.append(ctx.new_symbolic_dim())
            i += 1
    else:
        # Implicit output: ellipsis dims first, then non-repeated labels in
        # alphabetical order (by ASCII: uppercase before lowercase, matching numpy)
        output_dims.extend(ellipsis_dims[:num_ellipsis_indices])
        for label in sorted(label_order):
            if label not in repeated_labels:
                output_dims.append(label_dims[label])

    return ir.Shape(output_dims)


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
                            scan_len = ctx.new_symbolic_dim()
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
        output_shape = ir.Shape(
            [
                ctx.new_symbolic_dim(),
                ctx.new_symbolic_dim(),
            ]
        )
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
