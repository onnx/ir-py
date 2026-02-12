# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for control flow operators (If, Loop, Scan)."""

from __future__ import annotations

__all__ = [
    "infer_if",
    "infer_loop",
    "infer_scan",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


def _merge_shapes(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    shape1: ir.Shape,
    shape2: ir.Shape,
    output_idx: int,
) -> ir.Shape:
    """Merge two shapes from If branches into a compatible output shape.

    For each dimension pair:
    - If both are equal (concrete or symbolic), keep that value.
    - Otherwise (different concrete, different symbolic, mixed), create a new
      symbolic dim.

    Raises:
        OpUsageError: If ranks differ.
    """
    if shape1.rank() != shape2.rank():
        raise _context.OpUsageError(
            node,
            f"If output {output_idx}: rank mismatch between branches: "
            f"then={shape1.rank()}, else={shape2.rank()}",
        )

    result_dims: list[int | ir.SymbolicDim] = []
    for d1, d2 in zip(shape1.dims, shape2.dims):
        if isinstance(d1, int) and d1 == d2:
            result_dims.append(d1)
        elif (
            isinstance(d1, ir.SymbolicDim)
            and isinstance(d2, ir.SymbolicDim)
            and d1.value is not None
            and d1.value == d2.value
        ):
            result_dims.append(d1)
        else:
            result_dims.append(ctx.new_symbolic_dim())
    return ir.Shape(result_dims)


@_registry.registry.register("", "If", since_version=1)
def infer_if(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for If operator.

    The If operator takes a boolean condition and executes one of two
    subgraphs (then_branch / else_branch).  Each output of the If node
    corresponds to matching outputs of both branches.  The inferred
    shape is the *merge* of the two branch output shapes: dimensions
    that agree are kept; dimensions that differ become unknown.

    Spec: https://onnx.ai/onnx/operators/onnx__If.html
    """
    (_,) = _context.check_inputs(node, "cond")

    then_attr = _context.require_attr(node, "then_branch")
    else_attr = _context.require_attr(node, "else_branch")

    then_graph = then_attr.as_graph()
    else_graph = else_attr.as_graph()
    if then_graph is None or else_graph is None:
        return

    # NOTE: Subgraph inference is done by the engine (_engine._process_graph)
    # which recurses into subgraph attributes *before* calling the op's
    # infer function.  So by the time we get here, outputs of then/else
    # branches should already have shapes if they can be inferred.

    for i, output in enumerate(node.outputs):
        then_out = then_graph.outputs[i] if i < len(then_graph.outputs) else None
        else_out = else_graph.outputs[i] if i < len(else_graph.outputs) else None

        if then_out is None and else_out is None:
            continue

        # Handle non-tensor types (sequence, optional)
        then_type = then_out.type if then_out is not None else None
        else_type = else_out.type if else_out is not None else None

        if (then_type is not None and not isinstance(then_type, ir.TensorType)) or (
            else_type is not None and not isinstance(else_type, ir.TensorType)
        ):
            if then_type is not None and else_type is not None and then_type != else_type:
                raise _context.OpUsageError(
                    node,
                    f"If output {i}: type mismatch between branches: "
                    f"then={then_type}, else={else_type}",
                )
            ctx.set_type(output, then_type if then_type is not None else else_type)  # type: ignore[arg-type]
            continue

        # Determine dtype: prefer then-branch, fall back to else-branch
        then_dtype = then_out.dtype if then_out is not None else None
        else_dtype = else_out.dtype if else_out is not None else None
        if then_dtype is not None and else_dtype is not None and then_dtype != else_dtype:
            raise _context.OpUsageError(
                node,
                f"If output {i}: dtype mismatch between branches: "
                f"then={then_dtype}, else={else_dtype}",
            )
        dtype = then_dtype if then_dtype is not None else else_dtype

        # Merge shapes from both branches
        then_shape = then_out.shape if then_out is not None else None
        else_shape = else_out.shape if else_out is not None else None

        if then_shape is not None and else_shape is not None:
            merged: ir.Shape | None = _merge_shapes(ctx, node, then_shape, else_shape, i)
        elif then_shape is not None:
            merged = then_shape
        elif else_shape is not None:
            merged = else_shape
        else:
            merged = None

        ctx.set_shape_and_dtype(output, merged, dtype)


@_registry.registry.register("", "Loop", since_version=1)
def infer_loop(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Loop operator.

    Inputs:
      0: max_trip_count (INT64 scalar, optional — may be None/empty)
      1: cond          (BOOL scalar)
      2..N: loop-carried dependency initial values

    Body graph inputs:
      0: iteration_num (INT64 scalar)
      1: condition      (BOOL scalar)
      2..N: loop-carried dependencies

    Body graph outputs:
      0: condition      (BOOL scalar)
      1..N: loop-carried dependencies (updated)
      N+1..: scan outputs

    Node outputs:
      0..N-2: final loop-carried dependencies (= body outputs 1..N)
      N-1..:  scan outputs, each prepended with a trip-count dimension

    Spec: https://onnx.ai/onnx/operators/onnx__Loop.html
    """
    # max_trip_count (input 0) is optional and may be None.
    # cond (input 1) is required.
    if len(node.inputs) < 2:
        raise _context.OpUsageError(
            node, f"Expected at least 2 inputs, got {len(node.inputs)}"
        )
    if node.inputs[1] is None:
        raise _context.OpUsageError(node, "Required input 'cond' (#1) is None")

    body_attr = _context.require_attr(node, "body")

    body_graph = body_attr.as_graph()
    if body_graph is None:
        return

    # NOTE: Subgraph inference is done by the engine before this function
    # is called, so body_graph outputs should already have shapes.

    # Number of loop-carried dependencies (node inputs beyond max_trip_count and cond)
    num_loop_carried = len(node.inputs) - 2

    for i, output in enumerate(node.outputs):
        # Body output index: offset by 1 because body output[0] is the condition
        body_out_idx = i + 1
        if body_out_idx >= len(body_graph.outputs):
            continue

        body_out = body_graph.outputs[body_out_idx]
        dtype = body_out.dtype
        body_shape = body_out.shape

        if i < num_loop_carried:
            # Loop-carried dependency: shape matches body output directly
            body_type = body_out.type
            if body_type is not None and not isinstance(body_type, ir.TensorType):
                ctx.set_type(output, body_type)
            else:
                ctx.set_shape_and_dtype(output, body_shape, dtype)
        else:
            # Scan output: prepend a trip-count dimension to body output shape
            if body_shape is not None:
                trip_dim = ctx.new_symbolic_dim()
                scan_shape = ir.Shape([trip_dim, *body_shape.dims])
            else:
                scan_shape = None
            ctx.set_shape_and_dtype(output, scan_shape, dtype)


@_registry.registry.register("", "Scan", since_version=9)
def infer_scan(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Scan operator (opset >= 9).

    Scan iterates over one axis of each scan input and feeds slices into
    a body subgraph.

    State outputs keep body output shapes.  Scan outputs get a sequence-
    length dimension inserted at the axis specified by scan_output_axes
    (default axis 0).

    Spec: https://onnx.ai/onnx/operators/onnx__Scan.html
    """
    body_attr = node.attributes.get("body")
    if body_attr is None:
        return
    body_graph = body_attr.as_graph()
    if body_graph is None:
        return

    num_scan_inputs_attr = node.attributes.get("num_scan_inputs")
    num_scan_inputs = num_scan_inputs_attr.as_int() if num_scan_inputs_attr is not None else 0
    num_state = len(node.inputs) - num_scan_inputs

    # scan_output_axes: one per scan output, default [0, 0, ...]
    num_scan_outputs = len(node.outputs) - num_state
    output_axes_attr = node.attributes.get("scan_output_axes")
    if output_axes_attr is not None:
        output_axes = list(output_axes_attr.as_ints())
    else:
        output_axes = [0] * num_scan_outputs

    # Try to extract the sequence length from scan inputs
    scan_input_axes_attr = node.attributes.get("scan_input_axes")
    if scan_input_axes_attr is not None:
        input_axes = list(scan_input_axes_attr.as_ints())
    else:
        input_axes = [0] * num_scan_inputs

    scan_len_dim: int | ir.SymbolicDim | None = None
    for idx in range(num_scan_inputs):
        inp = node.inputs[num_state + idx]
        if inp is not None and inp.shape is not None:
            axis = input_axes[idx] if idx < len(input_axes) else 0
            rank = inp.shape.rank()
            if rank is not None and rank > 0:
                if axis < 0:
                    axis += rank
                if 0 <= axis < rank:
                    scan_len_dim = inp.shape[axis]
                    break

    if scan_len_dim is None:
        scan_len_dim = ctx.new_symbolic_dim()

    for i, output in enumerate(node.outputs):
        if i >= len(body_graph.outputs):
            continue
        body_out = body_graph.outputs[i]
        if i < num_state:
            # State output: same shape as body state output
            ctx.set_shape_and_dtype(output, body_out.shape, body_out.dtype)
        else:
            # Scan output: insert sequence-length dim at the output axis
            scan_out_idx = i - num_state
            axis = output_axes[scan_out_idx] if scan_out_idx < len(output_axes) else 0
            if body_out.shape is not None:
                body_dims = list(body_out.shape.dims)
                output_rank = len(body_dims) + 1
                if axis < 0:
                    axis += output_rank
                body_dims.insert(axis, scan_len_dim)
                out_shape = ir.Shape(body_dims)
            else:
                out_shape = None
            ctx.set_shape_and_dtype(output, out_shape, body_out.dtype)


@_registry.registry.register("", "Scan", since_version=8)
def infer_scan_v8(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Scan operator (opset 8).

    Scan v8 has an extra ``sequence_lens`` input at position 0 and includes
    a batch dimension.

    Spec: https://onnx.ai/onnx/operators/onnx__Scan.html (opset 8)
    """
    body_attr = node.attributes.get("body")
    if body_attr is None:
        return
    body_graph = body_attr.as_graph()
    if body_graph is None:
        return

    num_scan_inputs_attr = node.attributes.get("num_scan_inputs")
    num_scan_inputs = num_scan_inputs_attr.as_int() if num_scan_inputs_attr is not None else 0
    # v8: input[0] is sequence_lens (optional), rest are state + scan
    num_state = len(node.inputs) - 1 - num_scan_inputs

    # Extract batch dimension from a state input (input[1])
    batch_dim: int | ir.SymbolicDim | None = None
    if num_state > 0 and node.inputs[1] is not None:
        inp: ir.Value | None = node.inputs[1]
        if (
            inp is not None
            and inp.shape is not None
            and inp.shape.rank() is not None
            and inp.shape.rank() > 0
        ):
            batch_dim = inp.shape[0]
    if batch_dim is None:
        batch_dim = ctx.new_symbolic_dim()

    # Extract scan length from first scan input (has shape [batch, seq_len, ...])
    scan_len_dim: int | ir.SymbolicDim | None = None
    for idx in range(num_scan_inputs):
        scan_inp: ir.Value | None = node.inputs[1 + num_state + idx]
        if scan_inp is not None and scan_inp.shape is not None:
            rank = scan_inp.shape.rank()
            if rank is not None and rank >= 2:
                scan_len_dim = scan_inp.shape[1]  # axis 1 after batch
                break
    if scan_len_dim is None:
        scan_len_dim = ctx.new_symbolic_dim()

    for i, output in enumerate(node.outputs):
        if i >= len(body_graph.outputs):
            continue
        body_out = body_graph.outputs[i]
        if i < num_state:
            # State output: [batch] + body output shape
            if body_out.shape is not None:
                out_shape = ir.Shape([batch_dim, *body_out.shape.dims])
            else:
                out_shape = None
            ctx.set_shape_and_dtype(output, out_shape, body_out.dtype)
        else:
            # Scan output: [batch, scan_len] + body output shape
            if body_out.shape is not None:
                out_shape = ir.Shape([batch_dim, scan_len_dim, *body_out.shape.dims])
            else:
                out_shape = None
            ctx.set_shape_and_dtype(output, out_shape, body_out.dtype)
