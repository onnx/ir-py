# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for control flow operators (If, Loop)."""

from __future__ import annotations

__all__ = [
    "infer_if",
    "infer_loop",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


def _merge_shapes(
    ctx: _context.ShapeInferenceContext,
    shape1: ir.Shape | None,
    shape2: ir.Shape | None,
) -> ir.Shape | None:
    """Merge two shapes from If branches into a compatible output shape.

    For each dimension pair:
    - If both are equal (concrete or symbolic), keep that value.
    - Otherwise, create a new unique symbolic dim.

    Returns None if either shape is None or ranks differ.
    """
    if shape1 is None or shape2 is None:
        return None

    if shape1.rank() != shape2.rank():
        return None

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
            # Dimensions differ or are anonymous; assign a fresh name
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
    (_cond,) = _context.check_inputs(node, "cond")

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

        # Determine dtype: prefer then-branch, fall back to else-branch
        dtype = None
        if then_out is not None:
            dtype = then_out.dtype
        if dtype is None and else_out is not None:
            dtype = else_out.dtype

        # Merge shapes from both branches
        then_shape = then_out.shape if then_out is not None else None
        else_shape = else_out.shape if else_out is not None else None

        if then_shape is not None and else_shape is not None:
            merged = _merge_shapes(ctx, then_shape, else_shape)
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
            ctx.set_shape_and_dtype(output, body_shape, dtype)
        else:
            # Scan output: prepend a trip-count dimension to body output shape
            if body_shape is not None:
                trip_dim = ctx.new_symbolic_dim("_loop_len")
                scan_shape = ir.Shape([trip_dim, *body_shape.dims])
            else:
                scan_shape = None
            ctx.set_shape_and_dtype(output, scan_shape, dtype)
