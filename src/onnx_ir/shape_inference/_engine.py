# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference engine.

This module contains the core graph-traversal logic for symbolic shape
inference, decoupled from the ``ir.passes`` framework so that
``infer_symbolic_shapes`` can be called directly without pass-level
error wrapping.
"""

from __future__ import annotations

__all__ = [
    "infer_symbolic_shapes",
]

import logging

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

logger = logging.getLogger(__name__)


def infer_symbolic_shapes(
    model: ir.Model,
    *,
    policy: _context.ShapeMergePolicy = "refine",
    warn_on_missing: bool = True,
) -> ir.Model:
    """Perform symbolic shape inference on the model.

    Traverses every graph in *model* in topological order, applying
    registered shape-inference functions to each node.  The model is
    modified **in place** and also returned for convenience.

    Args:
        model: The model to perform shape inference on.
        policy: How to merge inferred shapes with existing shapes.
        warn_on_missing: If ``True``, log warnings for ops without
            registered shape inference.

    Returns:
        The same *model* object, with shapes updated in place.

    Example::

        import onnx_ir as ir
        from onnx_ir.shape_inference import infer_symbolic_shapes

        model = ir.load("model.onnx")
        model = infer_symbolic_shapes(model)
    """
    _infer_symbolic_shapes(model, policy=policy, warn_on_missing=warn_on_missing)
    return model


def _infer_symbolic_shapes(
    model: ir.Model,
    *,
    policy: _context.ShapeMergePolicy = "refine",
    warn_on_missing: bool = True,
) -> bool:
    """Core implementation that returns whether the model was modified."""
    # Import ops to trigger registration
    from onnx_ir.shape_inference import _ops  # noqa: F401

    ctx = _context.ShapeInferenceContext(model.opset_imports, policy=policy)

    return _process_graph(ctx, model.graph, warn_on_missing=warn_on_missing)


def _process_graph(
    ctx: _context.ShapeInferenceContext,
    graph: ir.Graph,
    *,
    warn_on_missing: bool = True,
) -> bool:
    """Process a single graph.

    Args:
        ctx: The shape inference context.
        graph: The graph to process.
        warn_on_missing: If ``True``, log warnings for ops without
            registered shape inference.

    Returns:
        ``True`` if any shapes were modified.
    """
    modified = False
    warned_ops: set[tuple[str, str]] = set()

    # Assign unique names to any anonymous (None) dims on graph inputs
    for value in graph.inputs:
        if ctx.name_anonymous_dims(value):
            modified = True

    # Traverse nodes in topological order
    for node in graph:
        # Recursively process any subgraphs first (e.g. If/Loop bodies)
        for attr in node.attributes.values():
            if (
                attr is not None
                and isinstance(attr, ir.Attr)
                and attr.type == ir.AttributeType.GRAPH
            ):
                subgraph = attr.as_graph()
                if subgraph is not None:
                    if _process_graph(ctx, subgraph, warn_on_missing=warn_on_missing):
                        modified = True

        domain = node.domain or ""
        op_type = node.op_type
        opset_version = ctx.get_opset_version(domain)

        # Look up shape inference function
        infer_func = _registry.registry.get(domain, op_type, version=opset_version)

        if infer_func is not None:
            try:
                # Name anonymous dims on node inputs before inference
                for inp in node.inputs:
                    if inp is not None:
                        ctx.name_anonymous_dims(inp)

                # Track which outputs had shapes and dtypes before
                old_states: list[tuple[ir.Shape | None, ir.TypeProtocol | None]] = []
                for out in node.outputs:
                    old_states.append((out.shape, out.type))

                # Run inference
                infer_func(ctx, node)

                # Check if any shapes or dtypes changed
                for out, (old_shape, old_type) in zip(node.outputs, old_states):
                    if out.shape != old_shape or out.type != old_type:
                        modified = True

            except Exception as e:
                raise RuntimeError(
                    f"Shape inference failed for {domain or 'ai.onnx'}::{op_type}"
                ) from e
        elif warn_on_missing:
            key = (domain, op_type)
            if key not in warned_ops:
                logger.warning(
                    "No shape inference registered for %s::%s",
                    domain or "ai.onnx",
                    op_type,
                )
                warned_ops.add(key)

    return modified
