# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Symbolic shape inference pass."""

from __future__ import annotations

__all__ = [
    "SymbolicShapeInferencePass",
]

import logging

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

logger = logging.getLogger(__name__)


class SymbolicShapeInferencePass(ir.passes.InPlacePass):
    """Pass that performs symbolic shape inference on the graph.

    This pass traverses the graph in topological order and applies
    registered shape inference functions to each node. Unlike the
    standard ONNX shape inference, this pass:

    - Operates directly on the IR (no serialization)
    - Supports symbolic expressions (e.g., N+1, batch*heads) via SymPy
    - Is extensible via the shape inference registry
    - Supports different merge policies for handling existing shapes

    Example::

        import onnx_ir as ir
        from onnx_ir.passes.common import SymbolicShapeInferencePass

        model = ir.load("model.onnx")
        pass_ = SymbolicShapeInferencePass()
        result = pass_(model)

        # Or use the convenience function
        from onnx_ir.shape_inference import infer_symbolic_shapes
        model = infer_symbolic_shapes(model)
    """

    def __init__(
        self,
        policy: _context.ShapeMergePolicy = "refine",
        warn_on_missing: bool = True,
    ) -> None:
        """Initialize the symbolic shape inference pass.

        Args:
            policy: How to merge inferred shapes with existing shapes.
            warn_on_missing: If True, log warnings for ops without registered
                shape inference.
        """
        # Import ops to trigger registration
        from onnx_ir.shape_inference import _ops  # noqa: F401

        super().__init__()
        self.policy = policy
        self.warn_on_missing = warn_on_missing

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Run shape inference on the model.

        Args:
            model: The model to process.

        Returns:
            PassResult with the model and whether it was modified.
        """
        ctx = _context.ShapeInferenceContext(model.opset_imports, policy=self.policy)
        modified = False

        # Process all graphs (main graph + subgraphs)
        for graph in model.graphs():
            graph_modified = self._process_graph(ctx, graph)
            modified = modified or graph_modified

        return ir.passes.PassResult(model, modified)

    def _process_graph(self, ctx: _context.ShapeInferenceContext, graph: ir.Graph) -> bool:
        """Process a single graph.

        Args:
            ctx: The shape inference context.
            graph: The graph to process.

        Returns:
            True if any shapes were modified.
        """
        modified = False
        warned_ops: set[tuple[str, str]] = set()

        # Traverse nodes in topological order
        for node in graph:
            domain = node.domain or ""
            op_type = node.op_type
            opset_version = ctx.get_opset_version(domain)

            # Look up shape inference function
            infer_func = _registry.registry.get(domain, op_type, version=opset_version)

            if infer_func is not None:
                try:
                    # Track which outputs had shapes and dtypes before
                    old_states: dict[int, tuple[object | None, object | None]] = {}
                    for out in node.outputs:
                        if out is None:
                            continue
                        old_shape = out.shape.copy() if out.shape is not None else None
                        old_dtype = getattr(out, "dtype", None)
                        old_states[id(out)] = (old_shape, old_dtype)

                    # Run inference
                    infer_func(ctx, node)

                    # Check if any shapes or dtypes changed
                    for out in node.outputs:
                        if out is None:
                            continue
                        old_shape, old_dtype = old_states.get(id(out), (None, None))
                        current_dtype = getattr(out, "dtype", None)
                        if out.shape != old_shape or current_dtype != old_dtype:
                            modified = True

                except Exception as e:
                    logger.warning(
                        "Shape inference failed for %s::%s: %s",
                        domain or "ai.onnx",
                        op_type,
                        e,
                    )
            elif self.warn_on_missing:
                key = (domain, op_type)
                if key not in warned_ops:
                    logger.warning(
                        "No shape inference registered for %s::%s",
                        domain or "ai.onnx",
                        op_type,
                    )
                    warned_ops.add(key)

        return modified
