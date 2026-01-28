# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Symbolic shape inference pass."""

from __future__ import annotations

__all__ = [
    "SymbolicShapeInferencePass",
]

import logging

import onnx_ir as ir

# Import ops to trigger registration
from onnx_ir.shape_inference import ops as _ops  # noqa: F401
from onnx_ir.shape_inference._context import ShapeInferenceContext, ShapeMergePolicy
from onnx_ir.shape_inference._registry import registry

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
        from onnx_ir.passes.common import infer_symbolic_shapes
        model = infer_symbolic_shapes(model)
    """

    def __init__(
        self,
        policy: ShapeMergePolicy = ShapeMergePolicy.REFINE,
        warn_on_missing: bool = True,
    ) -> None:
        """Initialize the symbolic shape inference pass.

        Args:
            policy: How to merge inferred shapes with existing shapes.
            warn_on_missing: If True, log warnings for ops without registered
                shape inference.
        """
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
        ctx = ShapeInferenceContext(model, policy=self.policy)
        modified = False

        # Process all graphs (main graph + subgraphs)
        for graph in model.graphs():
            graph_modified = self._process_graph(ctx, graph)
            modified = modified or graph_modified

        return ir.passes.PassResult(model, modified)

    def _process_graph(self, ctx: ShapeInferenceContext, graph: ir.Graph) -> bool:
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
            infer_func = registry.get(domain, op_type, opset_version)

            if infer_func is not None:
                try:
                    # Track which outputs had shapes before
                    old_shapes = {
                        id(out): out.shape.copy()
                        if out is not None and out.shape is not None
                        else None
                        for out in node.outputs
                        if out is not None
                    }

                    # Run inference
                    infer_func(ctx, node)

                    # Check if any shapes changed
                    for out in node.outputs:
                        if out is None:
                            continue
                        old = old_shapes.get(id(out))
                        if out.shape != old:
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
                    logger.debug(
                        "No shape inference registered for %s::%s",
                        domain or "ai.onnx",
                        op_type,
                    )
                    warned_ops.add(key)

        return modified
