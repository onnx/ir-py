# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Symbolic shape inference pass."""

from __future__ import annotations

__all__ = [
    "SymbolicShapeInferencePass",
]

from typing import TYPE_CHECKING

import onnx_ir as ir

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context


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
        from onnx_ir.shape_inference._engine import _infer_symbolic_shapes

        modified = _infer_symbolic_shapes(
            model, policy=self.policy, warn_on_missing=self.warn_on_missing
        )

        return ir.passes.PassResult(model, modified)
