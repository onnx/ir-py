# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Symbolic shape inference for ONNX IR.

This module provides native symbolic shape inference that operates directly
on the IR without serialization overhead. It supports SymPy expressions for
symbolic dimension arithmetic and is extensible via a registry system.

Example::

    import onnx_ir as ir
    from onnx_ir.shape_inference import infer_symbolic_shapes, ShapeMergePolicy

    # Load a model
    model = ir.load("model.onnx")

    # Run shape inference
    model = infer_symbolic_shapes(model)

    # Or with custom policy
    model = infer_symbolic_shapes(model, policy="strict")

Registering custom shape inference::

    from onnx_ir.shape_inference import registry

    @registry.register("com.custom", "MyOp", since_version=1)
    def infer_my_op(ctx, node):
        # Access inputs
        input_shape = node.inputs[0].shape

        # Compute output shape
        output_shape = ir.Shape([...])

        # Set output shape
        ctx.set_shape(node.outputs[0], output_shape)
"""

from __future__ import annotations

__all__ = [
    # Main API
    "infer_symbolic_shapes",
    # Context and policy
    "InvalidOpUsageError",
    "ShapeInferenceContext",
    "ShapeInferenceError",
    "ShapeMergePolicy",
    # Registry
    "OpShapeInferenceRegistry",
    "registry",
    # Utilities
    "broadcast_shapes",
    "check_inputs",
    "require_attr",
]

from typing import TYPE_CHECKING

# Import ops to ensure they are registered (but don't expose publicly)
from onnx_ir.shape_inference import _ops  # noqa: F401
from onnx_ir.shape_inference._broadcast import broadcast_shapes
from onnx_ir.shape_inference._context import (
    InvalidOpUsageError,
    ShapeInferenceContext,
    ShapeInferenceError,
    ShapeMergePolicy,
    check_inputs,
    require_attr,
)
from onnx_ir.shape_inference._registry import OpShapeInferenceRegistry, registry

if TYPE_CHECKING:
    import onnx_ir as ir


def infer_symbolic_shapes(
    model: ir.Model,
    *,
    policy: ShapeMergePolicy = "refine",
    warn_on_missing: bool = True,
) -> ir.Model:
    """Perform symbolic shape inference on the model.

    Convenience function that creates and runs a SymbolicShapeInferencePass.

    Args:
        model: The model to perform shape inference on.
        policy: How to merge inferred shapes with existing shapes.
        warn_on_missing: If True, log warnings for ops without registered
            shape inference.

    Returns:
        The model with shape inference applied (modified in place).

    Example::

        import onnx_ir as ir
        from onnx_ir.shape_inference import infer_symbolic_shapes

        model = ir.load("model.onnx")
        model = infer_symbolic_shapes(model)
    """
    from onnx_ir.passes.common.symbolic_shape_inference import SymbolicShapeInferencePass

    return SymbolicShapeInferencePass(
        policy=policy,
        warn_on_missing=warn_on_missing,
    )(model).model


def __set_module() -> None:
    """Set the module of all functions in this module to this public module."""
    global_dict = globals()
    for name in __all__:
        obj = global_dict[name]
        if hasattr(obj, "__module__"):
            obj.__module__ = __name__


__set_module()
