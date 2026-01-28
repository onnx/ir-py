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
    model = infer_symbolic_shapes(model, policy=ShapeMergePolicy.STRICT)

    # Using the pass directly
    from onnx_ir.shape_inference import SymbolicShapeInferencePass

    pass_ = SymbolicShapeInferencePass()
    result = pass_(model)

Registering custom shape inference::

    from onnx_ir.shape_inference import registry

    @registry.register("com.custom", "MyOp", versions=1)
    def infer_my_op(ctx, node):
        # Access inputs
        input_shape = node.inputs[0].shape

        # Compute output shape
        output_shape = ir.Shape([...])

        # Set output shape
        ctx.set_shape(node.outputs[0], output_shape)
"""

# Import ops to ensure they are registered (but don't expose publicly)
import onnx_ir
from onnx_ir.shape_inference import _ops  # noqa: F401
from onnx_ir.shape_inference._broadcast import broadcast_shapes
from onnx_ir.shape_inference._context import ShapeInferenceContext, ShapeMergePolicy
from onnx_ir.shape_inference._registry import OpShapeInferenceRegistry, registry

__all__ = [
    # Main API
    "infer_symbolic_shapes",
    # Context and policy
    "ShapeInferenceContext",
    "ShapeMergePolicy",
    # Registry
    "OpShapeInferenceRegistry",
    "registry",
    # Utilities
    "broadcast_shapes",
]


def infer_symbolic_shapes(
    model: onnx_ir.Model,
    *,
    policy: ShapeMergePolicy = ShapeMergePolicy.REFINE,
    warn_on_missing: bool = True,
) -> onnx_ir.Model:
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
        from onnx_ir.passes.common import infer_symbolic_shapes

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
        global_dict[name].__module__ = __name__


__set_module()
