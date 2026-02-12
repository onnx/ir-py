# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Common test infrastructure for op-level shape inference tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


def ts(
    dtype: ir.DataType | None = None,
    shape: Sequence[int | str | None] | None = None,
) -> ir.TypeAndShape:
    """Create a :class:`ir.TypeAndShape` from a dtype and a shape list.

    This is a concise helper for specifying input / expected-output type-and-shape
    in parameterized tests.

    Examples::

        ts(ir.DataType.FLOAT, [3, 4])          # Tensor(FLOAT), Shape([3, 4])
        ts(ir.DataType.FLOAT, ["batch", 128])  # Tensor(FLOAT), Shape([batch, 128])
        ts(ir.DataType.FLOAT)                  # Tensor(FLOAT), shape=None
        ts()                                   # type=None, shape=None

    Args:
        dtype: Element data type.  ``None`` means unset.
        shape: Shape dimensions.  ``None`` means unknown rank (unset).

    Returns:
        An :class:`ir.TypeAndShape` instance.
    """
    type_ = ir.TensorType(dtype) if dtype is not None else None
    shape_ = ir.Shape(shape) if shape is not None else None
    return ir.TypeAndShape(type_, shape_)


def const_value(
    data: Sequence[int] | np.ndarray,
    name: str = "const",
    dtype: np.dtype | type = np.int64,
) -> ir.Value:
    """Create an :class:`ir.Value` backed by a constant tensor.

    Useful for ops that read constant inputs (Reshape shape, Slice starts, etc.).

    Args:
        data: The constant data.
        name: Value name.
        dtype: NumPy dtype for the backing array.

    Returns:
        A :class:`ir.Value` with ``const_value``, ``shape`` and ``type`` set.
    """
    arr = np.array(data, dtype=dtype)
    tensor = ir.Tensor(arr, name=name)
    v = ir.Value(name=name, const_value=tensor, type=ir.TensorType(ir.DataType.INT64))
    v.shape = ir.Shape(list(arr.shape))
    return v


def run_shape_inference(
    domain: str,
    op_type: str,
    inputs: Sequence[ir.TypeAndShape],
    attributes: dict[str, ir.Attr] | None = None,
    *,
    opset_version: int,
    num_outputs: int = 1,
    policy: _context.ShapeMergePolicy = "override",
) -> list[ir.TypeAndShape]:
    """Run the registered shape inference function for an op and return output types/shapes.

    This creates temporary :class:`ir.Value` objects from the *inputs* specs,
    invokes the registered inference function directly (no pass), and returns
    the resulting type-and-shape for each output.

    Args:
        domain: ONNX domain (``""`` for the default domain).
        op_type: Operator type (e.g. ``"Add"``).
        inputs: Per-input :class:`ir.TypeAndShape` specs (use :func:`ts` to build them).
        attributes: Node attributes. ``None`` means no attributes.
        opset_version: Opset version for the default domain.
        num_outputs: Number of outputs to create.
        policy: Shape merge policy for the context.

    Returns:
        A list of :class:`ir.TypeAndShape`, one per output, representing the
        inferred type and shape.
    """
    # Build Value objects from TypeAndShape specs
    input_values: list[ir.Value] = []
    for i, spec in enumerate(inputs):
        v = ir.Value(name=f"input_{i}", shape=spec.shape, type=spec.type)
        input_values.append(v)

    output_values = [ir.Value(name=f"output_{i}") for i in range(num_outputs)]

    node = ir.Node(
        domain,
        op_type,
        inputs=input_values,
        outputs=output_values,
        attributes=attributes or {},
    )

    opset_imports = {domain: opset_version} if domain else {"": opset_version}
    ctx = _context.ShapeInferenceContext(opset_imports, policy=policy)

    # Name anonymous dims on inputs, matching what the engine does
    for v in input_values:
        ctx.name_anonymous_dims(v)

    func = _registry.registry.get(domain, op_type, version=opset_version)
    if func is None:
        raise ValueError(
            f"No shape inference registered for {domain}::{op_type} version {opset_version}"
        )
    func(ctx, node)

    return [ir.TypeAndShape(v.type, v.shape) for v in output_values]


def run_shape_inference_with_values(
    domain: str,
    op_type: str,
    input_values: Sequence[ir.Value | None],
    attributes: dict[str, ir.Attr] | Sequence[ir.Attr] | None = None,
    *,
    opset_version: int,
    num_outputs: int = 1,
    policy: _context.ShapeMergePolicy = "override",
) -> list[ir.TypeAndShape]:
    """Like :func:`run_shape_inference` but accepts pre-built :class:`ir.Value` objects.

    Use this when inputs need ``const_value`` set (e.g. for Reshape, Slice,
    Expand) or when testing error paths with ``None`` (missing optional) inputs.
    """
    output_values = [ir.Value(name=f"output_{i}") for i in range(num_outputs)]

    node = ir.Node(
        domain,
        op_type,
        inputs=list(input_values),
        outputs=output_values,
        attributes=attributes or {},
    )

    opset_imports = {domain: opset_version} if domain else {"": opset_version}
    ctx = _context.ShapeInferenceContext(opset_imports, policy=policy)

    # Name anonymous dims on inputs, matching what the engine does
    for v in list(input_values):
        if v is not None:
            ctx.name_anonymous_dims(v)

    func = _registry.registry.get(domain, op_type, version=opset_version)
    if func is None:
        raise ValueError(
            f"No shape inference registered for {domain}::{op_type} version {opset_version}"
        )
    func(ctx, node)

    return [ir.TypeAndShape(v.type, v.shape) for v in output_values]
