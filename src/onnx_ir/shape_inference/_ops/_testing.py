# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Common test infrastructure for op-level shape inference tests."""

from __future__ import annotations

from collections.abc import Sequence

import onnx_ir as ir
from onnx_ir.shape_inference._context import ShapeInferenceContext


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


def run_shape_inference(
    domain: str,
    op_type: str,
    inputs: Sequence[ir.TypeAndShape],
    attributes: dict[str, ir.Attr] | None = None,
    *,
    opset_version: int,
    num_outputs: int = 1,
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

    Returns:
        A list of :class:`ir.TypeAndShape`, one per output, representing the
        inferred type and shape.
    """
    from onnx_ir.shape_inference._registry import registry

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
    ctx = ShapeInferenceContext(opset_imports, policy="override")

    func = registry.get(domain, op_type, version=opset_version)
    if func is None:
        raise ValueError(
            f"No shape inference registered for {domain}::{op_type} version {opset_version}"
        )
    func(ctx, node)

    return [ir.TypeAndShape(v.type, v.shape) for v in output_values]
