# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Common test infrastructure for op-level shape inference tests."""

from __future__ import annotations

from collections.abc import Sequence

import onnx_ir as ir
from onnx_ir.shape_inference._context import ShapeInferenceContext


def run_shape_inference(
    domain: str,
    op_type: str,
    inputs: Sequence[ir.Value | None],
    outputs: Sequence[ir.Value] | None = None,
    attributes: dict[str, ir.Attr] | None = None,
    opset_version: int = 17,
) -> tuple[ir.Node, ShapeInferenceContext]:
    """Build a node from the given spec and run the registered shape inference on it.

    This creates a minimal model / context so the registered inference function
    can be invoked directly, without going through the full pass.

    Args:
        domain: ONNX domain.
        op_type: Operator type.
        inputs: Input values (may contain ``None`` for optional inputs).
        outputs: Output values. If ``None``, a single output ``"output"`` is created.
        attributes: Node attributes.
        opset_version: The opset version for the default domain.

    Returns:
        A ``(node, context)`` tuple so callers can inspect outputs and context.
    """
    from onnx_ir.shape_inference._registry import registry

    if outputs is None:
        outputs = [ir.Value(name="output")]

    node = ir.Node(
        domain,
        op_type,
        inputs=list(inputs),
        outputs=list(outputs),
        attributes=attributes or {},
    )

    # Build a minimal graph/model so ShapeInferenceContext can resolve the opset
    non_none_inputs = [v for v in inputs if v is not None]
    graph = ir.Graph(
        inputs=list(non_none_inputs),
        outputs=list(outputs),
        nodes=[node],
        opset_imports={domain: opset_version} if domain else {"": opset_version},
    )
    model = ir.Model(graph, ir_version=8)
    ctx = ShapeInferenceContext(model, policy="override")

    # Look up and call the registered inference function
    func = registry.get(domain, op_type, version=opset_version)
    if func is None:
        raise ValueError(
            f"No shape inference registered for {domain}::{op_type} version {opset_version}"
        )
    func(ctx, node)

    return node, ctx
