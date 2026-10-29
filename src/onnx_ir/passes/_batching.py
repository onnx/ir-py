# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Private batching rule contract and registry.

The transformation owns its context type, including graph construction and
mapped-axis bookkeeping. A batching rule receives that context and the source
node and updates the context in place. Rule-specific shape and input constraints
belong to the rule, not the shared dispatch machinery. This module is an
implementation home, not a stable plugin API.

Internal extension example (``Context`` and ``batch_scale`` are feature-owned)::

    registry = _BatchingRegistry[Context]()
    registry.register("com.example", "Scale", batch_scale,
                      since_version=1, until_version=2)
    rule = registry.lookup(node, graph.opset_imports)
    rule(context, node)

Pass the imports of the actual enclosing graph or function, not an unrelated
model's imports. A registration only applies to its declared interval; extending
coverage requires a new disjoint interval or a fresh registry. The same lookup
pattern applies independently to ``_autodiff._JvpRegistry`` and
``_autodiff._VjpRegistry``, using their distinct call signatures.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from typing_extensions import Protocol

import onnx_ir as ir
from onnx_ir.passes import _rule_registry

_ContextT = TypeVar("_ContextT")
_ContextT_contra = TypeVar("_ContextT_contra", contravariant=True)


class _BatchingRule(Protocol[_ContextT_contra]):
    def __call__(self, context: _ContextT_contra, node: ir.Node, /) -> None: ...


class _BatchingRegistry(
    _rule_registry._RuleRegistry[_BatchingRule[_ContextT]], Generic[_ContextT]
):
    """Instance-local batching rules with explicit supported opset intervals."""

    _kind = "batching"
