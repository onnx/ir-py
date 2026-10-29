# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Private, independent forward- and reverse-mode rule contracts.

The transformation owns its builder/context and derivative accumulation logic.
JVP rules receive a sparse mapping from input index to tangent and return
tangents in node-output order. VJP rules receive cotangents in node-output order
and requested input indices, and return a sparse input-index contribution map.
The caller is responsible for materializing zero output cotangents, checking
arity, and accumulating contributions across uses.

Registries are separate: registering a JVP never implies VJP support, or vice
versa. Structural inputs and active-path analysis remain transformation-owned.
Nothing in this module is a stable plugin API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Generic, TypeVar

from typing_extensions import Protocol

import onnx_ir as ir
from onnx_ir.passes import _rule_registry

_ContextT = TypeVar("_ContextT")
_ContextT_contra = TypeVar("_ContextT_contra", contravariant=True)


class _JvpRule(Protocol[_ContextT_contra]):
    def __call__(
        self,
        context: _ContextT_contra,
        node: ir.Node,
        tangents: Mapping[int, ir.Value],
        /,
    ) -> Sequence[ir.Value]: ...


class _VjpRule(Protocol[_ContextT_contra]):
    def __call__(
        self,
        context: _ContextT_contra,
        node: ir.Node,
        cotangents: Sequence[ir.Value],
        input_indices: Sequence[int],
        /,
    ) -> Mapping[int, ir.Value]: ...


class _JvpRegistry(_rule_registry._RuleRegistry[_JvpRule[_ContextT]], Generic[_ContextT]):
    """Instance-local JVP rules, independent of reverse-mode coverage."""

    _kind = "JVP"


class _VjpRegistry(_rule_registry._RuleRegistry[_VjpRule[_ContextT]], Generic[_ContextT]):
    """Instance-local VJP rules, independent of forward-mode coverage."""

    _kind = "VJP"
