# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Private operator-rule dispatch shared by batching and autodiff.

Registrations use inclusive, finite version intervals. Both ends are required:
a new opset must not silently acquire a rule validated only for older opsets.
Overlapping registrations (including duplicates) are errors, even when they use
the same callable. There is deliberately no override or process-global registry.
Create a separate registry when experimenting with alternative rules.

Dispatch uses the explicit ``Node.version`` when present, as elsewhere in the
IR; otherwise it requires the caller's enclosing graph/function opset imports.
It never consults the model's imports implicitly, falls back to another domain,
or chooses the nearest supported version across a gap. The standard ONNX domain
aliases ``""`` and ``"ai.onnx"`` are equivalent. Conflicting alias imports are
rejected rather than resolved by dictionary insertion order.

This utility checks dispatch applicability, not operator schemas, shapes, local
function bodies, or rule-specific input constraints. Transformations must check
those preconditions themselves before invoking a rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

import onnx_ir as ir

_RuleT = TypeVar("_RuleT")


def _normalize_domain(domain: str) -> str:
    return "" if domain == "ai.onnx" else domain


def _check_version(version: int) -> None:
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"Opset version must be a positive integer, got {version!r}.")


def _effective_opset(node: ir.Node, opset_imports: Mapping[str, int]) -> int:
    """Resolve the node version without borrowing imports from another scope."""
    domain = _normalize_domain(node.domain)
    if domain == "" and "" in opset_imports and "ai.onnx" in opset_imports:
        _check_version(opset_imports[""])
        _check_version(opset_imports["ai.onnx"])
        if opset_imports[""] != opset_imports["ai.onnx"]:
            raise ValueError("Conflicting opset imports for '' and 'ai.onnx'.")
    if node.version is not None:
        version = node.version
    elif domain in opset_imports:
        version = opset_imports[domain]
    elif domain == "" and "ai.onnx" in opset_imports:
        version = opset_imports["ai.onnx"]
    else:
        raise ValueError(f"No opset import for domain {node.domain!r}.")
    _check_version(version)
    return version


@dataclass(frozen=True)
class _Registration(Generic[_RuleT]):
    since_version: int
    until_version: int
    rule: _RuleT


class _RuleRegistry(Generic[_RuleT]):
    """Typed, instance-local registry; use the mode-specific subclasses.

    ``register`` is a direct registration operation, not a decorator. It rejects
    invalid ranges and any overlap atomically. Adjacent ranges are allowed.
    ``lookup`` returns a rule or raises ``NotImplementedError`` for missing or
    unsupported registrations; invalid version/import information raises
    ``ValueError``. Nonempty node overloads are rejected because a plain operator
    rule cannot safely stand in for a local function implementation.
    """

    _kind = "operator"

    def __init__(self) -> None:
        self._rules: dict[tuple[str, str], list[_Registration[_RuleT]]] = {}

    def register(
        self,
        domain: str,
        op_type: str,
        rule: _RuleT,
        *,
        since_version: int,
        until_version: int,
    ) -> None:
        """Register a callable for exactly the inclusive supported interval."""
        _check_version(since_version)
        _check_version(until_version)
        if since_version > until_version:
            raise ValueError("since_version must not exceed until_version.")
        if not op_type:
            raise ValueError("Operator name must not be empty.")
        if not callable(rule):
            raise TypeError("An operator rule must be callable.")
        key = (_normalize_domain(domain), op_type)
        registrations = self._rules.get(key, [])
        for existing in registrations:
            if (
                since_version <= existing.until_version
                and existing.since_version <= until_version
            ):
                raise ValueError(
                    f"Overlapping {self._kind} rules for {key!r}: "
                    f"[{since_version}, {until_version}] overlaps "
                    f"[{existing.since_version}, {existing.until_version}]. "
                    "Overrides are not supported; use a separate registry."
                )
        self._rules.setdefault(key, []).append(
            _Registration(since_version, until_version, rule)
        )
        self._rules[key].sort(key=lambda entry: entry.since_version)

    def lookup(self, node: ir.Node, opset_imports: Mapping[str, int]) -> _RuleT:
        """Resolve a rule using the imports of the node's enclosing scope."""
        version = _effective_opset(node, opset_imports)
        if node.overload:
            raise NotImplementedError(
                f"No {self._kind} rule for overloaded operator {node.op_identifier()!r}."
            )
        key = (_normalize_domain(node.domain), node.op_type)
        registrations = self._rules.get(key, ())
        for entry in registrations:
            if entry.since_version <= version <= entry.until_version:
                return entry.rule
        supported = ", ".join(
            f"[{entry.since_version}, {entry.until_version}]" for entry in registrations
        )
        raise NotImplementedError(
            f"No {self._kind} rule for {key!r} at opset {version}. "
            f"Registered intervals: {supported or 'none'}."
        )
