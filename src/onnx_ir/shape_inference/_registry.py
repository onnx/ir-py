# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Registry for shape inference functions."""

from __future__ import annotations

__all__ = [
    "OpShapeInferenceRegistry",
    "registry",
]

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx_ir as ir
    from onnx_ir.shape_inference._context import ShapeInferenceContext

logger = logging.getLogger(__name__)

# Type alias for shape inference functions
ShapeInferenceFunc = Callable[["ShapeInferenceContext", "ir.Node"], None]


class OpShapeInferenceRegistry:
    """Registry for operator shape inference functions.

    Supports registration by (domain, op_type) with optional opset version filtering.
    When looking up a function, falls back to the closest lower opset version if
    an exact match is not found.

    Example::

        from onnx_ir.shape_inference import registry

        # Register with decorator
        @registry.register("", "Add", versions=range(7, 14))
        def infer_add_v7(ctx, node):
            ...

        @registry.register("", "Add", versions=14)  # 14 and above
        def infer_add_v14(ctx, node):
            ...

        # Lookup
        func = registry.get("", "Add", version=13)
    """

    def __init__(self) -> None:
        # Exact version registrations: {(domain, op_type): {version: func}}
        # Key 0 is used as wildcard (all versions)
        self._versioned: dict[tuple[str, str], dict[int, ShapeInferenceFunc]] = {}
        # Minimum version registrations: {(domain, op_type): [(min_version, func), ...]}
        # Sorted by min_version descending for efficient lookup
        self._min_versioned: dict[tuple[str, str], list[tuple[int, ShapeInferenceFunc]]] = {}

    def register(
        self,
        domain: str,
        op_type: str,
        versions: range | int | None = None,
    ) -> Callable[[ShapeInferenceFunc], ShapeInferenceFunc]:
        """Register a shape inference function for an operator.

        Can be used as a decorator or called directly.

        Args:
            domain: ONNX domain (e.g., "", "com.microsoft").
            op_type: Operator type (e.g., "Add", "Transpose").
            versions: Opset versions to register for. Can be:
                - None: Register for all versions (stored as version 0)
                - int: Register for this version and all versions above (minimum version)
                - range: Register for a specific range of versions

        Returns:
            A decorator that registers the function.

        Example::

            @registry.register("", "Add", versions=range(7, 14))
            def infer_add_v7(ctx, node):
                ...

            @registry.register("", "Add", versions=14)  # 14 and above
            def infer_add_v14(ctx, node):
                ...
        """

        def decorator(func: ShapeInferenceFunc) -> ShapeInferenceFunc:
            key = (domain, op_type)

            if versions is None:
                # None means all versions - use 0 as a wildcard
                if key not in self._versioned:
                    self._versioned[key] = {}
                self._versioned[key][0] = func
            elif isinstance(versions, int):
                # int means this version and above
                if key not in self._min_versioned:
                    self._min_versioned[key] = []
                self._min_versioned[key].append((versions, func))
                # Keep sorted by min_version descending for efficient lookup
                self._min_versioned[key].sort(key=lambda x: x[0], reverse=True)
            else:
                # range - register for each version in range
                if key not in self._versioned:
                    self._versioned[key] = {}
                for version in versions:
                    self._versioned[key][version] = func

            logger.debug(
                "Registered shape inference for %s::%s (versions=%s)",
                domain or "ai.onnx",
                op_type,
                versions,
            )
            return func

        return decorator

    def get(
        self,
        domain: str,
        op_type: str,
        version: int,
    ) -> ShapeInferenceFunc | None:
        """Get the shape inference function for an operator.

        Args:
            domain: ONNX domain.
            op_type: Operator type.
            version: Opset version to look up.

        Returns:
            The shape inference function, or None if not found.
        """
        key = (domain, op_type)

        # Check exact version registrations first (from range or wildcard)
        if key in self._versioned:
            versions = self._versioned[key]

            # Check for wildcard (version 0)
            if 0 in versions:
                return versions[0]

            # Exact match only for range registrations (no fallback)
            if version in versions:
                return versions[version]

        # Check minimum version registrations (int registration means "this version and above")
        if key in self._min_versioned:
            # List is sorted by min_version descending, so first match is most specific
            for min_ver, func in self._min_versioned[key]:
                if min_ver <= version:
                    return func

        return None

    def has(self, domain: str, op_type: str) -> bool:
        """Check if any shape inference function is registered for an operator."""
        key = (domain, op_type)
        return key in self._versioned or key in self._min_versioned

    def clear(self) -> None:
        """Clear all registered functions (mainly for testing)."""
        self._versioned.clear()
        self._min_versioned.clear()


# Global registry instance
registry = OpShapeInferenceRegistry()
