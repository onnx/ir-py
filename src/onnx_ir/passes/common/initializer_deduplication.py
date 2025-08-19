# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pass for removing duplicated initializer tensors from a graph."""

from __future__ import annotations

__all__ = [
    "DeduplicateInitializersPass",
]


import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


class DeduplicateInitializersPass(ir.passes.InPlacePass):
    """Remove duplicated initializer tensors from the graph.

    This pass detects initializers with identical shape, dtype, and content,
    and replaces all duplicate references with a canonical one.

    .. versionadded:: 0.1.3
    .. versionchanged:: 0.1.7
        This pass now deduplicates initializers in subgraphs as well.
    """

    def __init__(self, size_limit: int = 1024):
        super().__init__()
        self.size_limit = size_limit

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        for graph in model.graphs():
            initializers: dict[tuple[ir.DataType, tuple[int, ...], bytes], ir.Value] = {}
            for initializer in tuple(graph.initializers.values()):
                if initializer.is_graph_input() or initializer.is_graph_output():
                    # Skip graph inputs and outputs
                    logger.warning(
                        "Skipped deduplication of initializer '%s' as it is a graph input or output",
                        initializer.name,
                    )
                    continue

                const_val = initializer.const_value
                if const_val is None:
                    # Skip if initializer has no constant value
                    logger.warning(
                        "Skipped deduplication of initializer '%s' as it has no constant value. This is not expected",
                        initializer.name,
                    )
                    continue

                if const_val.size > self.size_limit:
                    # Skip if the initializer is larger than the size limit
                    logger.debug(
                        "Skipped initializer '%s' as it exceeds the size limit of %d elements",
                        initializer.name,
                        self.size_limit,
                    )
                    continue

                key = (const_val.dtype, tuple(const_val.shape), const_val.tobytes())
                if key in initializers:
                    modified = True
                    initializer_to_keep = initializers[key]  # type: ignore[index]
                    ir.convenience.replace_all_uses_with(initializer, initializer_to_keep)
                    assert initializer.name is not None
                    graph.initializers.pop(initializer.name)
                    logger.info(
                        "Replaced initializer '%s' with existing initializer '%s'",
                        initializer.name,
                        initializer_to_keep.name,
                    )
                else:
                    initializers[key] = initializer  # type: ignore[index]

        return ir.passes.PassResult(model=model, modified=modified)
