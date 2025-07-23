# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pass for removing duplicated initializer tensors from a graph."""

from __future__ import annotations

__all__ = [
    "DeduplicateInitializersPass",
]


import onnx_ir as ir


class DeduplicateInitializersPass(ir.passes.InPlacePass):
    """Remove duplicated initializer tensors from the graph.

    This pass detects initializers with identical shape, dtype, and content,
    and replaces all duplicate references with a canonical one.

    To deduplicate initializers from subgraphs, use :class:`~onnx_ir.passes.common.LiftSubgraphInitializersToMainGraphPass`
    to lift the initializers to the main graph first before running pass.

    .. versionadded:: 0.1.3
    """

    def __init__(self, size_limit: int = 1024):
        super().__init__()
        self.size_limit = size_limit

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        initializers: dict[tuple[ir.DataType, tuple[int, ...], bytes], ir.Value] = {}
        modified = False

        for initializer in tuple(graph.initializers.values()):
            # TODO(justinchuby): Handle subgraphs as well. For now users can lift initializers
            # out from the main graph before running this pass.
            const_val = initializer.const_value
            if const_val is None:
                # Skip if initializer has no constant value
                continue

            if const_val.size > self.size_limit:
                continue

            key = (const_val.dtype, tuple(const_val.shape), const_val.tobytes())
            if key in initializers:
                modified = True
                ir.convenience.replace_all_uses_with(initializer, initializers[key])  # type: ignore[index]
                assert initializer.name is not None
                graph.initializers.pop(initializer.name)
            else:
                initializers[key] = initializer  # type: ignore[index]

        return ir.passes.PassResult(model=model, modified=modified)
