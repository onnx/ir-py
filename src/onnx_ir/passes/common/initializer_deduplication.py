# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pass for removing duplicated initializer tensors from a graph."""

from __future__ import annotations

__all__ = ["DeduplicateInitializersPass", "DeduplicateHashedInitializersPass"]


import hashlib
import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


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

            if const_val.dtype == ir.DataType.STRING:
                # Skip string initializers as they don't have a bytes representation
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


class DeduplicateHashedInitializersPass(ir.passes.InPlacePass):
    """Remove duplicated initializer tensors (using a hashed method) from the graph.

    This pass detects initializers with identical shape, dtype, and hashed content,
    and replaces all duplicate references with a canonical one.

    This pass should have a lower peak memory usage than :class:`DeduplicateInitializersPass`
    as it does not store the full tensor data in memory, but instead uses a hash of the tensor data.

    .. versionadded:: 0.1.7
    """

    def __init__(self, size_limit: int = 4 * 1024 * 1024 * 1024):
        super().__init__()
        # 4 GB default size limit for deduplication
        self.size_limit = size_limit

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        for graph in model.graphs():
            initializers: dict[tuple[ir.DataType, tuple[int, ...], str], ir.Value] = {}

            for initializer in tuple(graph.initializers.values()):
                const_val = initializer.const_value
                if const_val is None:
                    # Skip if initializer has no constant value
                    continue

                if const_val.size > self.size_limit:
                    continue

                if const_val.dtype == ir.DataType.STRING:
                    # Skip string initializers as they don't have a bytes representation
                    continue

                # Hash tensor data to avoid storing large amounts of data in memory
                hashed = hashlib.sha512()
                tensor_data = const_val.numpy()
                hashed.update(tensor_data)
                tensor_digest = hashed.hexdigest()

                tensor_dims = tuple(const_val.shape.numpy())

                key = (const_val.dtype, tensor_dims, tensor_digest)

                if key in initializers:
                    if initializers[key].const_value.tobytes() != const_val.tobytes():
                        logger.warning(
                            "Initializer deduplication failed: "
                            "hashes match but values differ with values %s and %s",
                            initializers[key],
                            initializer,
                        )
                        continue
                    modified = True
                    ir.convenience.replace_all_uses_with(initializer, initializers[key])  # type: ignore[index]
                    assert initializer.name is not None
                    graph.initializers.pop(initializer.name)
                else:
                    initializers[key] = initializer  # type: ignore[index]

        return ir.passes.PassResult(model=model, modified=modified)
